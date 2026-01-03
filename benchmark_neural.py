# benchmark_neural.py
# Version: 1.0
# Description:
# This script benchmarks the "Neural DTW" baseline using Wav2Vec 2.0.
# Wav2Vec 2.0 is a state-of-the-art Transformer model for audio representation.
# We extract frame-level embeddings from its encoder and run DTW.

import os
import sys
import numpy as np
import librosa
import torch
import torchaudio
from scipy.spatial.distance import cdist
from scipy.signal import chirp
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================
PROGRAM_NAME = "benchmark_neural.py"
VERSION = "1.0"
print(f"[{PROGRAM_NAME}] Version: {VERSION}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {DEVICE}")

print("Loading Wav2Vec 2.0 Model...")
try:
    # Load the base model (pretrained on LibriSpeech)
    # We use the feature extractor part
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model().to(DEVICE)
    model.eval()
    print("✅ Wav2Vec 2.0 Loaded Successfully.")
except Exception as e:
    print(f"❌ Model Load Error: {e}")
    sys.exit(1)

# ==========================================
# 2. FEATURE EXTRACTION
# ==========================================
def get_neural_features(audio, sr):
    """
    Extracts embeddings using Wav2Vec 2.0.
    """
    try:
        # 1. Resample to 16kHz (Required by Wav2Vec2)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # 2. Prepare Input Tensor
        # Wav2Vec expects [Batch, Time]
        inputs = torch.from_numpy(audio).float().unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # Extract features from the Transformer encoder
            # We use extract_features() which returns a list of layer outputs
            features, _ = model.extract_features(inputs)
            
            # Take the last layer (most semantic)
            # Shape: [Batch, Frames, Dim] -> [1, T, 768]
            embeddings = features[-1]
            
        return embeddings.squeeze(0).cpu().numpy()
        
    except Exception as e:
        print(f"Extraction Error: {e}")
        return np.zeros((1, 768))

def run_dtw(y1, y2, sr=22050):
    f1 = get_neural_features(y1, sr)
    f2 = get_neural_features(y2, sr)
    
    # Validation
    if f1.shape[0] < 2 or f2.shape[0] < 2:
        return [], 0
    
    # Cosine Distance
    try:
        cost = cdist(f1, f2, metric='cosine')
        if np.isnan(cost).any():
            cost = np.nan_to_num(cost, nan=2.0)
            
        _, wp = librosa.sequence.dtw(C=cost, subseq=False)
        return wp[::-1], f1.shape[0]
    except Exception:
        return [], 0

def calc_mae(wp, rate, n_frames):
    if len(wp) == 0 or n_frames == 0: return 2000.0
    
    # Calculate Wav2Vec2 Frame Duration
    # Wav2Vec2 has a stride of 320 samples at 16kHz
    # 320 / 16000 = 0.02 seconds (20ms)
    frame_sec = 0.02
    
    errs = []
    for p in wp:
        t_ref = p[0] * frame_sec
        t_query = p[1] * frame_sec
        t_gt = t_ref / rate
        errs.append(abs(t_query - t_gt))
        
    return np.mean(errs) * 1000 # ms

# ==========================================
# 3. BENCHMARK LOOP
# ==========================================
print("\n" + "="*60)
print("STARTING NEURAL BENCHMARK (N=20)")
print("Model: Wav2Vec 2.0 (Transformer)")
print("="*60)
print(f"{'Condition':<15} | {'MAE (ms)':<10}")
print("-" * 30)

results = {'Clean':[], 'Noise':[], 'Pitch':[]}
SAMPLE_RATE = 22050
DURATION = 4.0

for i in range(20):
    # Generate Chirp
    t = np.linspace(0, DURATION, int(DURATION*SAMPLE_RATE))
    f0, f1 = np.random.randint(50,200), np.random.randint(2000,4000)
    y = chirp(t, f0=f0, f1=f1, t1=DURATION)
    
    rate = 1.2
    y_str = librosa.effects.time_stretch(y, rate=rate)
    
    # 1. Clean
    wp, nf = run_dtw(y, y_str, SAMPLE_RATE)
    results['Clean'].append(calc_mae(wp, rate, nf))
    
    # 2. Noise (0dB)
    rms = np.sqrt(np.mean(y_str**2))
    y_noise = y_str + np.random.normal(0, rms, len(y_str))
    wp, nf = run_dtw(y, y_noise, SAMPLE_RATE)
    results['Noise'].append(calc_mae(wp, rate, nf))
    
    # 3. Pitch (-4st)
    y_pitch = librosa.effects.pitch_shift(y_str, sr=SAMPLE_RATE, n_steps=-4)
    wp, nf = run_dtw(y, y_pitch, SAMPLE_RATE)
    results['Pitch'].append(calc_mae(wp, rate, nf))
    
    if i % 5 == 0:
        val = results['Clean'][-1]
        print(f"{i:<5} | Last Clean: {val:.1f} ms")

print("\n" + "="*60)
print("FINAL BASELINE RESULTS: Wav2Vec 2.0")
print(f"Clean MAE       : {np.mean(results['Clean']):.2f} ms")
print(f"Noisy (0dB) MAE : {np.mean(results['Noise']):.2f} ms")
print(f"Pitch (-4st) MAE: {np.mean(results['Pitch']):.2f} ms")
print("="*60)
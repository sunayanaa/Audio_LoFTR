# benchmark_vggish.py
# Version: 1.4 (Fix: Global Variable Scope)
# Description:
# This script benchmarks the VGGish + DTW baseline.
# FIX: Moves 'global model' to the top of the function to resolve SyntaxError.
# FEATURES: Automatic fallback to Mel-Spectrograms if VGGish fails or crashes.

import os
import sys
import subprocess
import numpy as np
import librosa
import torch
import traceback
from scipy.spatial.distance import cdist
from scipy.signal import chirp
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP: INSTALL VGGish
# ==========================================
PROGRAM_NAME = "benchmark_vggish.py"
VERSION = "1.4"
print(f"[{PROGRAM_NAME}] Version: {VERSION}")

def setup_vggish():
    print("⏳ Setting up VGGish (PyTorch)...")
    try:
        if not os.path.exists("torchvggish"):
            subprocess.check_call(["git", "clone", "https://github.com/harritaylor/torchvggish.git"])
        
        if os.path.abspath("torchvggish") not in sys.path:
            sys.path.append(os.path.abspath("torchvggish"))
            
        subprocess.check_call([sys.executable, "-m", "pip", "install", "resampy", "soundfile"])
        print("✅ VGGish setup complete.")
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

setup_vggish()

try:
    from torchvggish import vggish, vggish_input
except ImportError:
    sys.path.append("torchvggish")
    from torchvggish import vggish, vggish_input

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {DEVICE}")

# ==========================================
# 2. MODEL INITIALIZATION
# ==========================================
print("Loading VGGish Model...")
model = None
try:
    weight_urls = {
        'vggish': 'https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth',
        'pca': 'https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish_pca_params-970ea276.pth'
    }
    model = vggish.VGGish(urls=weight_urls, postprocess=False)
    model.eval()
    model.to(DEVICE)
    print("✅ Model loaded.")
except Exception as e:
    print(f"❌ Model Init Error: {e}")
    print("⚠️ Switching to Mel-Spectrogram Fallback Mode.")

# ==========================================
# 3. FEATURE EXTRACTION (With Fallback)
# ==========================================
def get_features(audio, sr):
    """
    Extracts features: Tries VGGish first, falls back to Mel Spec.
    """
    # FIX: Declare global at the top to avoid SyntaxError
    global model

    # 1. Normalize
    if len(audio.shape) > 1: audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    # ----------------------------------------
    # STRATEGY A: VGGish
    # ----------------------------------------
    if model is not None:
        try:
            # Resample for VGGish
            audio_vgg = audio
            if sr != 16000:
                audio_vgg = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # Get Examples
            examples = vggish_input.waveform_to_examples(audio_vgg, 16000)
            
            # Check if Examples is Empty
            if examples.shape[0] == 0:
                raise ValueError("VGGish generated 0 frames (audio too short?)")

            # Forward Pass
            if not isinstance(examples, torch.Tensor):
                examples = torch.from_numpy(examples)
            
            examples = examples.to(DEVICE)
            
            with torch.no_grad():
                embeddings = model(examples)
                
            return embeddings.cpu().numpy()
            
        except Exception as e:
            # ONLY PRINT ERROR ONCE
            if not hasattr(get_features, "has_printed_error"):
                print("\n❌ VGGish Failed. Stack Trace:")
                traceback.print_exc()
                print("⚠️ Switching to Standard Mel-Spectrogram Baseline for remaining samples.")
                get_features.has_printed_error = True
            
            # Disable VGGish for future calls to speed up
            model = None 

    # ----------------------------------------
    # STRATEGY B: Mel-Spectrogram (Standard Baseline)
    # ----------------------------------------
    try:
        n_fft = 2048
        hop_length = 512
        mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128)
        log_mels = librosa.power_to_db(mels, ref=np.max)
        return log_mels.T
    except Exception as e:
        return np.zeros((0, 128))

def run_dtw(y1, y2, sr=22050):
    f1 = get_features(y1, sr)
    f2 = get_features(y2, sr)
    
    # Validation
    if f1.shape[0] < 2 or f2.shape[0] < 2:
        return [], 0, 0
    
    # Cosine Distance
    try:
        cost = cdist(f1, f2, metric='cosine')
        if np.isnan(cost).any():
            cost = np.nan_to_num(cost, nan=2.0)
            
        _, wp = librosa.sequence.dtw(C=cost, subseq=False)
        return wp[::-1], f1.shape[0], f2.shape[0]
    except Exception:
        return [], 0, 0

def calc_mae(wp, rate, n_frames_ref, method_type="VGGish"):
    if len(wp) == 0: return 2000.0 
    
    # Determine Frame Duration
    # VGGish = 0.96s fixed
    # MelSpec = hop_length / sr = 512 / 22050 ~= 0.023s
    
    if method_type == "VGGish" and model is not None:
        frame_sec = 0.96
    else:
        # Mel Spec Fallback duration
        frame_sec = 512 / 22050.0
    
    errs = []
    for p in wp:
        t_ref = p[0] * frame_sec
        t_query = p[1] * frame_sec
        t_gt = t_ref / rate
        errs.append(abs(t_query - t_gt))
        
    return np.mean(errs) * 1000 # ms

# ==========================================
# 4. BENCHMARK LOOP
# ==========================================
print("\n" + "="*60)
print("STARTING BENCHMARK (N=20)")
print("="*60)
print(f"{'Condition':<15} | {'MAE (ms)':<10}")
print("-" * 30)

results = {'Clean':[], 'Noise':[], 'Pitch':[]}
SAMPLE_RATE = 22050
DURATION = 5.0

for i in range(20):
    # Generate Chirp
    t = np.linspace(0, DURATION, int(DURATION*SAMPLE_RATE))
    f0, f1 = np.random.randint(50,200), np.random.randint(2000,4000)
    y = chirp(t, f0=f0, f1=f1, t1=DURATION)
    
    rate = 1.2
    y_str = librosa.effects.time_stretch(y, rate=rate)
    
    # 1. Clean
    wp, nf, _ = run_dtw(y, y_str, SAMPLE_RATE)
    m_type = "VGGish" if model is not None else "Mel"
    results['Clean'].append(calc_mae(wp, rate, nf, m_type))
    
    # 2. Noise (0dB)
    rms = np.sqrt(np.mean(y_str**2))
    y_noise = y_str + np.random.normal(0, rms, len(y_str))
    wp, nf, _ = run_dtw(y, y_noise, SAMPLE_RATE)
    results['Noise'].append(calc_mae(wp, rate, nf, m_type))
    
    # 3. Pitch (-4st)
    y_pitch = librosa.effects.pitch_shift(y_str, sr=SAMPLE_RATE, n_steps=-4)
    wp, nf, _ = run_dtw(y, y_pitch, SAMPLE_RATE)
    results['Pitch'].append(calc_mae(wp, rate, nf, m_type))
    
    if i % 5 == 0:
        val = results['Clean'][-1]
        method_str = "VGGish" if model is not None else "MelSpec"
        print(f"{i:<5} | Last Clean: {val:.1f} ms ({method_str})")

print("\n" + "="*60)
print(f"FINAL BASELINE RESULTS: {'VGGish' if model is not None else 'Mel-Spectrogram'}")
print(f"Clean MAE       : {np.mean(results['Clean']):.2f} ms")
print(f"Noisy (0dB) MAE : {np.mean(results['Noise']):.2f} ms")
print(f"Pitch (-4st) MAE: {np.mean(results['Pitch']):.2f} ms")
print("="*60)
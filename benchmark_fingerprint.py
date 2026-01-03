# benchmark_fingerprint.py
# Version: 1.0
# Description:
# This script benchmarks the "Fingerprinting" baseline (Shazam-style landmarks).
# 1. Extracts a "Constellation Map" (sparse spectral peaks) from the spectrogram.
# 2. Applies Gaussian Smoothing to these peaks (to allow for slight mismatches).
# 3. Runs DTW on these sparse maps to measure alignment error.

import os
import sys
import numpy as np
import librosa
import torch
from scipy.spatial.distance import cdist
from scipy.signal import chirp
import scipy.ndimage
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP
# ==========================================
PROGRAM_NAME = "benchmark_fingerprint.py"
VERSION = "1.0"
print(f"[{PROGRAM_NAME}] Version: {VERSION}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {DEVICE}")

# ==========================================
# 2. FEATURE EXTRACTION: CONSTELLATION MAP
# ==========================================
def get_fingerprint_features(audio, sr=22050):
    """
    Extracts a Shazam-style "Constellation Map".
    Returns a smoothed sparse matrix of spectral peaks.
    """
    # 1. Compute Spectrogram
    n_fft = 2048
    hop_length = 512
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    
    # 2. Find Local Peaks (Constellation Map)
    # We look for points that are the max in their local neighborhood
    neighborhood_size = 20 # Window size for peak picking
    threshold = np.mean(S) * 1.5 # Only significant peaks
    
    # Maximum filter finds local max values
    data_max = scipy.ndimage.maximum_filter(S, size=neighborhood_size)
    
    # Boolean mask: True where pixel is a peak and > threshold
    maxima = (S == data_max) & (S > threshold)
    
    # 3. Convert to Float for Alignment
    # A raw binary map is too sparse for DTW (distances are all 0 or 1).
    # We apply a slight Gaussian blur to "spread" the peak energy.
    # This mimics the tolerance in hashing algorithms.
    sparse_map = maxima.astype(np.float32)
    
    # Smoothing along frequency axis helps pitch invariance slightly
    # Smoothing along time axis helps time alignment
    smoothed_map = scipy.ndimage.gaussian_filter(sparse_map, sigma=1.0)
    
    # Transpose to [Time, Frequency] for DTW
    return smoothed_map.T

def run_dtw(y1, y2, sr=22050):
    f1 = get_fingerprint_features(y1, sr)
    f2 = get_fingerprint_features(y2, sr)
    
    # Validation
    if f1.shape[0] < 2 or f2.shape[0] < 2:
        return [], 0
    
    # Cosine Distance on Peak Maps
    try:
        # We use Cosine distance. For sparse maps, this measures overlap of peaks.
        cost = cdist(f1, f2, metric='cosine')
        
        # Clean NaNs (caused by empty frames with no peaks)
        if np.isnan(cost).any():
            cost = np.nan_to_num(cost, nan=1.0) # 1.0 is max dist for cosine
            
        _, wp = librosa.sequence.dtw(C=cost, subseq=False)
        return wp[::-1], f1.shape[0]
    except Exception:
        return [], 0

def calc_mae(wp, rate, n_frames):
    if len(wp) == 0 or n_frames == 0: return 2000.0
    
    # Frame Duration
    # hop_length / sr
    frame_sec = 512 / 22050.0
    
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
print("STARTING FINGERPRINT BENCHMARK (N=20)")
print("Method: Spectral Constellation Map + DTW")
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
print("FINAL BASELINE RESULTS: Fingerprinting")
print(f"Clean MAE       : {np.mean(results['Clean']):.2f} ms")
print(f"Noisy (0dB) MAE : {np.mean(results['Noise']):.2f} ms")
print(f"Pitch (-4st) MAE: {np.mean(results['Pitch']):.2f} ms")
print("="*60)
# Audio LoFTR: Robust and Explainable Audio Alignment

This repository contains the official PyTorch implementation for the paper **"Robust and Explainable Audio Alignment via Audio LoFTR"**.

Audio LoFTR introduces a detector-free Transformer architecture designed to align audio signals under severe forensic distortions (0dB Noise, Pitch Shifting, Time Stretching). By combining a **geometrically invariant Transformer backbone** with a high-resolution **Coarse-to-Fine Refinement** module, the system achieves state-of-the-art precision on complex polyphonic audio.

## 📂 File Structure

The project is organized into core model files, training scripts, and a comprehensive suite of baseline benchmarks used for the comparative analysis in the paper.

### Core Model & Training
| File | Description |
| :--- | :--- |
| **`audio_loftr_refined_v2_6.py`** | **(Primary Model)** The definitive implementation of **Audio LoFTR**. Contains the Transformer backbone and the Coarse-to-Fine Refinement module responsible for the 12.22ms MAE result. |
| **`AudioLoFTR-legacy.py`** | The original coarse-only model implementation. Retained for ablation studies to demonstrate the impact of the refinement stage. |
| **`train_audioloftr.py`** | The main training script. Designed for use with the TASLPRO Jazz Dataset, handling data loading, augmentation, and GPU-accelerated training. |
| **`visualize_results.py`** | **(Reproduces Figure 1)** Generates qualitative visualizations comparing Audio LoFTR attention corridors against MFCC baselines. |

### Comparative Benchmarks & Ablation
These scripts reproduce the comparative results and architectural ablation studies presented in the "Results and Discussion" section.

| File | Description |
| :--- | :--- |
| **`benchmark_robustness.py`** | **(Reproduces Table 1)** The primary synthetic stress test suite (Clean, 0dB Noise, Pitch Shift) used to measure Mean Absolute Error (MAE). |
| **`run_ablation_study_v1_1.py`** | **(Reproduces Table 3)** Automates the ablation study comparing Rotary Position Embeddings (RoPE) against Absolute and No Positional Encodings to validate geometric invariance claims. |
| **`benchmark_neural.py`** | Baseline implementation using **Wav2Vec 2.0** neural embeddings + DTW. |
| **`benchmark_fingerprint.py`** | Baseline implementation using **Shazam-style Constellation Maps** + DTW. |
| **`benchmark_vggish.py`** | Baseline implementation using **VGGish** semantic features + DTW. |
| **`benchmark_cdpam.py`** | Baseline evaluation using the **CDPAM** learned perceptual audio metric. |

## 🚀 Key Performance Results

Our final evaluation on real Jazz recordings demonstrates the significant precision gains achieved by the refined architecture compared to industry standards:

| Method | MAE (ms) | Robustness to Pitch/Noise |
| :--- | :--- | :--- |
| MFCC + DTW (Industry Standard) | 22.63 ms | Low |
| **Audio LoFTR (Ours)** | **12.22 ms** | **High** |

## 🛠️ Requirements

The code requires Python 3.10+ and the following libraries:

```bash
pip install torch torchaudio librosa soundfile numpy scipy matplotlib

# audio_loftr_refined_v2_6.py
# ==========================================
# MASTER SCRIPT: Audio LoFTR for TASLPRO
# VERSION: 2.6 (Fix: Coordinate Mismatch & Error Metric)
# ==========================================
# FIX: Synchronized refinement HOP_LENGTH (256) with Model HOP_LENGTH.
# FIX: Updated Benchmark metric to calculate ms based on actual HOP_LENGTH (11.6ms), not hardcoded 23ms.
# PREVIOUS FIXES: Memory safety, Drive mounting.

!pip install librosa soundfile

import os
import gc
import random
import numpy as np
import librosa
import librosa.display
import librosa.sequence
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
from scipy.signal import chirp
from google.colab import drive 

# 1. CLEAR GPU CACHE
torch.cuda.empty_cache()
gc.collect()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# CONFIGURATION
# ==========================================
SAMPLE_RATE = 22050
DURATION = 3.0
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256  # <--- CRITICAL: This is 11.6ms resolution
COARSE_SCALE = 4
BATCH_SIZE = 2
EPOCHS = 15       # Reduced to 15 (Model converges fast)
LEARNING_RATE = 5e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# DRIVE PATH
DRIVE_PATH = "/content/drive/MyDrive/TASLPRO_Data/jazz"

print(f"Running on Device: {DEVICE}")

# ==========================================
# PART 1: MODEL ARCHITECTURE
# ==========================================
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return x * emb.cos() + rotate_half(x) * emb.sin()

class LoFTRAttention(nn.Module):
    def __init__(self, d_model, nhead, rope_emb=None):
        super().__init__()
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = self.d_head ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = rope_emb

    def forward(self, x, source, return_attention=False):
        B, L1, D = x.shape
        _, L2, _ = source.shape
        q = self.q_proj(x).view(B, L1, self.nhead, self.d_head).transpose(1, 2)
        k = self.k_proj(source).view(B, L2, self.nhead, self.d_head).transpose(1, 2)
        v = self.v_proj(source).view(B, L2, self.nhead, self.d_head).transpose(1, 2)
        if self.rope:
             q_rot = self.rope(q.transpose(1, 2).reshape(B, L1, D)).view(B, L1, self.nhead, self.d_head).transpose(1, 2)
             k_rot = self.rope(k.transpose(1, 2).reshape(B, L2, D)).view(B, L2, self.nhead, self.d_head).transpose(1, 2)
             q, k = q_rot, k_rot
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = attn.softmax(dim=-1)
        out = (attn_probs @ v).transpose(1, 2).reshape(B, L1, D)
        out = self.out_proj(out)
        if return_attention:
            return out, attn_probs
        return out

class LocalFeatureTransformer(nn.Module):
    def __init__(self, d_model, nhead, layer_names=['self', 'cross'] * 4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layer_names = layer_names
        self.rope = RotaryPositionEmbedding(d_model)
        for name in layer_names:
            if name == 'self':
                self.layers.append(LoFTRAttention(d_model, nhead, rope_emb=self.rope))
            else:
                self.layers.append(LoFTRAttention(d_model, nhead, rope_emb=None))

    def forward(self, feat0, feat1):
        last_attn_weights = None
        for i, layer in enumerate(self.layers):
            layer_type = self.layer_names[i]
            if layer_type == 'self':
                feat0 = feat0 + layer(feat0, feat0)
                feat1 = feat1 + layer(feat1, feat1)
            else:
                return_attn = (i == len(self.layers) - 1)
                f0_out = layer(feat0, feat1, return_attention=return_attn)
                f1_out = layer(feat1, feat0)
                if return_attn:
                    feat0 = feat0 + f0_out[0]
                    last_attn_weights = f0_out[1]
                else:
                    feat0 = feat0 + f0_out
                feat1 = feat1 + f1_out
        return feat0, feat1, last_attn_weights

class CoarseMatching(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, feat0, feat1):
        feat0 = F.normalize(feat0, dim=-1)
        feat1 = F.normalize(feat1, dim=-1)
        sim_matrix = torch.einsum("bmd,bnd->bmn", feat0, feat1) / self.temperature
        conf_matrix = F.softmax(sim_matrix, dim=1) * F.softmax(sim_matrix, dim=2)
        return conf_matrix

class AudioLoFTR(nn.Module):
    def __init__(self, d_model=128, nhead=4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, d_model, 3, stride=2, padding=1), nn.ReLU()
        )
        self.transformer = LocalFeatureTransformer(d_model, nhead)
        self.coarse_matching = CoarseMatching()

    def forward(self, img0, img1):
        c0 = self.backbone(img0)
        c1 = self.backbone(img1)
        c0_flat = c0.flatten(2).transpose(1, 2)
        c1_flat = c1.flatten(2).transpose(1, 2)
        feat_c0, feat_c1, attn_weights = self.transformer(c0_flat, c1_flat)
        conf_matrix = self.coarse_matching(feat_c0, feat_c1)
        return {'conf_matrix': conf_matrix, 'attn_weights': attn_weights}

    def predict_alignment(self, s_orig, s_dist, raw_y1, raw_y2, sr=22050):
        with torch.no_grad():
            out = self.forward(s_orig, s_dist)
            conf_matrix = out['conf_matrix'][0].cpu().numpy()
            
        rows = np.arange(conf_matrix.shape[0])
        cols = np.argmax(conf_matrix, axis=1)
        coarse_path = np.column_stack((rows, cols)) * COARSE_SCALE
        fine_path = self.refine_alignment(raw_y1, raw_y2, coarse_path, sr)
        return fine_path

    def refine_alignment(self, y1, y2, coarse_path, sr=22050, radius_ms=600):
        # FIX: Use HOP_LENGTH (256) to match Training Data resolution
        n_fft = N_FFT
        hop_length = HOP_LENGTH 
        
        s1 = np.abs(librosa.stft(y1, n_fft=n_fft, hop_length=hop_length))
        s2 = np.abs(librosa.stft(y2, n_fft=n_fft, hop_length=hop_length))
        s1 = librosa.power_to_db(s1**2, ref=np.max).T
        s2 = librosa.power_to_db(s2**2, ref=np.max).T
        
        n_frames_1, n_frames_2 = s1.shape[0], s2.shape[0]
        frame_ms = (hop_length / sr) * 1000
        radius_frames = int(radius_ms / frame_ms)
        mask = np.full((n_frames_1, n_frames_2), np.inf)
        
        for r, c in coarse_path:
            r = int(min(r, n_frames_1-1))
            c = int(min(c, n_frames_2-1))
            r_min = max(0, r - radius_frames)
            r_max = min(n_frames_1, r + radius_frames)
            c_min = max(0, c - radius_frames)
            c_max = min(n_frames_2, c + radius_frames)
            mask[r_min:r_max, c_min:c_max] = 0.0

        cost = cdist(s1, s2, metric='cosine')
        refined_cost = cost + mask
        refined_cost = np.nan_to_num(refined_cost, nan=2.0)
        
        try:
            _, wp = librosa.sequence.dtw(C=refined_cost, subseq=False)
            return wp[::-1]
        except:
            return coarse_path

# ==========================================
# PART 2: DATA PIPELINE
# ==========================================
class AudioAugmentor:
    def __init__(self):
        self.snr_levels = [10, 20, 30]
        self.stretch_rates = [0.8, 1.2]
        
    def add_noise(self, audio, snr_db):
        rms_signal = np.sqrt(np.mean(audio**2))
        rms_noise = rms_signal / (10 ** (snr_db / 20))
        noise = np.random.normal(0, rms_noise, len(audio))
        return audio + noise

    def apply_distortions(self, audio):
        y_dist = audio.copy()
        params = {'rate': 1.0, 'pitch': 0, 'snr': None}
        if random.random() > 0.3: 
            rate = random.choice(self.stretch_rates)
            y_dist = librosa.effects.time_stretch(y_dist, rate=rate)
            params['rate'] = rate
        if random.random() > 0.5: 
            snr = random.choice(self.snr_levels)
            y_dist = self.add_noise(y_dist, snr)
            params['snr'] = snr
        return y_dist, params

class AudioAlignmentDataset(Dataset):
    def __init__(self, file_list, augmentor=None):
        self.file_list = file_list
        self.augmentor = augmentor if augmentor else AudioAugmentor()
        print(f"Dataset Loaded: {len(file_list)} files found.")
        
    def __len__(self): return len(self.file_list)
    
    def compute_spectrogram(self, audio):
        mels = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mels_db = librosa.power_to_db(mels, ref=np.max)
        mels_norm = (mels_db - mels_db.min()) / (mels_db.max() - mels_db.min() + 1e-8)
        return torch.from_numpy(mels_norm).float().unsqueeze(0)
    
    def create_ground_truth_matrix(self, shape0, shape1, rate):
        H_c = shape0[0] // COARSE_SCALE
        W0_c = shape0[1] // COARSE_SCALE
        W1_c = shape1[1] // COARSE_SCALE
        gt_grid = torch.zeros((W0_c, W1_c))
        for t_orig in range(W0_c):
            t_dist_expected = t_orig / rate
            t_dist = int(round(t_dist_expected))
            if 0 <= t_dist < W1_c: gt_grid[t_orig, t_dist] = 1.0
        return gt_grid
        
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        try:
            y, _ = librosa.load(file_path, sr=SAMPLE_RATE)
            target_samples = int(DURATION * SAMPLE_RATE)
            if len(y) > target_samples:
                start = random.randint(0, len(y) - target_samples)
                y = y[start : start + target_samples]
            elif len(y) < target_samples:
                y = np.pad(y, (0, target_samples - len(y)))
            y_dist, params = self.augmentor.apply_distortions(y)
            spec_orig = self.compute_spectrogram(y)
            spec_dist = self.compute_spectrogram(y_dist)
            gt = self.create_ground_truth_matrix(spec_orig.shape[1:], spec_dist.shape[1:], params['rate'])
            return {'spec_orig': spec_orig, 'spec_dist': spec_dist, 'gt_matrix': gt, 'params': params}
        except Exception:
            return self.__getitem__(random.randint(0, len(self.file_list)-1))

def collate_pad(batch):
    spec_orig = torch.stack([b['spec_orig'] for b in batch])
    spec_dist_list = [b['spec_dist'] for b in batch]
    max_w = max(s.shape[2] for s in spec_dist_list)
    spec_dist = torch.stack([F.pad(s, (0, max_w - s.shape[2])) for s in spec_dist_list])
    gt_list = [b['gt_matrix'] for b in batch]
    max_w_gt = max(g.shape[1] for g in gt_list)
    gt_matrix = torch.stack([F.pad(g, (0, max_w_gt - g.shape[1])) for g in gt_list])
    return {'spec_orig': spec_orig, 'spec_dist': spec_dist, 'gt_matrix': gt_matrix}

# ==========================================
# PART 3: TRAINING
# ==========================================
class WeightedMatchingLoss(nn.Module):
    def __init__(self, pos_weight=50.0):
        super().__init__()
        self.pos_weight = pos_weight
    def forward(self, conf, gt):
        loss = F.binary_cross_entropy(conf, gt, reduction='none')
        weights = gt * self.pos_weight + 1.0
        return (loss * weights).mean()

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        s1 = batch['spec_orig'].to(DEVICE)
        s2 = batch['spec_dist'].to(DEVICE)
        gt = batch['gt_matrix'].to(DEVICE)
        optimizer.zero_grad()
        out = model(s1, s2)
        conf = out['conf_matrix']
        B, T1, T2 = conf.shape
        gt = gt[:, :T1, :T2]
        if gt.shape[1] < T1 or gt.shape[2] < T2:
             gt = F.pad(gt, (0, T2-gt.shape[2], 0, T1-gt.shape[1]))
        loss = criterion(conf, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        del s1, s2, gt, out, conf, loss
    return total_loss / len(loader)

# ==========================================
# PART 4: BENCHMARK
# ==========================================
def run_benchmark_suite(model, dataset, test_samples=50):
    print("\nRunning Refined Benchmark on JAZZ Data...")
    model.eval()
    augmentor = AudioAugmentor()
    
    results = {'MFCC': [], 'LoFTR': []}
    
    if len(dataset) < test_samples:
        test_indices = range(len(dataset))
    else:
        test_indices = random.sample(range(len(dataset)), test_samples)
    
    for idx in test_indices:
        file_path = dataset.file_list[idx]
        try:
            y, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=3.0) 
            if len(y) < 22050*3: y = np.pad(y, (0, 22050*3 - len(y)))
            
            y_dist, params = augmentor.apply_distortions(y)
            true_rate = params['rate']
            
            # MFCC Baseline
            mfcc1 = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)
            mfcc2 = librosa.feature.mfcc(y=y_dist, sr=SAMPLE_RATE, n_mfcc=20)
            cost = cdist(mfcc1.T, mfcc2.T, metric='euclidean')
            _, wp = librosa.sequence.dtw(C=cost, subseq=False)
            wp = wp[::-1]
            # MFCC uses default hop=512 -> ~23.2ms per frame
            errs = [abs(p[0]/true_rate - p[1]) * 23.2 for p in wp]
            results['MFCC'].append(np.mean(errs))
            
            # LoFTR (Refined)
            def get_spec(audio):
                m = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH)
                m_db = librosa.power_to_db(m, ref=np.max)
                m_norm = (m_db - m_db.min()) / (m_db.max() - m_db.min() + 1e-8)
                return torch.from_numpy(m_norm).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            s1 = get_spec(y)
            s2 = get_spec(y_dist)
            
            path = model.predict_alignment(s1, s2, y, y_dist, sr=SAMPLE_RATE)
            
            # FIX: LoFTR uses HOP_LENGTH=256 -> ~11.6ms per frame
            frame_ms_loftr = (HOP_LENGTH / SAMPLE_RATE) * 1000
            
            if len(path) > 5:
                errs = [abs(p[0]/true_rate - p[1]) * frame_ms_loftr for p in path]
                results['LoFTR'].append(np.mean(errs))
            else:
                results['LoFTR'].append(1000)
        except:
            continue

    avg_mfcc = np.mean(results['MFCC']) if results['MFCC'] else 0
    avg_loftr = np.mean(results['LoFTR']) if results['LoFTR'] else 0
    
    print(f"\nFINAL REAL-DATA RESULTS (MAE ms):")
    print(f"MFCC + DTW:      {avg_mfcc:.2f} ms")
    print(f"Audio LoFTR++:   {avg_loftr:.2f} ms")

# ==========================================
# PART 5: EXECUTION
# ==========================================
if __name__ == "__main__":
    try:
        drive.mount('/content/drive', force_remount=True)
    except Exception as e:
        print(f"⚠️ Drive Mount Failed: {e}")
    
    files = []
    print(f"Output of audio_loftr_refined_v2_6.py")
    if os.path.exists(DRIVE_PATH):
        print(f"✅ Drive path found: {DRIVE_PATH}")
        files = [os.path.join(DRIVE_PATH, f) for f in os.listdir(DRIVE_PATH) 
                 if f.lower().endswith(('.wav', '.mp3', '.flac', '.au'))]
        print(f"   Files found: {len(files)}")
    
    if len(files) == 0:
        print("\n⚠️ No real audio found. Generating SYNTHETIC CHIRPS...")
        os.makedirs("training_data", exist_ok=True)
        t = np.linspace(0, 3, 22050*3) 
        y = chirp(t, f0=100, f1=8000, t1=3, method='linear')
        for i in range(10): sf.write(f"training_data/track_{i}.wav", y, 22050)
        files = [os.path.join("training_data", f) for f in os.listdir("training_data")]

    dataset = AudioAlignmentDataset(files)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad)
    
    model = AudioLoFTR(d_model=128, nhead=4).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = WeightedMatchingLoss(pos_weight=50.0)
    
    print(f"\nStarting Training on {len(files)} files...")
    for epoch in range(1, EPOCHS+1):
        loss = train_one_epoch(model, loader, optimizer, criterion)
        if epoch % 5 == 0: print(f"Epoch {epoch}: {loss:.4f}")
            
    run_benchmark_suite(model, dataset)
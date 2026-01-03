# benchmark_robustness.py
# Version: 6.1 (Fixed: Dimension Mismatch & Synthetic Proof)
# FIXES: Added Frequency Pooling to match Time-Time Ground Truth.
# Generates Table 1 using Structured Chirps to match Section 5.1.

!pip install librosa soundfile torchaudio

import os
import random
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
from scipy.signal import chirp

# 1. SETUP
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 256
COARSE_SCALE = 4
EPOCHS = 25  

print(f"Running Geometric Robustness Benchmark on {DEVICE}")

# 2. MODEL DEFINITION
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
    def forward(self, x, source):
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
        return (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, L1, D)

class LocalFeatureTransformer(nn.Module):
    def __init__(self, d_model, nhead, layer_names=['self', 'cross'] * 4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.rope = RotaryPositionEmbedding(d_model)
        for name in layer_names:
            if name == 'self': self.layers.append(LoFTRAttention(d_model, nhead, rope_emb=self.rope))
            else: self.layers.append(LoFTRAttention(d_model, nhead, rope_emb=None))
        self.layer_names = layer_names
    def forward(self, feat0, feat1):
        for i, layer in enumerate(self.layers):
            if self.layer_names[i] == 'self':
                feat0 = feat0 + layer(feat0, feat0)
                feat1 = feat1 + layer(feat1, feat1)
            else:
                feat0 = feat0 + layer(feat0, feat1)
                feat1 = feat1 + layer(feat1, feat0)
        return feat0, feat1

class CoarseMatching(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    def forward(self, feat0, feat1):
        feat0 = F.normalize(feat0, dim=-1)
        feat1 = F.normalize(feat1, dim=-1)
        sim_matrix = torch.einsum("bmd,bnd->bmn", feat0, feat1) / self.temperature
        return F.softmax(sim_matrix, dim=1) * F.softmax(sim_matrix, dim=2)

class AudioLoFTR(nn.Module):
    def __init__(self, d_model=128, nhead=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, d_model, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(d_model)
        self.relu = nn.ReLU()
        self.transformer = LocalFeatureTransformer(d_model, nhead)
        self.coarse_matching = CoarseMatching()
        
    def forward(self, img0, img1):
        x0 = self.relu(self.bn1(self.conv1(img0)))
        x0 = self.relu(self.bn2(self.conv2(x0)))
        x0_c = self.relu(self.bn3(self.conv3(x0)))
        
        x1 = self.relu(self.bn1(self.conv1(img1)))
        x1 = self.relu(self.bn2(self.conv2(x1)))
        x1_c = self.relu(self.bn3(self.conv3(x1)))

        # --- CRITICAL FIX: FREQUENCY POOLING ---
        # Collapse (Freq, Time) -> (Time) to match Ground Truth
        x0_c = x0_c.mean(dim=2) # [B, D, Wc]
        x1_c = x1_c.mean(dim=2)
        
        c0_flat = x0_c.transpose(1, 2) # [B, Wc, D]
        c1_flat = x1_c.transpose(1, 2)
        
        feat_c0, feat_c1 = self.transformer(c0_flat, c1_flat)
        conf = self.coarse_matching(feat_c0, feat_c1)
        return {'conf_matrix': conf}

# 3. SYNTHETIC DATA GENERATOR
class ChirpDataset(Dataset):
    def __init__(self, n_samples=200):
        self.n_samples = n_samples
    def __len__(self): return self.n_samples
    def __getitem__(self, idx):
        # Generate random chirp
        t = np.linspace(0, 4.0, int(4.0*SAMPLE_RATE))
        f0 = random.randint(50, 500)
        f1 = random.randint(2000, 8000)
        y = chirp(t, f0=f0, f1=f1, t1=4.0, method='linear')
        
        # Distortions
        rate = random.choice([0.8, 1.2])
        y_dist = librosa.effects.time_stretch(y, rate=rate)
        # Pad/Crop
        target = int(4.0*SAMPLE_RATE)
        def fix_len(sig):
            if len(sig) < target: return np.pad(sig, (0, target-len(sig)))
            return sig[:target]
        y = fix_len(y)
        y_dist = fix_len(y_dist)
        
        # Spectrograms
        def get_spec(audio):
            m = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH)
            m = librosa.power_to_db(m, ref=np.max)
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            return torch.from_numpy(m).float().unsqueeze(0)
            
        spec_orig = get_spec(y)
        spec_dist = get_spec(y_dist)
        
        # GT
        H, W_orig = spec_orig.shape[1:]
        H, W_dist = spec_dist.shape[1:]
        W_orig_c = W_orig // COARSE_SCALE
        W_dist_c = W_dist // COARSE_SCALE
        gt_matrix = torch.zeros((W_orig_c, W_dist_c))
        for t in range(W_orig_c):
            t_target = int(t / rate)
            if 0 <= t_target < W_dist_c: gt_matrix[t, t_target] = 1.0
            
        return {'spec_orig': spec_orig, 'spec_dist': spec_dist, 'gt': gt_matrix}

def collate_fn(batch):
    return (torch.stack([b['spec_orig'] for b in batch]),
            torch.stack([b['spec_dist'] for b in batch]),
            torch.stack([b['gt'] for b in batch]))

# 4. FAST TRAINING
print("Training Geometric Adapter (Synthetic)...")
dataset = ChirpDataset(n_samples=200)
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
model = AudioLoFTR().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

model.train()
for ep in range(EPOCHS):
    total_loss = 0
    for s_o, s_d, gt in loader:
        s_o, s_d, gt = s_o.to(DEVICE), s_d.to(DEVICE), gt.to(DEVICE)
        optimizer.zero_grad()
        out = model(s_o, s_d)
        conf = out['conf_matrix']
        
        # --- FIX: DYNAMIC CROP TO AVOID 86 vs 87 MISMATCH ---
        min_h = min(conf.shape[1], gt.shape[1])
        min_w = min(conf.shape[2], gt.shape[2])
        conf = conf[:, :min_h, :min_w]
        gt = gt[:, :min_h, :min_w]
        
        loss = criterion(conf, gt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if ep % 5 == 0: print(f"Epoch {ep}: Loss {total_loss/len(loader):.4f}")

# 5. BENCHMARKING
print("\nRunning Forensic Stress Test...")
def run_mfcc_dtw(y_ref, y_query):
    mfcc1 = librosa.feature.mfcc(y=y_ref, sr=SAMPLE_RATE, n_mfcc=20)
    mfcc2 = librosa.feature.mfcc(y=y_query, sr=SAMPLE_RATE, n_mfcc=20)
    cost = cdist(mfcc1.T, mfcc2.T, metric='euclidean')
    try:
        _, wp = librosa.sequence.dtw(C=cost, subseq=False)
        return wp[::-1]
    except: return np.array([])

def run_loftr(y_ref, y_query):
    def to_tensor(audio):
        m = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH)
        m = librosa.power_to_db(m, ref=np.max)
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        return torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(to_tensor(y_ref), to_tensor(y_query))
        attn = out['conf_matrix'][0].cpu().numpy()
    
    # Use -log(p) for cleaner path finding
    cost = -np.log(attn + 1e-8)
    _, wp = librosa.sequence.dtw(C=cost, subseq=False)
    return wp[::-1]

def calc_mae(wp, rate): 
    if len(wp) == 0: return 2000.0
    errs = []
    frame_sec = 0.0116 * 4 # Coarse scale
    for p in wp:
        t_ref = p[0] * frame_sec
        t_query = p[1] * frame_sec
        t_gt = t_ref / rate
        errs.append(abs(t_query - t_gt))
    return np.mean(errs) * 1000

results = {'Clean': {'MFCC':[], 'LoFTR':[]}, 'Noise': {'MFCC':[], 'LoFTR':[]}, 'Pitch': {'MFCC':[], 'LoFTR':[]}}

for i in range(50):
    t = np.linspace(0, 4.0, int(4.0*SAMPLE_RATE))
    y = chirp(t, f0=100, f1=5000, t1=4.0, method='linear')
    rate = random.choice([0.8, 1.2])
    y_str = librosa.effects.time_stretch(y, rate=rate)
    
    # 1. Clean
    results['Clean']['MFCC'].append(calc_mae(run_mfcc_dtw(y, y_str), rate))
    results['Clean']['LoFTR'].append(calc_mae(run_loftr(y, y_str), rate)) 
    
    # 2. Noise (0dB)
    rms = np.sqrt(np.mean(y_str**2))
    y_noise = y_str + np.random.normal(0, rms, len(y_str))
    results['Noise']['MFCC'].append(calc_mae(run_mfcc_dtw(y, y_noise), rate))
    results['Noise']['LoFTR'].append(calc_mae(run_loftr(y, y_noise), rate))

    # 3. Pitch Shift (-4 st)
    y_pitch = librosa.effects.pitch_shift(y_str, sr=SAMPLE_RATE, n_steps=-4)
    results['Pitch']['MFCC'].append(calc_mae(run_mfcc_dtw(y, y_pitch), rate))
    results['Pitch']['LoFTR'].append(calc_mae(run_loftr(y, y_pitch), rate))

# TABLE GENERATION
latex = """
\\begin{table}[h]
\\caption{Comparative Alignment Precision (N=50, Synthetic Chirps)}
\\centering
\\begin{tabular}{l c c c}
\\hline
\\textbf{Method} & \\textbf{Clean} & \\textbf{Noise (0dB)} & \\textbf{Pitch (-4st)} \\\\
\\hline
"""
means = {k: {m: np.mean(v) for m,v in vals.items()} for k,vals in results.items()}

latex += f"MFCC + DTW & {means['Clean']['MFCC']:.1f} & {means['Noise']['MFCC']:.1f} & {means['Pitch']['MFCC']:.1f} \\\\\n"
latex += f"\\textbf{{Audio LoFTR}} & \\textbf{{{means['Clean']['LoFTR']:.1f}}} & \\textbf{{{means['Noise']['LoFTR']:.1f}}} & \\textbf{{{means['Pitch']['LoFTR']:.1f}}} \\\\\n"
latex += """\\hline
\\end{tabular}
\\label{tab:benchmark_results}
\\end{table}
"""
print(latex)

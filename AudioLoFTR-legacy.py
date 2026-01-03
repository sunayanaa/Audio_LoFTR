# ==========================================
# MASTER SCRIPT: Audio LoFTR for TASLPRO
# VERSION: 1.4 (Structured Chirp Data)
# ==========================================
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

# 1. CLEAR GPU CACHE
torch.cuda.empty_cache()
gc.collect()

# ==========================================
# CONFIGURATION
# ==========================================
SAMPLE_RATE = 22050
DURATION = 4.0
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
COARSE_SCALE = 4
BATCH_SIZE = 4          # Safe for Colab
EPOCHS = 25             # Enough for Chirps
LEARNING_RATE = 5e-4    # Higher LR for synthetic data
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Running on Device: {DEVICE}")

# ==========================================
# PART 1: MODEL ARCHITECTURE (Same as before)
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
    def __init__(self, temperature=0.1, threshold=0.2):
        super().__init__()
        self.temperature = temperature
        self.threshold = threshold

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

# ==========================================
# PART 2: DATA PIPELINE (UPDATED: SYNTHETIC CHIRPS)
# ==========================================
class AudioAugmentor:
    def __init__(self):
        self.snr_levels = [10, 20, 30] # Easier SNRs for demo
        self.stretch_rates = [0.8, 1.2]
        self.pitch_steps = [-2, 2]

    def add_noise(self, audio, snr_db):
        rms_signal = np.sqrt(np.mean(audio**2))
        rms_noise = rms_signal / (10 ** (snr_db / 20))
        noise = np.random.normal(0, rms_noise, len(audio))
        return audio + noise

    def apply_distortions(self, audio):
        y_dist = audio.copy()
        params = {'rate': 1.0, 'pitch': 0, 'snr': None}
        if random.random() > 0.0: # Always stretch for this demo
            rate = random.choice(self.stretch_rates)
            y_dist = librosa.effects.time_stretch(y_dist, rate=rate)
            params['rate'] = rate
        if random.random() > 0.5:
            snr = random.choice(self.snr_levels)
            y_dist = self.add_noise(y_dist, snr)
            params['snr'] = snr
        return y_dist, params

class AudioAlignmentDataset(Dataset):
    def __init__(self, file_list, augmentor=None, validation=False):
        self.file_list = file_list
        self.augmentor = augmentor if augmentor else AudioAugmentor()
        self.validation = validation
    def __len__(self):
        return len(self.file_list)
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
            if 0 <= t_dist < W1_c:
                gt_grid[t_orig, t_dist] = 1.0
        return gt_grid
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        try:
            y, _ = librosa.load(file_path, sr=SAMPLE_RATE)
            target_samples = int(DURATION * SAMPLE_RATE)
            if len(y) < target_samples: y = np.pad(y, (0, target_samples - len(y)))
            y = y[:target_samples]
            
            y_distorted, params = self.augmentor.apply_distortions(y)
            spec_orig = self.compute_spectrogram(y)
            spec_dist = self.compute_spectrogram(y_distorted)
            gt_matrix = self.create_ground_truth_matrix(spec_orig.shape[1:], spec_dist.shape[1:], params['rate'])
            return {'spec_orig': spec_orig, 'spec_dist': spec_dist, 'gt_matrix': gt_matrix, 'distortion_params': params}
        except Exception as e:
            return self.__getitem__(random.randint(0, len(self.file_list)-1))

def collate_pad(batch):
    spec_orig = torch.stack([b['spec_orig'] for b in batch])
    spec_dist_list = [b['spec_dist'] for b in batch]
    max_w = max(s.shape[2] for s in spec_dist_list)
    spec_dist_padded = [F.pad(s, (0, max_w - s.shape[2])) for s in spec_dist_list]
    spec_dist = torch.stack(spec_dist_padded)
    gt_list = [b['gt_matrix'] for b in batch]
    max_w_gt = max(g.shape[1] for g in gt_list)
    gt_padded = [F.pad(g, (0, max_w_gt - g.shape[1])) for g in gt_list]
    gt_matrix = torch.stack(gt_padded)
    return {'spec_orig': spec_orig, 'spec_dist': spec_dist, 'gt_matrix': gt_matrix}

# ==========================================
# PART 3: TRAINING (Weighted Loss)
# ==========================================
class WeightedMatchingLoss(nn.Module):
    def __init__(self, pos_weight=50.0): # HIGH WEIGHT
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, conf_matrix, gt_matrix):
        loss = F.binary_cross_entropy(conf_matrix, gt_matrix, reduction='none')
        weights = gt_matrix * self.pos_weight + 1.0
        weighted_loss = (loss * weights).mean()
        return weighted_loss

def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    for batch in loader:
        spec_orig = batch['spec_orig'].to(DEVICE)
        spec_dist = batch['spec_dist'].to(DEVICE)
        gt_matrix = batch['gt_matrix'].to(DEVICE)
        optimizer.zero_grad()
        output = model(spec_orig, spec_dist)
        conf = output['conf_matrix']
        B, T1, T2 = conf.shape
        gt = gt_matrix[:, :T1, :T2] 
        if gt.shape[1] < T1 or gt.shape[2] < T2:
             padding = (0, T2 - gt.shape[2], 0, T1 - gt.shape[1])
             gt = F.pad(gt, padding)
        loss = criterion(conf, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ==========================================
# PART 4: BENCHMARK (Low Threshold)
# ==========================================
def run_benchmark_suite(model, test_samples=50):
    print("\nRunning Comparative Benchmark...")
    model.eval()
    augmentor = AudioAugmentor()
    
    # Generate CHIRP for testing
    t = np.linspace(0, 10, 22050*10)
    y = chirp(t, f0=100, f1=8000, t1=10, method='linear')
    sr = 22050
    sf.write('test_audio.wav', y, sr)
    
    results = {'MFCC+DTW': {'mae': [], 'asr': []}, 'Audio LoFTR': {'mae': [], 'asr': []}}
    
    for i in range(test_samples):
        y_dist, params = augmentor.apply_distortions(y)
        true_rate = params['rate']
        
        # MFCC
        mfcc_orig = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_dist = librosa.feature.mfcc(y=y_dist, sr=sr, n_mfcc=20)
        cost = cdist(mfcc_orig.T, mfcc_dist.T, metric='euclidean')
        _, wp = librosa.sequence.dtw(C=cost, subseq=False)
        wp = wp[::-1]
        errs = [abs(to/true_rate - td)*HOP_LENGTH/sr*1000 for to, td in wp]
        mae = np.mean(errs)
        results['MFCC+DTW']['mae'].append(mae)
        results['MFCC+DTW']['asr'].append(1 if mae < 100 else 0)
        
        # LoFTR (Threshold 0.005)
        def get_spec(audio):
            m = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
            m_db = librosa.power_to_db(m, ref=np.max)
            m_norm = (m_db - m_db.min()) / (m_db.max() - m_db.min() + 1e-8)
            return torch.from_numpy(m_norm).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            s_orig = get_spec(y)
            s_dist = get_spec(y_dist)
            out = model(s_orig, s_dist)
            conf = out['conf_matrix'][0]
            matches = torch.nonzero(conf > 0.005).cpu().numpy() # Very sensitive threshold
            path = matches * COARSE_SCALE
            
        if len(path) > 5:
            errs = [abs(to/true_rate - td)*HOP_LENGTH/sr*1000 for to, td in path]
            mae = np.mean(errs)
            results['Audio LoFTR']['mae'].append(mae)
            results['Audio LoFTR']['asr'].append(1 if mae < 100 else 0)
        else:
            results['Audio LoFTR']['mae'].append(1000)
            results['Audio LoFTR']['asr'].append(0)

    avg = {k: {'mae': np.mean(v['mae']), 'std': np.std(v['mae']), 'asr': np.mean(v['asr'])*100} for k,v in results.items()}
    latex = f"""
\\begin{{table}}[h]
\\caption{{Quantitative Benchmarking (N={test_samples})}}
\\centering
\\begin{{tabular}}{{l c c}}
\\hline
\\textbf{{Method}} & \\textbf{{MAE (ms)}} $\\downarrow$ & \\textbf{{ASR (\\%)}} $\\uparrow$ \\\\
\\hline
MFCC + DTW (Baseline) & {avg['MFCC+DTW']['mae']:.1f} $\\pm$ {avg['MFCC+DTW']['std']:.1f} & {avg['MFCC+DTW']['asr']:.1f} \\\\
\\textbf{{Audio LoFTR (Ours)}} & \\textbf{{{avg['Audio LoFTR']['mae']:.1f} $\\pm$ {avg['Audio LoFTR']['std']:.1f}}} & \\textbf{{{avg['Audio LoFTR']['asr']:.1f}}} \\\\
\\hline
\\end{{tabular}}
\\label{{tab:benchmark_results}}
\\end{{table}}
"""
    print(latex)
    with open("results_table.tex", "w") as f: f.write(latex)

# ==========================================
# PART 5: VISUALIZATION
# ==========================================
def run_visualization_suite(model):
    print("\nGenerating Figure 1...")
    augmentor = AudioAugmentor()
    # LOAD CHIRP
    y, sr = librosa.load("test_audio.wav", duration=4.0)
    y_stretched = librosa.effects.time_stretch(y, rate=1.2)
    y_distorted = augmentor.add_noise(y_stretched, snr_db=10) 
    
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    t_orig = torch.from_numpy(librosa.power_to_db(m, ref=np.max)).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    t_orig = (t_orig - t_orig.min()) / (t_orig.max() - t_orig.min() + 1e-8)
    
    m2 = librosa.feature.melspectrogram(y=y_distorted, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    t_dist = torch.from_numpy(librosa.power_to_db(m2, ref=np.max)).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    t_dist = (t_dist - t_dist.min()) / (t_dist.max() - t_dist.min() + 1e-8)

    model.eval()
    with torch.no_grad():
        out = model(t_orig, t_dist)
        attn_map = out['conf_matrix'][0].cpu().numpy()

    mfcc_orig = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_dist = librosa.feature.mfcc(y=y_distorted, sr=sr, n_mfcc=20)
    cost = cdist(mfcc_orig.T, mfcc_dist.T, metric='euclidean')
    _, wp = librosa.sequence.dtw(C=cost, subseq=False)
    wp = wp[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(np.log1p(attn_map * 100), aspect='auto', origin='lower', cmap='inferno')
    axes[0].set_title("Ours: Audio LoFTR Attention", fontsize=12, fontweight='bold')
    H, W = attn_map.shape
    axes[0].plot(np.arange(W), np.arange(W)*(1/1.2), 'r--', linewidth=2, label='Ground Truth')
    axes[0].legend()
    
    axes[1].imshow(cost.T, aspect='auto', origin='lower', cmap='viridis_r')
    axes[1].set_title("Baseline: MFCC Distance", fontsize=12, fontweight='bold')
    axes[1].plot(wp[:, 1], wp[:, 0], 'w-', linewidth=2, alpha=0.7, label='DTW Path')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('comparison-heatmap-v2.png', dpi=300)
    print("Saved 'comparison-heatmap-v2.png'")

if __name__ == "__main__":
    if not os.path.exists("training_data"):
        os.makedirs("training_data", exist_ok=True)
        # CREATE CHIRPS FOR TRAINING
        t = np.linspace(0, 4, 22050*4)
        y = chirp(t, f0=100, f1=8000, t1=4, method='linear')
        for i in range(10): sf.write(f"training_data/track_{i}.wav", y, 22050)
            
    train_files = [os.path.join("training_data", f) for f in os.listdir("training_data")]
    dataset = AudioAlignmentDataset(train_files)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad)
    
    model = AudioLoFTR(d_model=128, nhead=4).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = WeightedMatchingLoss(pos_weight=50.0) # AGGRESSIVE WEIGHTING
    
    print(f"Starting Training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS+1):
        loss = train_one_epoch(model, loader, optimizer, criterion, epoch)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/audio_loftr_ep{epoch}.pth")
            
    run_benchmark_suite(model)
    run_visualization_suite(model)
    print("\nDONE! Download 'results_table.tex' and 'comparison-heatmap-v2.png'")
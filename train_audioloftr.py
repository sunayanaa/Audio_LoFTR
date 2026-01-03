# ==========================================
# train_audioloftr.py
# MASTER SCRIPT: Audio LoFTR v2.7 (Ultra-Light / Low RAM)
# CHANGES: Batch Size=2 (Safe), GradAccum=4, Memory Cleanup
# ==========================================
!pip install librosa soundfile torchaudio

import os
import gc
import random
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
from google.colab import drive

# 1. MOUNT DRIVE
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

DRIVE_FOLDER = "/content/drive/MyDrive/TASLPRO_Data/jazz" 

# CONFIGURATION (Ultra-Light)
SAMPLE_RATE = 22050
DURATION = 4.0
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
COARSE_SCALE = 4
BATCH_SIZE = 2          # REDUCED TO 2 (Maximum Safety)
GRAD_ACCUM_STEPS = 4    # Accumulate 4 steps = Effective Batch Size 8
EPOCHS = 100            
LEARNING_RATE = 1e-3    
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Running on Device: {DEVICE}")

# 2. CLEAR MEMORY
torch.cuda.empty_cache()
gc.collect()

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
        self.rope = RotaryPositionEmbedding(d_model)
        for name in layer_names:
            if name == 'self':
                self.layers.append(LoFTRAttention(d_model, nhead, rope_emb=self.rope))
            else:
                self.layers.append(LoFTRAttention(d_model, nhead, rope_emb=None))
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
        conf_matrix = F.softmax(sim_matrix, dim=1) * F.softmax(sim_matrix, dim=2)
        return conf_matrix

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

        B, D, Hc, Wc = x0_c.shape
        c0_flat = x0_c.flatten(2).transpose(1, 2)
        c1_flat = x1_c.flatten(2).transpose(1, 2)
        
        feat_c0, feat_c1 = self.transformer(c0_flat, c1_flat)
        conf_matrix = self.coarse_matching(feat_c0, feat_c1)
        return {'conf_matrix': conf_matrix}

# ==========================================
# PART 2: DATA LOADING
# ==========================================
class AudioAugmentor:
    def __init__(self):
        self.snr_levels = [5, 10, 20]
        self.stretch_rates = [0.8, 0.9, 1.1, 1.2]

    def apply_distortions(self, audio):
        y_dist = audio.copy()
        params = {'rate': 1.0}
        if random.random() > 0.3:
            rate = random.choice(self.stretch_rates)
            try:
                y_dist = librosa.effects.time_stretch(y_dist, rate=rate)
                params['rate'] = rate
            except: pass
        snr = random.choice(self.snr_levels)
        rms = np.sqrt(np.mean(y_dist**2))
        noise_rms = rms / (10**(snr/20))
        noise = np.random.normal(0, noise_rms, len(y_dist))
        y_dist = y_dist + noise
        return y_dist, params

class AudioDataset(Dataset):
    def __init__(self, folder_path):
        self.files = []
        if os.path.exists(folder_path):
            for f in os.listdir(folder_path):
                if f.endswith('.au') or f.endswith('.wav'):
                    self.files.append(os.path.join(folder_path, f))
        print(f"Dataset: {len(self.files)} files found.")
        self.augmentor = AudioAugmentor()

    def __len__(self): return len(self.files) * 2

    def __getitem__(self, idx):
        fpath = self.files[idx % len(self.files)]
        try:
            y, sr = librosa.load(fpath, sr=SAMPLE_RATE, duration=4.0)
        except: return self.__getitem__((idx + 1) % len(self.files))

        target_len = int(DURATION * SAMPLE_RATE)
        if len(y) < target_len: y = np.pad(y, (0, target_len - len(y)))
        y = y[:target_len]
        y_dist, params = self.augmentor.apply_distortions(y)
        
        def get_spec(audio):
            m = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH)
            m = librosa.power_to_db(m, ref=np.max)
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            return torch.from_numpy(m).float().unsqueeze(0)

        spec_orig = get_spec(y)
        spec_dist = get_spec(y_dist)
        
        H, W_orig = spec_orig.shape[1:]
        H, W_dist = spec_dist.shape[1:]
        W_orig_c = W_orig // COARSE_SCALE
        W_dist_c = W_dist // COARSE_SCALE
        
        gt_matrix = torch.zeros((W_orig_c, W_dist_c))
        rate = params['rate']
        for t in range(W_orig_c):
            t_target = int(t / rate)
            if 0 <= t_target < W_dist_c:
                gt_matrix[t, t_target] = 1.0
                
        return {'spec_orig': spec_orig, 'spec_dist': spec_dist, 'gt': gt_matrix}

def collate_fn(batch):
    max_w_orig = max([b['spec_orig'].shape[2] for b in batch])
    max_w_dist = max([b['spec_dist'].shape[2] for b in batch])
    max_wc_orig = max([b['gt'].shape[0] for b in batch])
    max_wc_dist = max([b['gt'].shape[1] for b in batch])
    
    spec_origs, spec_dists, gts = [], [], []
    for b in batch:
        p1 = max_w_orig - b['spec_orig'].shape[2]
        spec_origs.append(F.pad(b['spec_orig'], (0, p1)))
        p2 = max_w_dist - b['spec_dist'].shape[2]
        spec_dists.append(F.pad(b['spec_dist'], (0, p2)))
        pg1 = max_wc_dist - b['gt'].shape[1]
        pg0 = max_wc_orig - b['gt'].shape[0]
        gts.append(F.pad(b['gt'], (0, pg1, 0, pg0)))
    return torch.stack(spec_origs), torch.stack(spec_dists), torch.stack(gts)

# ==========================================
# PART 3: TRAINING (Ultra-Light)
# ==========================================
def train():
    dataset = AudioDataset(DRIVE_FOLDER)
    if len(dataset.files) == 0: return None
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    model = AudioLoFTR().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)
    criterion = nn.BCELoss(reduction='none')
    
    print(f"Starting Phase 1.5 Training (100 Epochs, Low RAM)...")
    
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for i, (s_o, s_d, gt) in enumerate(loader):
            s_o, s_d, gt = s_o.to(DEVICE), s_d.to(DEVICE), gt.to(DEVICE)
            out = model(s_o, s_d)
            conf = out['conf_matrix']
            
            if conf.shape[1:] != gt.shape[1:]:
                min_h = min(conf.shape[1], gt.shape[1])
                min_w = min(conf.shape[2], gt.shape[2])
                conf = conf[:, :min_h, :min_w]
                gt = gt[:, :min_h, :min_w]

            loss = criterion(conf, gt)
            weights = gt * 200.0 + 1.0
            loss = (loss * weights).mean()
            loss = loss / GRAD_ACCUM_STEPS 
            
            loss.backward()
            
            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            total_loss += loss.item() * GRAD_ACCUM_STEPS
            
            # MEMORY CLEANUP
            del s_o, s_d, gt, out, conf, loss
            
        scheduler.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss {total_loss/len(loader):.4f} (LR: {scheduler.get_last_lr()[0]:.6f})")
            
    torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/model_phase1_fixed.pth")
    return model

# ==========================================
# PART 4: EVALUATION (With Debug Prints)
# ==========================================
def evaluate(model):
    if model is None: return
    print("Running Evaluation...")
    model.eval()
    
    test_files = [f for f in os.listdir(DRIVE_FOLDER) if f.endswith('.au')]
    if not test_files: return
    y, sr = librosa.load(os.path.join(DRIVE_FOLDER, test_files[0]), duration=4.0)
    y_dist = librosa.effects.time_stretch(y, rate=1.2)
    
    def to_tensor(audio):
        m = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH)
        m = librosa.power_to_db(m, ref=np.max)
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        return torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    t_orig = to_tensor(y)
    t_dist = to_tensor(y_dist)
    
    with torch.no_grad():
        out = model(t_orig, t_dist)
        attn = out['conf_matrix'][0].cpu().numpy()

    print(f"\n--- DEBUG: Attention Map Stats ---")
    print(f"Range: [{attn.min():.4f}, {attn.max():.4f}]")
    print(f"Mean: {attn.mean():.4f}")
        
    mfcc1 = librosa.feature.mfcc(y=y, sr=sr)
    mfcc2 = librosa.feature.mfcc(y=y_dist, sr=sr)
    cost = cdist(mfcc1.T, mfcc2.T, metric='euclidean')
    _, wp = librosa.sequence.dtw(C=cost)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    # Log scale visualization
    plt.imshow(np.log1p(attn * 1000), origin='lower', aspect='auto', cmap='inferno')
    plt.title("Audio LoFTR (Log Scale)")
    plt.subplot(1, 2, 2)
    plt.imshow(cost.T, origin='lower', aspect='auto')
    plt.plot(wp[:, 0], wp[:, 1], 'w')
    plt.title("MFCC + DTW")
    plt.savefig("phase1_results_final.png")
    print("Saved phase1_results_final.png")

if __name__ == "__main__":
    model = train()
    evaluate(model)
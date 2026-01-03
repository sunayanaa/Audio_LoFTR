# run_ablation_study_v1_1.py
# ==========================================
# AUTOMATED ABLATION STUDY: RoPE vs Absolute vs None
# VERSION: 1.1 (Fix: Shape Mismatch via Frequency Pooling)
# ==========================================
# FIX: Added adaptive_avg_pool2d to squeeze frequency dimension to 1.
# FIX: Ensures model output (Time x Time) matches Ground Truth dimensions.

!pip install librosa soundfile

import os
import gc
import random
import math
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from google.colab import drive

# 1. SETUP
torch.cuda.empty_cache()
gc.collect()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DRIVE_PATH = "/content/drive/MyDrive/TASLPRO_Data/jazz"
BATCH_SIZE = 2
EPOCHS = 5

print(f"PROGRAM: run_ablation_study_v1_1.py")
print(f"Running Ablation on Device: {DEVICE}")

# ==========================================
# 2. FLEXIBLE MODEL ARCHITECTURE
# ==========================================
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return x * emb.cos() + rotate_half(x) * emb.sin()

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

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
             q = self.rope(q.transpose(1, 2).reshape(B, L1, D)).view(B, L1, self.nhead, self.d_head).transpose(1, 2)
             k = self.rope(k.transpose(1, 2).reshape(B, L2, D)).view(B, L2, self.nhead, self.d_head).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = attn.softmax(dim=-1)
        out = (attn_probs @ v).transpose(1, 2).reshape(B, L1, D)
        return self.out_proj(out)

class LocalFeatureTransformer(nn.Module):
    def __init__(self, d_model, nhead, pe_type='rope'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.pe_type = pe_type
        
        self.rope = RotaryPositionEmbedding(d_model) if pe_type == 'rope' else None
        self.absolute_pe = SinusoidalPositionEmbedding(d_model) if pe_type == 'absolute' else None

        for _ in range(4): 
            rope_ref = self.rope if pe_type == 'rope' else None
            self.layers.append(LoFTRAttention(d_model, nhead, rope_emb=rope_ref)) 
            self.layers.append(LoFTRAttention(d_model, nhead, rope_emb=None))     

    def forward(self, feat0, feat1):
        if self.pe_type == 'absolute':
            feat0 = self.absolute_pe(feat0)
            feat1 = self.absolute_pe(feat1)
            
        for i, layer in enumerate(self.layers):
            if i % 2 == 0: 
                feat0 = feat0 + layer(feat0, feat0)
                feat1 = feat1 + layer(feat1, feat1)
            else: 
                feat0 = feat0 + layer(feat0, feat1)
                feat1 = feat1 + layer(feat1, feat0)
        return feat0, feat1

class AudioLoFTR_Ablation(nn.Module):
    def __init__(self, pe_type='rope'):
        super().__init__()
        d_model = 128
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, d_model, 3, stride=2, padding=1), nn.ReLU()
        )
        self.transformer = LocalFeatureTransformer(d_model, nhead=4, pe_type=pe_type)

    def forward(self, img0, img1):
        # 1. EXTRACT FEATURES
        c0 = self.backbone(img0) # [B, 128, H, W]
        c1 = self.backbone(img1)
        
        # 2. FREQUENCY POOLING (CRITICAL FIX)
        # Squeeze height (freq) to 1 so we match Time-to-Time
        c0 = F.adaptive_avg_pool2d(c0, (1, None)) # [B, 128, 1, W]
        c1 = F.adaptive_avg_pool2d(c1, (1, None))
        
        # 3. FLATTEN
        c0 = c0.flatten(2).transpose(1, 2) # [B, W, 128]
        c1 = c1.flatten(2).transpose(1, 2)
        
        # 4. TRANSFORMER
        f0, f1 = self.transformer(c0, c1)
        
        # 5. MATCHING
        f0 = F.normalize(f0, dim=-1)
        f1 = F.normalize(f1, dim=-1)
        conf = torch.einsum("bmd,bnd->bmn", f0, f1) * 10.0
        return F.softmax(conf, dim=1) * F.softmax(conf, dim=2)

# ==========================================
# 3. DATA & UTILS
# ==========================================
class SimpleJazzDataset(Dataset):
    def __init__(self, files): self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        try:
            y, _ = librosa.load(self.files[idx], sr=22050, duration=3.0)
            if len(y) < 22050*3: y = np.pad(y, (0, 22050*3 - len(y)))
            
            rate = random.choice([0.8, 1.0, 1.2])
            y_aug = librosa.effects.time_stretch(y, rate=rate)
            
            def get_mel(audio):
                m = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128, hop_length=256)
                return torch.from_numpy(np.log1p(m)).float().unsqueeze(0)
            
            s1, s2 = get_mel(y), get_mel(y_aug)
            
            # Ground Truth Matching dimensions (Downsample by 4)
            W1 = s1.shape[2]
            W2 = s2.shape[2]
            
            W1_c = W1 // 4
            W2_c = W2 // 4
            
            gt = torch.zeros((W1_c, W2_c))
            for t in range(W1_c):
                target = int(round(t / rate))
                if 0 <= target < W2_c: gt[t, target] = 1.0
                
            return s1, s2, gt
        except Exception as e: 
            return self.__getitem__(random.randint(0, len(self.files)-1))

def collate_fn(b):
    # Padding to match batch dimensions
    max_w1 = max([x[0].shape[2] for x in b])
    max_w2 = max([x[1].shape[2] for x in b])
    
    # Pad GT matrix both dims
    max_gt_h = max([x[2].shape[0] for x in b])
    max_gt_w = max([x[2].shape[1] for x in b])
    
    s1 = torch.stack([F.pad(x[0], (0, max_w1 - x[0].shape[2])) for x in b])
    s2 = torch.stack([F.pad(x[1], (0, max_w2 - x[1].shape[2])) for x in b])
    
    gt_list = []
    for x in b:
        g = x[2]
        pad_h = max_gt_h - g.shape[0]
        pad_w = max_gt_w - g.shape[1]
        gt_list.append(F.pad(g, (0, pad_w, 0, pad_h)))
    gt = torch.stack(gt_list)
    
    return s1, s2, gt

# ==========================================
# 4. EXPERIMENT RUNNER
# ==========================================
def run_ablation_experiment(pe_type, train_loader, test_chirp_y):
    print(f"\n--- Training Variant: {pe_type.upper()} ---")
    model = AudioLoFTR_Ablation(pe_type=pe_type).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train
    model.train()
    for ep in range(EPOCHS):
        avg_loss = 0
        for s1, s2, gt in train_loader:
            s1, s2, gt = s1.to(DEVICE), s2.to(DEVICE), gt.to(DEVICE)
            optimizer.zero_grad()
            conf = model(s1, s2)
            
            # Align shapes if pooling/padding caused minimal diff
            min_h = min(conf.shape[1], gt.shape[1])
            min_w = min(conf.shape[2], gt.shape[2])
            conf_c = conf[:, :min_h, :min_w]
            gt_c = gt[:, :min_h, :min_w]
                
            loss = F.binary_cross_entropy(conf_c, gt_c)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            
            del s1, s2, gt, conf, loss
            
        print(f"Ep {ep+1}: Loss={avg_loss/len(train_loader):.4f}")

    # Evaluate on Chirps
    model.eval()
    results = {}
    rates = [1.0, 0.8, 1.2]
    
    base_spec = librosa.feature.melspectrogram(y=test_chirp_y, sr=22050, n_mels=128, hop_length=256)
    base_t = torch.from_numpy(np.log1p(base_spec)).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    for r in rates:
        y_str = librosa.effects.time_stretch(test_chirp_y, rate=r)
        str_spec = librosa.feature.melspectrogram(y=y_str, sr=22050, n_mels=128, hop_length=256)
        str_t = torch.from_numpy(np.log1p(str_spec)).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            conf = model(base_t, str_t)[0].cpu().numpy()
            
        path = np.column_stack(np.where(conf > 0.05)) # Lower threshold for Coarse
        if len(path) < 5: 
            mae = 500.0
        else:
            # Scale=4, Hop=256, SR=22050 -> 1 pixel = 46.4ms
            pixel_ms = (4 * 256 / 22050) * 1000 
            errs = [abs(t_orig/r - t_dist) * pixel_ms for (t_orig, t_dist) in path]
            mae = np.mean(errs)
        results[r] = mae
        
    print(f"Result {pe_type}: {results}")
    return results

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    try:
        drive.mount('/content/drive', force_remount=True)
    except: pass
    
    files = []
    if os.path.exists(DRIVE_PATH):
        files = [os.path.join(DRIVE_PATH, f) for f in os.listdir(DRIVE_PATH) 
                 if f.lower().endswith(('.wav', '.mp3', '.flac', '.au'))]
        print(f"Files found: {len(files)}")
    
    if len(files) == 0:
        print("Using Synthetic Training Data (No Drive Found)")
        t = np.linspace(0, 3, 22050*3)
        y = np.sin(2*np.pi*440*t) 
        sf.write("dummy.wav", y, 22050)
        files = ["dummy.wav"] * 20

    train_files = files[:40] 
    dataset = SimpleJazzDataset(train_files)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    from scipy.signal import chirp
    tc = np.linspace(0, 3, 22050*3)
    test_chirp = chirp(tc, f0=100, f1=8000, t1=3, method='linear')

    # RUN ABLATIONS
    res_rope = run_ablation_experiment('rope', loader, test_chirp)
    res_abs = run_ablation_experiment('absolute', loader, test_chirp)
    res_none = run_ablation_experiment('none', loader, test_chirp)

    print("\n\n" + "="*30)
    print("LATEX TABLE OUTPUT")
    print("="*30)
    latex = f"""
\\begin{{table}}[h]
\\caption{{RoPE Ablation Study (Coarse-Only MAE in ms). N={len(train_files)} Training Samples.}}
\\centering
\\begin{{tabular}}{{l c c c}}
\\hline
\\textbf{{Configuration}} & \\textbf{{r=1.0}} & \\textbf{{r=0.8}} & \\textbf{{r=1.2}} \\\\
\\hline
Full Model (RoPE) & {res_rope[1.0]:.1f} & {res_rope[0.8]:.1f} & {res_rope[1.2]:.1f} \\\\
Absolute Encoding & {res_abs[1.0]:.1f} & {res_abs[0.8]:.1f} & {res_abs[1.2]:.1f} \\\\
No Position Encoding & {res_none[1.0]:.1f} & {res_none[0.8]:.1f} & {res_none[1.2]:.1f} \\\\
\\hline
\\end{{tabular}}
\\label{{tab:ablation_rope}}
\\end{{table}}
"""
    print(latex)
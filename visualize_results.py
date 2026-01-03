# ==========================================
# VISUALIZATION FIX: Scatter Plot Strategy
# visualize_results.py
# ==========================================
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import cdist

# CONFIGURATION
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DRIVE_FOLDER = "/content/drive/MyDrive/TASLPRO_Data/jazz"
CHECKPOINT_PATH = "checkpoints/model_phase1_fixed.pth"

# 1. LOAD MODEL (Re-define class structure briefly to load weights)
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

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

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
        B, D, Hc, Wc = x0_c.shape
        c0_flat = x0_c.flatten(2).transpose(1, 2)
        c1_flat = x1_c.flatten(2).transpose(1, 2)
        feat_c0, feat_c1 = self.transformer(c0_flat, c1_flat)
        return {'conf_matrix': self.coarse_matching(feat_c0, feat_c1)}

# Initialize and Load
model = AudioLoFTR().to(DEVICE)
if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print("✅ Model loaded successfully.")
else:
    print("❌ Checkpoint not found. Run training first.")

# 2. RUN INFERENCE ON TEST FILE
model.eval()
test_files = [f for f in os.listdir(DRIVE_FOLDER) if f.endswith('.au')]
y, sr = librosa.load(os.path.join(DRIVE_FOLDER, test_files[0]), duration=4.0)
y_dist = librosa.effects.time_stretch(y, rate=1.2) # +20% Speed

def to_tensor(audio):
    m = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128, hop_length=256)
    m = librosa.power_to_db(m, ref=np.max)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

t_orig = to_tensor(y)
t_dist = to_tensor(y_dist)

with torch.no_grad():
    out = model(t_orig, t_dist)
    attn = out['conf_matrix'][0].cpu().numpy()

# 3. GENERATE PAPER-READY FIGURE
# Strategy: Thresholding + Scatter Plot
threshold = 0.5  # Only show strong matches
y_coords, x_coords = np.where(attn > threshold)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Audio LoFTR (Matches Only)
ax[0].scatter(x_coords, y_coords, s=2, c='cyan', alpha=0.8, label='LoFTR Matches')
# Draw Ground Truth Line
H, W = attn.shape
ax[0].plot([0, W], [0, H*1.2], 'r--', linewidth=2, label='Ground Truth (1.2x)')
ax[0].set_xlim(0, W)
ax[0].set_ylim(0, H)
ax[0].set_title("Ours: Audio LoFTR (Matches > 0.5)", fontsize=12, fontweight='bold')
ax[0].set_facecolor('black') # Black background for contrast
ax[0].legend()

# Plot 2: Baseline
mfcc1 = librosa.feature.mfcc(y=y, sr=sr)
mfcc2 = librosa.feature.mfcc(y=y_dist, sr=sr)
cost = cdist(mfcc1.T, mfcc2.T, metric='euclidean')
_, wp = librosa.sequence.dtw(C=cost)

ax[1].imshow(cost.T, origin='lower', aspect='auto', cmap='viridis')
ax[1].plot(wp[:, 0], wp[:, 1], 'w-', linewidth=2, label='DTW Path')
ax[1].set_title("Baseline: MFCC + DTW", fontsize=12, fontweight='bold')
ax[1].legend()

plt.tight_layout()
plt.savefig("paper_figure_final.png", dpi=300)
print("Saved paper_figure_final.png - Use this in your LaTeX!")
"""
Multi-Modal Fusion Classifier — 5 Branch

Branch 1 — Angle Transformer Encoder
    입력:  (batch, MAX_LEN, 3)
    구조:  Linear proj → CLS token → Positional Embed → TransformerEncoder
    출력:  (batch, D_MODEL)

Branch 2 — Dense Flow ViT Encoder
    입력:  (batch, K, 3, H, W)   ← _optflow_dense.mp4 프레임
    구조:  Conv2d patch embed → CLS token → TransformerEncoder → K프레임 평균
    출력:  (batch, D_MODEL)

Branch 3 — Sparse Flow ViT Encoder
    입력:  (batch, K, 3, H, W)   ← _optflow_sparse.mp4 프레임
    구조:  Branch 2와 동일 구조, 별도 가중치
    출력:  (batch, D_MODEL)

Branch 4 — Neural ODE Encoder
    입력:  (batch, MAX_LEN, 9)   [위치, 속도, 가속도]
    구조:  GRU → h0 → odeint(rk4) t=0→1
    출력:  (batch, D_MODEL)

Branch 5 — Graph ResNet34 Encoder
    입력:  (batch, 3, H, W)   ← alpha/beta/gamma 그래프 이미지 3채널
    구조:  ResNet34 (pretrained, head 제거) → Linear(512→D_MODEL)
    출력:  (batch, D_MODEL)

Fusion:
    z = concat([z_angle, z_dense, z_sparse, z_ode, z_graph])  (batch, D_MODEL*5=640)
    → Linear(640→256) → GELU → Dropout
    → Linear(256→64)  → GELU → Dropout
    → Linear(64→2)
"""
import torch
import torch.nn as nn
from torchdiffeq import odeint
from torchvision import models

D_MODEL = 128


# ══════════════════════════════════════════════════════════════════════════════
# Branch 1 — Angle Transformer Encoder
# ══════════════════════════════════════════════════════════════════════════════

class AngleTransformerEncoder(nn.Module):
    def __init__(self, input_size=3, d_model=D_MODEL, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1, max_len=400):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed  = nn.Parameter(torch.zeros(1, max_len + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x   = self.input_proj(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + self.pos_embed[:, :x.size(1)]
        x   = self.transformer(x)
        return self.norm(x[:, 0])


# ══════════════════════════════════════════════════════════════════════════════
# Branch 2 & 3 — Flow ViT Encoder (Dense / Sparse 공통 클래스, 별도 인스턴스)
# ══════════════════════════════════════════════════════════════════════════════

class FlowViTEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, d_model=D_MODEL, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1, k_frames=3):
        super().__init__()
        self.k_frames   = k_frames
        num_patches     = (img_size // patch_size) ** 2   # 196

        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _encode_frame(self, img):
        x   = self.patch_embed(img).flatten(2).transpose(1, 2)  # (B, num_patches, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + self.pos_embed
        x   = self.transformer(x)
        return self.norm(x[:, 0])

    def forward(self, frames):
        # frames: (B, K, 3, H, W)
        B, K, C, H, W = frames.shape
        cls = self._encode_frame(frames.view(B * K, C, H, W))   # (B*K, D)
        return cls.view(B, K, -1).mean(dim=1)                    # (B, D)


# ══════════════════════════════════════════════════════════════════════════════
# Branch 4 — Neural ODE Encoder
# ══════════════════════════════════════════════════════════════════════════════

class ODEFunc(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t, h):
        return self.net(h)


class NeuralODEEncoder(nn.Module):
    def __init__(self, input_size=9, d_model=D_MODEL, dropout=0.1):
        super().__init__()
        self.gru      = nn.GRU(input_size, d_model, num_layers=1, batch_first=True)
        self.ode_func = ODEFunc(d_model)
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x):
        _, h0  = self.gru(x)
        h0     = h0.squeeze(0)
        t      = torch.tensor([0.0, 1.0], dtype=x.dtype, device=x.device)
        h_traj = odeint(self.ode_func, h0, t, method='rk4')
        return self.norm(self.dropout(h_traj[-1]))


# ══════════════════════════════════════════════════════════════════════════════
# Branch 5 — Graph ResNet34 Encoder
# ══════════════════════════════════════════════════════════════════════════════

class GraphResNetEncoder(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        # classifier head(fc) 제거 → (B, 512, 1, 1) 출력
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.proj     = nn.Linear(512, d_model)

    def forward(self, x):
        feat = self.backbone(x).flatten(1)   # (B, 512)
        return self.proj(feat)               # (B, D_MODEL)


# ══════════════════════════════════════════════════════════════════════════════
# Full Multi-Modal Classifier — 5 Branch Fusion
# ══════════════════════════════════════════════════════════════════════════════

class MultiModalClassifier(nn.Module):
    def __init__(self, d_model=D_MODEL, dropout=0.1):
        super().__init__()
        self.angle_encoder  = AngleTransformerEncoder(d_model=d_model)
        self.dense_encoder  = FlowViTEncoder(d_model=d_model)   # Branch 2
        self.sparse_encoder = FlowViTEncoder(d_model=d_model)   # Branch 3 (별도 가중치)
        self.ode_encoder    = NeuralODEEncoder(d_model=d_model, dropout=dropout)
        self.graph_encoder  = GraphResNetEncoder(d_model=d_model)  # Branch 5

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 5, 256),   # 640 → 256
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, angles, dense_frames, sparse_frames, kinematics, graph_img):
        z_angle  = self.angle_encoder(angles)           # (B, D)
        z_dense  = self.dense_encoder(dense_frames)     # (B, D)
        z_sparse = self.sparse_encoder(sparse_frames)   # (B, D)
        z_ode    = self.ode_encoder(kinematics)         # (B, D)
        z_graph  = self.graph_encoder(graph_img)        # (B, D)

        z = torch.cat([z_angle, z_dense, z_sparse, z_ode, z_graph], dim=1)  # (B, D*5=640)
        return self.classifier(z)

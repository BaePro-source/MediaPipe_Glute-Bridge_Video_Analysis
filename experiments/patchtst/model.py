"""
PatchTST 기반 분류기 (A Time Series is Worth 64 Words, ICLR 2023)
입력: (batch, seq_len=400, 3)
출력: (batch, 2)

핵심 아이디어:
  1. 시계열을 겹치는 패치(patch)로 분할 — ViT의 이미지 패치와 동일한 개념
  2. Channel Independence(CI): 각 변수(alpha/beta/gamma)를 독립적으로 처리
  3. 패치별 Linear embedding → Transformer Encoder → CLS 토큰으로 분류
  4. 3채널 CLS 출력을 concat → FC → 2-class
"""
import math
import torch
import torch.nn as nn


class PatchTSTClassifier(nn.Module):
    def __init__(
        self,
        seq_len=400,
        n_vars=3,           # alpha, beta, gamma
        patch_len=16,       # 패치 하나의 길이 (프레임 수)
        stride=8,           # 패치 간격 (overlap = patch_len - stride)
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        num_classes=2,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.n_vars    = n_vars

        num_patches = (seq_len - patch_len) // stride + 1  # 49 (seq_len=400 기준)

        # 패치 임베딩 — 채널 공유 (CI 원칙)
        self.patch_embed = nn.Linear(patch_len, d_model)

        # CLS 토큰 + 위치 임베딩 (학습 가능)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3채널 CLS concat → 분류 헤드
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * n_vars),
            nn.Linear(d_model * n_vars, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, seq_len, n_vars)
        batch = x.size(0)
        cls_outs = []

        for i in range(self.n_vars):
            xi = x[:, :, i]  # (batch, seq_len)

            # 패치 분할: unfold → (batch, num_patches, patch_len)
            xi = xi.unfold(dimension=1, size=self.patch_len, step=self.stride)

            # 패치 임베딩: (batch, num_patches, d_model)
            xi = self.patch_embed(xi)

            # CLS 토큰 prepend
            cls = self.cls_token.expand(batch, -1, -1)
            xi  = torch.cat([cls, xi], dim=1)   # (batch, num_patches+1, d_model)

            # 위치 임베딩
            xi = xi + self.pos_embed

            # Transformer Encoder
            xi = self.transformer(xi)           # (batch, num_patches+1, d_model)

            # CLS 토큰 출력만 사용
            cls_outs.append(xi[:, 0])           # (batch, d_model)

        # 채널별 CLS concat → (batch, n_vars * d_model)
        out = torch.cat(cls_outs, dim=1)
        return self.classifier(out)

"""
Transformer Encoder 기반 분류기 (Attention is All You Need)
입력: (batch, seq_len=512, 3)
출력: (batch, 2)

구조:
  1. Linear projection: 3 → d_model
  2. Positional Encoding (sinusoidal)
  3. [CLS] 토큰 prepend
  4. TransformerEncoder (N layers)
  5. CLS 토큰 output → FC → 2-class
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=513):  # 512 + CLS
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position  = torch.arange(0, max_len).unsqueeze(1).float()
        div_term  = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size=3,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj  = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm: 더 안정적인 학습
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, seq_len, 3)
        x = self.input_proj(x)                              # (batch, seq_len, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)     # (batch, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                   # (batch, seq_len+1, d_model)
        x   = self.pos_encoding(x)
        x   = self.transformer(x)                          # (batch, seq_len+1, d_model)
        return self.classifier(x[:, 0])                    # CLS token → (batch, 2)

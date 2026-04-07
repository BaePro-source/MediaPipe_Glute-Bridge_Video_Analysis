"""
Bidirectional LSTM 분류기
입력: (batch, seq_len=512, 3)
출력: (batch, 2)  — best / worst
"""
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),  # *2: bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        # x: (batch, seq_len, 3)
        _, (hn, _) = self.lstm(x)
        # hn shape: (num_layers*2, batch, hidden_size) — 마지막 레이어 양방향
        hn = torch.cat([hn[-2], hn[-1]], dim=1)  # (batch, hidden*2)
        return self.classifier(hn)

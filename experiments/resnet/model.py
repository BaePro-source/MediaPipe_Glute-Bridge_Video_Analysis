"""
ResNet18 기반 분류기
pretrained weights 활용, 마지막 FC 레이어만 교체 (2-class)
"""
import torch.nn as nn
from torchvision import models


def build_resnet18(pretrained=True, num_classes=2, dropout=0.3):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model   = models.resnet18(weights=weights)

    # 마지막 FC 레이어 교체
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model

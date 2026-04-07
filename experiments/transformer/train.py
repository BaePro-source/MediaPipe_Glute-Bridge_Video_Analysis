"""
Transformer 5-Fold 학습 실행
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils import get_kfold_splits
from lstm.dataset import AngleSequenceDataset
from model import TransformerClassifier

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS     = 50
BATCH_SIZE = 8
LR         = 5e-4


def run_fold(fold_idx, train_subjects, val_subjects):
    train_ds = AngleSequenceDataset(train_subjects)
    val_ds   = AngleSequenceDataset(val_subjects, mean=train_ds.mean, std=train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model     = TransformerClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    # Cosine annealing: 학습률을 부드럽게 감소
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Val Acc: {val_acc:.4f}")

    print(f"  Best Val Acc: {best_val_acc:.4f}")
    return best_val_acc


def main():
    print(f"Device: {DEVICE}")
    splits  = get_kfold_splits()
    results = []

    for fold_idx, (train_subjects, val_subjects) in enumerate(splits):
        print(f"\n=== Fold {fold_idx + 1}/5 ===")
        acc = run_fold(fold_idx, train_subjects, val_subjects)
        results.append(acc)

    print("\n" + "="*40)
    print("[Transformer] 5-Fold 결과")
    print(f"각 Fold 정확도: {[round(r, 4) for r in results]}")
    print(f"평균: {np.mean(results):.4f}  표준편차: {np.std(results):.4f}")


if __name__ == '__main__':
    main()

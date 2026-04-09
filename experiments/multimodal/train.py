"""
Multi-Modal Fusion Classifier (4-Branch) — 5-Fold 학습
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from data_utils import get_kfold_splits
from dataset import MultiModalDataset
from model import MultiModalClassifier

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS     = 100
BATCH_SIZE = 4
LR         = 1e-3
SEED       = 30
PATIENCE   = 15

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def run_fold(fold_idx, train_subjects, val_subjects):
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f'fold{fold_idx + 1}_best.pt')

    train_labels  = np.array([l for _, l in train_subjects])
    raw_weights   = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
    class_weights = torch.tensor(raw_weights, dtype=torch.float32).to(DEVICE)
    print(f"  class weight — worst:{raw_weights[0]:.3f}  best:{raw_weights[1]:.3f}")

    train_ds = MultiModalDataset(train_subjects, augment=True)
    val_ds   = MultiModalDataset(
        val_subjects,
        angle_stats=(train_ds.angle_mean, train_ds.angle_std),
        kine_stats =(train_ds.kine_mean,  train_ds.kine_std),
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=False, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    model     = MultiModalClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1  = 0.0
    best_metrics = {}
    patience_cnt = 0

    for epoch in range(EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for angles, dense_frames, sparse_frames, kine, y in train_loader:
            angles, dense_frames, sparse_frames, kine, y = (
                angles.to(DEVICE), dense_frames.to(DEVICE),
                sparse_frames.to(DEVICE), kine.to(DEVICE), y.to(DEVICE)
            )
            optimizer.zero_grad()
            loss = criterion(model(angles, dense_frames, sparse_frames, kine), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)
        train_loss /= len(train_ds)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for angles, dense_frames, sparse_frames, kine, y in val_loader:
                angles, dense_frames, sparse_frames, kine, y = (
                    angles.to(DEVICE), dense_frames.to(DEVICE),
                    sparse_frames.to(DEVICE), kine.to(DEVICE), y.to(DEVICE)
                )
                logits = model(angles, dense_frames, sparse_frames, kine)
                val_loss += criterion(logits, y).item() * y.size(0)
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_labels.extend(y.cpu().tolist())
        val_loss /= len(val_ds)

        val_acc  = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        val_f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        val_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        val_rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_metrics = {'acc': val_acc, 'f1': val_f1,
                            'precision': val_prec, 'recall': val_rec, 'epoch': epoch + 1}
            torch.save(model.state_dict(), ckpt_path)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  [Early Stop] Epoch {epoch + 1} — val F1 {PATIENCE}epoch 개선 없음")
                break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
                  f"(patience {patience_cnt}/{PATIENCE}) | "
                  f"Val Acc: {val_acc:.4f} | Val F1(macro): {val_f1:.4f}")

    print(f"  Best → Epoch:{best_metrics.get('epoch','-')}  "
          f"Acc:{best_metrics.get('acc',0):.4f}  F1:{best_metrics.get('f1',0):.4f}  "
          f"Prec:{best_metrics.get('precision',0):.4f}  Rec:{best_metrics.get('recall',0):.4f}")
    print(f"  체크포인트 저장: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    final_preds, final_labels = [], []
    with torch.no_grad():
        for angles, dense_frames, sparse_frames, kine, y in val_loader:
            angles, dense_frames, sparse_frames, kine = (
                angles.to(DEVICE), dense_frames.to(DEVICE),
                sparse_frames.to(DEVICE), kine.to(DEVICE)
            )
            final_preds.extend(model(angles, dense_frames, sparse_frames, kine)
                                .argmax(dim=1).cpu().tolist())
            final_labels.extend(y.tolist())
    print(classification_report(final_labels, final_preds,
                                target_names=['worst', 'best'], zero_division=0))
    return best_metrics


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}  |  Seed: {SEED}")

    splits  = get_kfold_splits()
    results = []

    for fold_idx, (train_subjects, val_subjects) in enumerate(splits):
        print(f"\n=== Fold {fold_idx + 1}/5 ===")
        results.append(run_fold(fold_idx, train_subjects, val_subjects))

    print("\n" + "=" * 50)
    print("[MultiModal 4-Branch] 5-Fold 결과 요약")
    print("=" * 50)
    for key in ['acc', 'f1', 'precision', 'recall']:
        vals = [r[key] for r in results]
        print(f"  {key:10s}: {[round(v, 4) for v in vals]}  "
              f"평균={np.mean(vals):.4f}  std={np.std(vals):.4f}")


if __name__ == '__main__':
    main()

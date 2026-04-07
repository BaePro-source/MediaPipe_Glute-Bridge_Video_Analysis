"""
LSTM 5-Fold 학습 실행
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
from data_utils import get_kfold_splits, load_angles_csv
from dataset import AngleSequenceDataset, MAX_LEN
from model import LSTMClassifier

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS     = 100
BATCH_SIZE = 8
LR         = 1e-3
SEED       = 42
PATIENCE   = 15   # val F1이 PATIENCE epoch 동안 개선 없으면 조기 종료

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def log_seq_length_stats(subjects, tag: str):
    """피험자별 실제 시퀀스 길이 통계 출력 (패딩 전)."""
    lengths = [load_angles_csv(p).shape[0] for p, _ in subjects]
    arr = np.array(lengths)
    padded = np.sum(arr < MAX_LEN)
    print(f"  [{tag}] seq_len — min:{arr.min()} max:{arr.max()} "
          f"mean:{arr.mean():.1f} std:{arr.std():.1f} | "
          f"MAX_LEN={MAX_LEN}, 패딩 필요:{padded}/{len(lengths)}")


def run_fold(fold_idx, train_subjects, val_subjects):
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f'fold{fold_idx + 1}_best.pt')

    # 시퀀스 길이 통계
    log_seq_length_stats(train_subjects, 'train')
    log_seq_length_stats(val_subjects,   'val  ')

    # class weight: worst(0), best(1) 불균형 보정
    train_labels  = np.array([l for _, l in train_subjects])
    raw_weights   = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
    class_weights = torch.tensor(raw_weights, dtype=torch.float32).to(DEVICE)
    print(f"  class weight — worst:{raw_weights[0]:.3f}  best:{raw_weights[1]:.3f}")

    train_ds = AngleSequenceDataset(train_subjects, augment=True)
    val_ds   = AngleSequenceDataset(val_subjects, mean=train_ds.mean, std=train_ds.std, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model     = LSTMClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1  = 0.0
    best_metrics = {}
    patience_cnt = 0

    for epoch in range(EPOCHS):
        # ── Train ──────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)
        train_loss /= len(train_ds)

        # ── Validate ───────────────────────────────────────
        model.eval()
        val_loss   = 0.0
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                logits = model(X)
                val_loss += criterion(logits, y).item() * y.size(0)
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_labels.extend(y.cpu().tolist())
        val_loss /= len(val_ds)

        val_acc  = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        val_f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        val_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        val_rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        # best model 저장 + early stopping: 둘 다 val F1 기준으로 통일
        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_metrics = {'acc': val_acc, 'f1': val_f1,
                            'precision': val_prec, 'recall': val_rec,
                            'epoch': epoch + 1}
            torch.save(model.state_dict(), ckpt_path)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  [Early Stop] Epoch {epoch + 1} — "
                      f"val F1 {PATIENCE}epoch 개선 없음")
                break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
                  f"(patience {patience_cnt}/{PATIENCE}) | "
                  f"Val Acc: {val_acc:.4f} | Val F1(macro): {val_f1:.4f}")

    print(f"  Best → Epoch:{best_metrics['epoch']}  "
          f"Acc:{best_metrics['acc']:.4f}  F1:{best_metrics['f1']:.4f}  "
          f"Prec:{best_metrics['precision']:.4f}  Rec:{best_metrics['recall']:.4f}")
    print(f"  체크포인트 저장: {ckpt_path}")

    # 최종 classification report (best 모델로)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    final_preds, final_labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            final_preds.extend(model(X).argmax(dim=1).cpu().tolist())
            final_labels.extend(y.cpu().tolist())
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
        metrics = run_fold(fold_idx, train_subjects, val_subjects)
        results.append(metrics)

    print("\n" + "="*40)
    print("[LSTM] 5-Fold 결과")
    for key in ['acc', 'f1', 'precision', 'recall']:
        vals = [r[key] for r in results]
        print(f"  {key:10s}: {[round(v, 4) for v in vals]}  "
              f"평균={np.mean(vals):.4f}  std={np.std(vals):.4f}")


if __name__ == '__main__':
    main()

"""
Branch Ablation Study — Leave-one-out (재학습 없음)

조건 (fold마다):
  all        : 4 branch 모두 사용 (baseline)
  no_angle   : Angle Transformer 제거
  no_dense   : Dense Flow ViT 제거
  no_sparse  : Sparse Flow ViT 제거
  no_ode     : Neural ODE 제거

실행:
  cd experiments
  python multimodal/ablation.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from data_utils import get_kfold_splits
from dataset import MultiModalDataset
from model import MultiModalClassifier

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
BATCH_SIZE = 4

# (zero_angle, zero_dense, zero_sparse, zero_ode)
CONDITIONS = {
    'all'      : (False, False, False, False),
    'no_angle' : (True,  False, False, False),
    'no_dense' : (False, True,  False, False),
    'no_sparse': (False, False, True,  False),
    'no_ode'   : (False, False, False, True),
}


def evaluate(model, loader, zero_angle=False, zero_dense=False,
             zero_sparse=False, zero_ode=False):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for angles, dense_frames, sparse_frames, kine, y in loader:
            angles       = angles.to(DEVICE)
            dense_frames = dense_frames.to(DEVICE)
            sparse_frames= sparse_frames.to(DEVICE)
            kine         = kine.to(DEVICE)

            z_angle  = model.angle_encoder(angles)
            z_dense  = model.dense_encoder(dense_frames)
            z_sparse = model.sparse_encoder(sparse_frames)
            z_ode    = model.ode_encoder(kine)

            if zero_angle:  z_angle  = torch.zeros_like(z_angle)
            if zero_dense:  z_dense  = torch.zeros_like(z_dense)
            if zero_sparse: z_sparse = torch.zeros_like(z_sparse)
            if zero_ode:    z_ode    = torch.zeros_like(z_ode)

            z      = torch.cat([z_angle, z_dense, z_sparse, z_ode], dim=1)
            logits = model.classifier(z)

            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(y.tolist())

    return (f1_score(labels, preds, average='macro', zero_division=0),
            accuracy_score(labels, preds))


def main():
    splits       = get_kfold_splits()
    fold_results = {cond: {'f1': [], 'acc': []} for cond in CONDITIONS}

    for fold_idx, (train_subjects, val_subjects) in enumerate(splits):
        ckpt_path = os.path.join(CKPT_DIR, f'fold{fold_idx + 1}_best.pt')
        if not os.path.exists(ckpt_path):
            print(f"[Fold {fold_idx+1}] 체크포인트 없음 — 건너뜀")
            continue

        print(f"\n=== Fold {fold_idx + 1}/5 ===")

        model = MultiModalClassifier().to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

        train_ds = MultiModalDataset(train_subjects, augment=False)
        val_ds   = MultiModalDataset(
            val_subjects,
            angle_stats=(train_ds.angle_mean, train_ds.angle_std),
            kine_stats =(train_ds.kine_mean,  train_ds.kine_std),
            augment=False,
        )
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=2, pin_memory=True)

        for cond, (za, zd, zs, zo) in CONDITIONS.items():
            f1, acc = evaluate(model, val_loader,
                               zero_angle=za, zero_dense=zd,
                               zero_sparse=zs, zero_ode=zo)
            fold_results[cond]['f1'].append(f1)
            fold_results[cond]['acc'].append(acc)
            marker = '★' if cond == 'all' else ' '
            print(f"  {marker} {cond:12s} | F1: {f1:.4f}  Acc: {acc:.4f}")

    print("\n" + "=" * 58)
    print("[Ablation] 5-Fold 평균 결과")
    print("=" * 58)
    print(f"  {'조건':<14} {'F1 (mean±std)':<26} {'Acc (mean±std)'}")
    print("-" * 58)

    baseline_f1 = np.mean(fold_results['all']['f1'])
    for cond in CONDITIONS:
        f1s  = fold_results[cond]['f1']
        accs = fold_results[cond]['acc']
        delta = np.mean(f1s) - baseline_f1
        delta_str = f"  (Δ{delta:+.4f})" if cond != 'all' else ""
        print(f"  {cond:<14} {np.mean(f1s):.4f} ± {np.std(f1s):.4f}{delta_str:<18}  "
              f"{np.mean(accs):.4f} ± {np.std(accs):.4f}")

    print("\n※ Δ값이 음수일수록 해당 branch 제거 시 성능 하락 → 기여도 높음")


if __name__ == '__main__':
    main()

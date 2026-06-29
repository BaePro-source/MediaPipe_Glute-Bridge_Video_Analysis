"""
학습된 branch_weights 확인 스크립트
"""
import os
import torch
import numpy as np

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
BRANCHES = ['angle', 'dense', 'sparse', 'ode', 'graph']

all_weights = []

print(f"{'Fold':<6} " + "  ".join(f"{b:>8}" for b in BRANCHES))
print("-" * 58)

for fold in range(1, 11):
    ckpt_path = os.path.join(CKPT_DIR, f'fold{fold}_best.pt')
    if not os.path.exists(ckpt_path):
        print(f"Fold {fold:<2} — 체크포인트 없음")
        continue

    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'branch_weights' not in ckpt:
        print(f"Fold {fold:<2} — branch_weights 없음 (구버전 체크포인트)")
        continue

    w = torch.softmax(ckpt['branch_weights'], dim=0).numpy()
    all_weights.append(w)
    print(f"Fold {fold:<2}  " + "  ".join(f"{v:>8.4f}" for v in w))

if all_weights:
    mean_w = np.mean(all_weights, axis=0)
    std_w  = np.std(all_weights, axis=0)
    print("-" * 58)
    print(f"{'mean':<6} " + "  ".join(f"{v:>8.4f}" for v in mean_w))
    print(f"{'std':<6} " + "  ".join(f"{v:>8.4f}" for v in std_w))

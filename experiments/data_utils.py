"""
공통 데이터 유틸리티
- output/ 디렉토리에서 피험자 목록 자동 감지
- Train 70 / Test 30 분리 후 Train에 Stratified 10-Fold CV 적용
- angles.csv 및 graph 이미지 로딩
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')


def get_all_subjects():
    """
    output/best/ 와 output/worst/ 에서 피험자 디렉토리를 탐색.
    Returns: list of (subject_dir_path, label)  [best=1, worst=0]
    """
    subjects = []
    for label_name, label in [('best', 1), ('worst', 0)]:
        label_dir = os.path.join(OUTPUT_DIR, label_name)
        if not os.path.exists(label_dir):
            continue
        for subj in sorted(os.listdir(label_dir)):
            subj_path = os.path.join(label_dir, subj)
            if not os.path.isdir(subj_path):
                continue
            # angles.csv 있는지 확인 (처리 완료된 피험자만)
            csv_files = [f for f in os.listdir(subj_path) if f.endswith('_angles.csv')]
            if csv_files:
                subjects.append((subj_path, label))
    return subjects


def get_data_splits(n_folds=10, seed=42):
    """
    전체 데이터를 Train 70 / Test 30으로 분리 후
    Train 70개를 n_folds-Fold CV로 분할.

    제약: gb13_worst, gb27_worst는 반드시 Train set에 포함,
          같은 val fold에 들어가지 않음.

    Returns:
        kfold_splits : list of (train_subjects, val_subjects)  — n_folds개
        test_subjects: list of (path, label)                   — 30개
    """
    subjects = get_all_subjects()
    paths  = [s[0] for s in subjects]
    labels = [s[1] for s in subjects]

    # ── 1. gb13_worst / gb27_worst 인덱스 파악 ─────────────────────────────
    SPECIAL = ('gb13_worst', 'gb27_worst')
    special_idx = {}
    for i, path in enumerate(paths):
        for name in SPECIAL:
            if name in os.path.basename(path):
                special_idx[name] = i

    # ── 2. Stratified 70 / 30 split ────────────────────────────────────────
    indices = list(range(len(subjects)))
    train_idx, test_idx = train_test_split(
        indices, test_size=30, stratify=labels, random_state=seed
    )
    train_idx = list(train_idx)
    test_idx  = list(test_idx)

    # ── 3. special 피험자가 test에 있으면 train으로 swap ──────────────────
    for sidx in special_idx.values():
        if sidx in test_idx:
            for j, ti in enumerate(train_idx):   # train에서 교환 대상 탐색
                if labels[ti] == 0 and ti not in special_idx.values():
                    test_idx[test_idx.index(sidx)] = ti   # test: sidx → ti
                    train_idx[j]                   = sidx  # train: ti → sidx
                    print(f"[train 강제 배정] {os.path.basename(paths[sidx])} "
                          f"← {os.path.basename(paths[ti])}")
                    break

    train_subjects = [(paths[i], labels[i]) for i in train_idx]
    test_subjects  = [(paths[i], labels[i]) for i in test_idx]

    # ── 4. Train 70개에 n_folds-Fold CV ────────────────────────────────────
    tr_paths  = [s[0] for s in train_subjects]
    tr_labels = [s[1] for s in train_subjects]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    raw_splits = list(skf.split(tr_paths, tr_labels))

    val_sets   = [list(v) for _, v in raw_splits]
    train_sets = [list(t) for t, _ in raw_splits]

    # ── 5. gb13_worst / gb27_worst val fold 분리 제약 ─────────────────────
    special_tr_idx = {}
    for name in SPECIAL:
        for i, (path, _) in enumerate(train_subjects):
            if name in os.path.basename(path):
                special_tr_idx[name] = i

    if len(special_tr_idx) == 2:
        idx_a = special_tr_idx.get('gb13_worst')
        idx_b = special_tr_idx.get('gb27_worst')

        if idx_a is not None and idx_b is not None:
            fold_a = next(f for f, val in enumerate(val_sets) if idx_a in val)
            fold_b = next(f for f, val in enumerate(val_sets) if idx_b in val)

            if fold_a == fold_b:
                same_fold = fold_a
                swapped   = False
                for target_fold in range(n_folds):
                    if target_fold == same_fold or swapped:
                        continue
                    for swap_idx in val_sets[target_fold]:
                        if tr_labels[swap_idx] == 0 and swap_idx not in special_tr_idx.values():
                            val_sets[same_fold].remove(idx_b)
                            val_sets[same_fold].append(swap_idx)
                            val_sets[target_fold].remove(swap_idx)
                            val_sets[target_fold].append(idx_b)
                            train_sets[same_fold].remove(swap_idx)
                            train_sets[same_fold].append(idx_b)
                            train_sets[target_fold].remove(idx_b)
                            train_sets[target_fold].append(swap_idx)
                            print(f"[fold 분리 적용] gb27_worst: fold {same_fold+1} → "
                                  f"fold {target_fold+1} "
                                  f"(교환: {os.path.basename(tr_paths[swap_idx])})")
                            swapped = True
                            break

    kfold_splits = [
        ([(tr_paths[i], tr_labels[i]) for i in t],
         [(tr_paths[i], tr_labels[i]) for i in v])
        for t, v in zip(train_sets, val_sets)
    ]

    return kfold_splits, test_subjects


def print_split_info(seed=42, n_folds=10):
    """seed에 따른 데이터 분할 구성을 출력."""
    splits, test_subjects = get_data_splits(n_folds=n_folds, seed=seed)

    print(f"\n{'='*62}")
    print(f"  Seed: {seed}  |  {n_folds}-Fold CV  |  Train 70 / Test 30")
    print(f"{'='*62}")

    best_test  = sorted(os.path.basename(p) for p, l in test_subjects if l == 1)
    worst_test = sorted(os.path.basename(p) for p, l in test_subjects if l == 0)
    print(f"\n[Test Set]  {len(test_subjects)}개  "
          f"(best {len(best_test)}개 / worst {len(worst_test)}개)")
    print(f"  best : {best_test}")
    print(f"  worst: {worst_test}")

    print()
    for fold_idx, (tr, val) in enumerate(splits):
        best_val  = sorted(os.path.basename(p) for p, l in val if l == 1)
        worst_val = sorted(os.path.basename(p) for p, l in val if l == 0)
        print(f"  Fold {fold_idx+1:2d}  val({len(val):2d})  "
              f"best:{best_val}  worst:{worst_val}")


def load_angles_csv(subject_path):
    """
    angles.csv 로딩.
    Returns: numpy array (T, 3) — alpha, beta, gamma (float32)
    """
    csv_files = [f for f in os.listdir(subject_path) if f.endswith('_angles.csv')]
    if not csv_files:
        raise FileNotFoundError(f"angles.csv 없음: {subject_path}")
    df = pd.read_csv(os.path.join(subject_path, csv_files[0]))
    angles = df[['alpha', 'beta', 'gamma']].values.astype(np.float32)
    # NaN 처리: 앞뒤 보간 후 0으로 채움
    angles = pd.DataFrame(angles).interpolate().fillna(0).values.astype(np.float32)
    return angles


def get_graph_paths(subject_path):
    """
    graphs/ 폴더에서 alpha, beta, gamma 이미지 경로 반환.
    Returns: (alpha_path, beta_path, gamma_path)
    """
    graphs_dir = os.path.join(subject_path, 'graphs')
    files = os.listdir(graphs_dir)
    alpha = next(f for f in files if 'alpha' in f)
    beta  = next(f for f in files if 'beta' in f)
    gamma = next(f for f in files if 'gamma' in f)
    return (
        os.path.join(graphs_dir, alpha),
        os.path.join(graphs_dir, beta),
        os.path.join(graphs_dir, gamma),
    )


if __name__ == '__main__':
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42

    subjects = get_all_subjects()
    best_count  = sum(1 for _, l in subjects if l == 1)
    worst_count = sum(1 for _, l in subjects if l == 0)
    print(f"전체 피험자: {len(subjects)}명  (best={best_count}, worst={worst_count})")

    print_split_info(seed=seed)

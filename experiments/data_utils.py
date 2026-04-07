"""
공통 데이터 유틸리티
- output/ 디렉토리에서 피험자 목록 자동 감지
- Stratified 5-Fold 분할
- angles.csv 및 graph 이미지 로딩
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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


def get_kfold_splits(n_splits=5, random_state=42):
    """
    Stratified K-Fold 분할.
    제약 조건: gb13_worst 와 gb27_worst 는 절대 같은 fold(val)에 함께 들어가지 않음.
               같은 fold에 배정되면 다른 fold의 worst 피험자와 자동 swap.
    Returns: list of (train_subjects, val_subjects) — 각 fold마다
    """
    subjects = get_all_subjects()
    paths  = [s[0] for s in subjects]
    labels = [s[1] for s in subjects]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    raw_splits = list(skf.split(paths, labels))

    # 각 fold의 val 인덱스를 수정 가능한 리스트로 변환
    val_sets   = [list(val_idx)   for _, val_idx   in raw_splits]
    train_sets = [list(train_idx) for train_idx, _ in raw_splits]

    # gb13_worst, gb27_worst 의 전체 인덱스 탐색
    SPECIAL = ('gb13_worst', 'gb27_worst')
    special_idx = {}
    for i, path in enumerate(paths):
        for name in SPECIAL:
            if name in os.path.basename(path):
                special_idx[name] = i

    # 두 피험자 모두 존재할 때만 제약 적용
    if len(special_idx) == 2:
        idx_a = special_idx['gb13_worst']
        idx_b = special_idx['gb27_worst']

        fold_a = next(f for f, val in enumerate(val_sets) if idx_a in val)
        fold_b = next(f for f, val in enumerate(val_sets) if idx_b in val)

        if fold_a == fold_b:
            # idx_b 를 다른 fold로 이동 — 같은 클래스(worst=0)의 일반 피험자와 swap
            same_fold = fold_a
            swapped   = False
            for target_fold in range(n_splits):
                if target_fold == same_fold or swapped:
                    continue
                for swap_idx in val_sets[target_fold]:
                    # worst 클래스이고 special 이 아닌 피험자
                    if labels[swap_idx] == 0 and swap_idx not in special_idx.values():
                        # val 세트 swap
                        val_sets[same_fold].remove(idx_b)
                        val_sets[same_fold].append(swap_idx)
                        val_sets[target_fold].remove(swap_idx)
                        val_sets[target_fold].append(idx_b)
                        # train 세트도 반대로 swap
                        train_sets[same_fold].remove(swap_idx)
                        train_sets[same_fold].append(idx_b)
                        train_sets[target_fold].remove(idx_b)
                        train_sets[target_fold].append(swap_idx)
                        print(f"[fold 분리 적용] gb27_worst: fold {same_fold+1} → fold {target_fold+1} "
                              f"(교환 대상: {os.path.basename(paths[swap_idx])})")
                        swapped = True
                        break

    splits = []
    for train_idx, val_idx in zip(train_sets, val_sets):
        train = [(paths[i], labels[i]) for i in train_idx]
        val   = [(paths[i], labels[i]) for i in val_idx]
        splits.append((train, val))
    return splits


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
    subjects = get_all_subjects()
    best_count  = sum(1 for _, l in subjects if l == 1)
    worst_count = sum(1 for _, l in subjects if l == 0)
    print(f"전체 피험자: {len(subjects)}명  (best={best_count}, worst={worst_count})")

    splits = get_kfold_splits()
    for i, (train, val) in enumerate(splits):
        train_b = sum(1 for _, l in train if l == 1)
        train_w = sum(1 for _, l in train if l == 0)
        val_b   = sum(1 for _, l in val   if l == 1)
        val_w   = sum(1 for _, l in val   if l == 0)
        print(f"Fold {i+1}: train={len(train)} (best={train_b}, worst={train_w}) | "
              f"val={len(val)} (best={val_b}, worst={val_w})")

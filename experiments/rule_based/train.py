"""
Rule-based: Logistic Regression
입력: angles.csv에서 추출한 통계 피처 (15개)
      alpha, beta, gamma 각각 → mean, std, min, max, range(ROM)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report)
from data_utils import get_kfold_splits, load_angles_csv

ANGLE_NAMES = ['alpha', 'beta', 'gamma']
FEATURE_NAMES = [f"{a}_{s}" for a in ANGLE_NAMES for s in ['mean', 'std', 'min', 'max', 'range']]


def extract_features(subject_path):
    """angles.csv에서 통계 피처 15개 추출"""
    angles = load_angles_csv(subject_path)  # (T, 3)
    features = []
    for i in range(3):
        col = angles[:, i]
        features.extend([
            np.mean(col),
            np.std(col),
            np.min(col),
            np.max(col),
            np.max(col) - np.min(col),  # ROM (Range of Motion)
        ])
    return np.array(features, dtype=np.float32)


def run_fold(fold_idx, train_subjects, val_subjects):
    X_train = np.array([extract_features(p) for p, _ in train_subjects])
    y_train = np.array([l for _, l in train_subjects])
    X_val   = np.array([extract_features(p) for p, _ in val_subjects])
    y_val   = np.array([l for _, l in val_subjects])

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    model = LogisticRegression(max_iter=1000, random_state=42, C=1.0,
                               class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc    = accuracy_score(y_val, y_pred)
    f1     = f1_score(y_val, y_pred, average='macro', zero_division=0)
    prec   = precision_score(y_val, y_pred, average='macro', zero_division=0)
    rec    = recall_score(y_val, y_pred, average='macro', zero_division=0)

    print(f"\n=== Fold {fold_idx + 1}/5 ===")
    print(f"Val Acc: {acc:.4f}  F1(macro): {f1:.4f}  "
          f"Prec: {prec:.4f}  Rec: {rec:.4f}")
    print(classification_report(y_val, y_pred, target_names=['worst', 'best'],
                                zero_division=0))

    # 중요 피처 확인
    coef = model.coef_[0]
    top3 = np.argsort(np.abs(coef))[::-1][:3]
    print("Top 3 피처:", [(FEATURE_NAMES[i], round(coef[i], 3)) for i in top3])

    return {'acc': acc, 'f1': f1, 'precision': prec, 'recall': rec}


def main():
    splits  = get_kfold_splits()
    results = []

    for fold_idx, (train_subjects, val_subjects) in enumerate(splits):
        metrics = run_fold(fold_idx, train_subjects, val_subjects)
        results.append(metrics)

    print("\n" + "="*40)
    print("[Rule-based Logistic Regression] 5-Fold 결과")
    for key in ['acc', 'f1', 'precision', 'recall']:
        vals = [r[key] for r in results]
        print(f"  {key:10s}: {[round(v, 4) for v in vals]}  "
              f"평균={np.mean(vals):.4f}  std={np.std(vals):.4f}")


if __name__ == '__main__':
    main()

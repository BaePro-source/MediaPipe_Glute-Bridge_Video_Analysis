"""
Test Set Precision & Recall 계산 스크립트

기존 체크포인트를 로드해서 test set에 대한 precision/recall을 계산합니다.
모델 재학습 없음 — 파일 읽기만 수행합니다.

실행:
    cd experiments
    python eval_precision_recall.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data_utils import get_data_splits, load_angles_csv

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED       = 42
BATCH_SIZE = 8
N_FOLDS    = 10


# ─────────────────────────────────────────────────────────────────────────────
# 공통 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _print_results(model_name, fold_prec, fold_rec):
    print(f"\n{'='*55}")
    print(f"[{model_name}] Test Set Precision & Recall")
    print(f"{'='*55}")
    print(f"  {'Fold':<8} {'Precision':>10}  {'Recall':>10}")
    print(f"  {'-'*34}")
    for i, (p, r) in enumerate(zip(fold_prec, fold_rec), 1):
        print(f"  Fold {i:<4}  {p:>10.4f}  {r:>10.4f}")
    print(f"  {'-'*34}")
    print(f"  {'mean':<8} {np.mean(fold_prec):>10.4f}  {np.mean(fold_rec):>10.4f}")
    print(f"  {'std':<8} {np.std(fold_prec):>10.4f}  {np.std(fold_rec):>10.4f}")


def _torch_inference(model, loader):
    """모델 eval 모드로 test loader 전체 예측 반환."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            *inputs, y = batch
            inputs = [x.to(DEVICE) for x in inputs]
            logits = model(*inputs)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(y.tolist())
    return labels, preds


# ─────────────────────────────────────────────────────────────────────────────
# 1. Rule-based (Logistic Regression)
# ─────────────────────────────────────────────────────────────────────────────

def eval_rule_based(splits, test_subjects):
    def extract_features(subject_path):
        angles = load_angles_csv(subject_path)
        feats  = []
        for i in range(3):
            col = angles[:, i]
            feats.extend([col.mean(), col.std(), col.min(), col.max(),
                          col.max() - col.min()])
        return np.array(feats, dtype=np.float32)

    X_test  = np.array([extract_features(p) for p, _ in test_subjects])
    y_test  = np.array([l for _, l in test_subjects])

    fold_prec, fold_rec = [], []
    for fold_idx, (train_subjects, _) in enumerate(splits):
        X_train = np.array([extract_features(p) for p, _ in train_subjects])
        y_train = np.array([l for _, l in train_subjects])

        scaler  = StandardScaler()
        X_tr    = scaler.fit_transform(X_train)
        X_te    = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=42,
                                 C=1.0, class_weight='balanced')
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)

        fold_prec.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        fold_rec.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        print(f"  Rule-based Fold {fold_idx+1} done")

    _print_results('Rule-based', fold_prec, fold_rec)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LSTM
# ─────────────────────────────────────────────────────────────────────────────

def eval_lstm(splits, test_subjects):
    from lstm.dataset import AngleSequenceDataset
    from lstm.model   import LSTMClassifier

    ckpt_dir   = os.path.join(os.path.dirname(__file__), 'lstm', 'checkpoints')
    fold_prec, fold_rec = [], []

    for fold_idx, (train_subjects, _) in enumerate(splits):
        ckpt_path = os.path.join(ckpt_dir, f'fold{fold_idx+1}_best.pt')
        if not os.path.exists(ckpt_path):
            print(f"  LSTM Fold {fold_idx+1} 체크포인트 없음 — 건너뜀")
            continue

        train_ds = AngleSequenceDataset(train_subjects, augment=False)
        test_ds  = AngleSequenceDataset(test_subjects,
                                        mean=train_ds.mean, std=train_ds.std,
                                        augment=False)
        loader   = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = LSTMClassifier().to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

        labels, preds = _torch_inference(model, loader)
        fold_prec.append(precision_score(labels, preds, average='macro', zero_division=0))
        fold_rec.append(recall_score(labels, preds, average='macro', zero_division=0))
        print(f"  LSTM Fold {fold_idx+1} done")

    _print_results('LSTM', fold_prec, fold_rec)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Transformer
# ─────────────────────────────────────────────────────────────────────────────

def eval_transformer(splits, test_subjects):
    from lstm.dataset        import AngleSequenceDataset
    from transformer.model   import TransformerClassifier

    ckpt_dir   = os.path.join(os.path.dirname(__file__), 'transformer', 'checkpoints')
    fold_prec, fold_rec = [], []

    for fold_idx, (train_subjects, _) in enumerate(splits):
        ckpt_path = os.path.join(ckpt_dir, f'fold{fold_idx+1}_best.pt')
        if not os.path.exists(ckpt_path):
            print(f"  Transformer Fold {fold_idx+1} 체크포인트 없음 — 건너뜀")
            continue

        train_ds = AngleSequenceDataset(train_subjects, augment=False)
        test_ds  = AngleSequenceDataset(test_subjects,
                                        mean=train_ds.mean, std=train_ds.std,
                                        augment=False)
        loader   = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = TransformerClassifier().to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

        labels, preds = _torch_inference(model, loader)
        fold_prec.append(precision_score(labels, preds, average='macro', zero_division=0))
        fold_rec.append(recall_score(labels, preds, average='macro', zero_division=0))
        print(f"  Transformer Fold {fold_idx+1} done")

    _print_results('Transformer', fold_prec, fold_rec)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PatchTST
# ─────────────────────────────────────────────────────────────────────────────

def eval_patchtst(splits, test_subjects):
    from lstm.dataset      import AngleSequenceDataset
    from patchtst.model    import PatchTSTClassifier

    ckpt_dir   = os.path.join(os.path.dirname(__file__), 'patchtst', 'checkpoints')
    fold_prec, fold_rec = [], []

    for fold_idx, (train_subjects, _) in enumerate(splits):
        ckpt_path = os.path.join(ckpt_dir, f'fold{fold_idx+1}_best.pt')
        if not os.path.exists(ckpt_path):
            print(f"  PatchTST Fold {fold_idx+1} 체크포인트 없음 — 건너뜀")
            continue

        train_ds = AngleSequenceDataset(train_subjects, augment=False)
        test_ds  = AngleSequenceDataset(test_subjects,
                                        mean=train_ds.mean, std=train_ds.std,
                                        augment=False)
        loader   = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = PatchTSTClassifier().to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

        labels, preds = _torch_inference(model, loader)
        fold_prec.append(precision_score(labels, preds, average='macro', zero_division=0))
        fold_rec.append(recall_score(labels, preds, average='macro', zero_division=0))
        print(f"  PatchTST Fold {fold_idx+1} done")

    _print_results('PatchTST', fold_prec, fold_rec)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ResNet152
# ─────────────────────────────────────────────────────────────────────────────

def eval_resnet(splits, test_subjects):
    from resnet.dataset import GraphImageDataset
    from resnet.model   import build_resnet152

    ckpt_dir   = os.path.join(os.path.dirname(__file__), 'resnet', 'checkpoints')
    fold_prec, fold_rec = [], []

    test_ds = GraphImageDataset(test_subjects, is_train=False)
    loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    for fold_idx in range(N_FOLDS):
        ckpt_path = os.path.join(ckpt_dir, f'fold{fold_idx+1}_best.pt')
        if not os.path.exists(ckpt_path):
            print(f"  ResNet Fold {fold_idx+1} 체크포인트 없음 — 건너뜀")
            continue

        model = build_resnet152(pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

        labels, preds = _torch_inference(model, loader)
        fold_prec.append(precision_score(labels, preds, average='macro', zero_division=0))
        fold_rec.append(recall_score(labels, preds, average='macro', zero_division=0))
        print(f"  ResNet Fold {fold_idx+1} done")

    _print_results('ResNet152', fold_prec, fold_rec)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Multimodal 공통 로직
# ─────────────────────────────────────────────────────────────────────────────

def _eval_multimodal_from_dir(ckpt_dir, splits, test_subjects, label):
    from multimodal.dataset import MultiModalDataset
    from multimodal.model   import MultiModalClassifier

    fold_prec, fold_rec = [], []

    for fold_idx, (train_subjects, _) in enumerate(splits):
        ckpt_path = os.path.join(ckpt_dir, f'fold{fold_idx+1}_best.pt')
        if not os.path.exists(ckpt_path):
            print(f"  {label} Fold {fold_idx+1} 체크포인트 없음 — 건너뜀")
            continue

        train_ds = MultiModalDataset(train_subjects, augment=False)
        test_ds  = MultiModalDataset(
            test_subjects,
            angle_stats=(train_ds.angle_mean, train_ds.angle_std),
            augment=False,
        )
        loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                            num_workers=2, pin_memory=True)

        state_dict  = torch.load(ckpt_path, map_location=DEVICE)
        has_weights = 'branch_weights' in state_dict
        model       = MultiModalClassifier().to(DEVICE)
        model.load_state_dict(state_dict, strict=has_weights)

        labels, preds = _torch_inference(model, loader)
        fold_prec.append(precision_score(labels, preds, average='macro', zero_division=0))
        fold_rec.append(recall_score(labels, preds, average='macro', zero_division=0))
        print(f"  {label} Fold {fold_idx+1} done")

    _print_results(label, fold_prec, fold_rec)


def eval_multimodal(splits, test_subjects):
    """branch_weights 없는 버전 (백업 폴더 사용)"""
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'multimodal',
                            'checkpoints_no_branch_weights')
    _eval_multimodal_from_dir(ckpt_dir, splits, test_subjects, 'Multimodal')


def eval_multimodal_with_weights(splits, test_subjects):
    """branch_weights 있는 버전 (재학습 후 checkpoints 사용)"""
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'multimodal', 'checkpoints')
    _eval_multimodal_from_dir(ckpt_dir, splits, test_subjects,
                              'Multimodal with Training Parameter')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    splits, test_subjects = get_data_splits(seed=SEED)

    eval_rule_based(splits, test_subjects)
    eval_lstm(splits, test_subjects)
    eval_transformer(splits, test_subjects)
    eval_patchtst(splits, test_subjects)
    eval_resnet(splits, test_subjects)
    eval_multimodal(splits, test_subjects)
    eval_multimodal_with_weights(splits, test_subjects)


if __name__ == '__main__':
    main()

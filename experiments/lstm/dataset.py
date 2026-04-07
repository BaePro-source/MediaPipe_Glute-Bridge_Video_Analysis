"""
LSTM용 데이터셋
입력: angles.csv 시계열 (T, 3) → MAX_LEN으로 패딩/잘라내기
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import Dataset
from data_utils import load_angles_csv

MAX_LEN = 400  # 실제 데이터 최대 372프레임 기준으로 축소 (512 → 400)


class AngleSequenceDataset(Dataset):
    def __init__(self, subjects, mean=None, std=None, augment=False):
        """
        subjects: list of (path, label)
        mean, std: 정규화 파라미터 (None이면 이 데이터셋 기준으로 계산 — train용)
        augment:   True이면 학습 시 시계열 augmentation 적용
        """
        self.subjects = subjects
        self.augment  = augment

        # 정규화 파라미터 계산 (train에서만)
        if mean is None:
            all_angles = np.concatenate([load_angles_csv(p) for p, _ in subjects], axis=0)
            self.mean = all_angles.mean(axis=0)  # (3,)
            self.std  = all_angles.std(axis=0) + 1e-8
        else:
            self.mean = mean
            self.std  = std

    def __len__(self):
        return len(self.subjects)

    def _augment(self, angles: np.ndarray) -> np.ndarray:
        """
        시계열 augmentation (학습 전용):
          - Gaussian noise: 각도 값에 미세한 노이즈 추가
          - Random scale:   전체 시퀀스를 [0.9, 1.1] 범위로 스케일 조정
        """
        angles = angles + np.random.normal(0, 0.05, angles.shape).astype(np.float32)
        angles = angles * np.random.uniform(0.9, 1.1)
        return angles

    def __getitem__(self, idx):
        path, label = self.subjects[idx]
        angles = load_angles_csv(path)  # (T, 3)

        # 정규화
        angles = (angles - self.mean) / self.std

        # augmentation (train only)
        if self.augment:
            angles = self._augment(angles)

        # 패딩 or 잘라내기
        T = angles.shape[0]
        if T >= MAX_LEN:
            angles = angles[:MAX_LEN]
        else:
            pad    = np.zeros((MAX_LEN - T, 3), dtype=np.float32)
            angles = np.concatenate([angles, pad], axis=0)

        x = torch.tensor(angles, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

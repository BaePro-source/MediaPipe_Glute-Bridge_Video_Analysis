"""
MultiModal Dataset — 4-Branch

Branch 1 — Angle Transformer 입력:
    angles.csv → (T, 3) 정규화 후 MAX_LEN 패딩/절사

Branch 2 — Dense Flow ViT 입력:
    _optflow_dense.mp4 에서 K_FRAMES 프레임 균등 샘플링
    → (K, 3, IMG_SIZE, IMG_SIZE) ImageNet 정규화

Branch 3 — Sparse Flow ViT 입력:
    _optflow_sparse.mp4 에서 K_FRAMES 프레임 균등 샘플링
    → (K, 3, IMG_SIZE, IMG_SIZE) ImageNet 정규화

Branch 4 — Neural ODE 입력:
    [위치=angles, 속도=Δangles, 가속도=ΔΔangles]
    → (T, 9) 정규화 후 MAX_LEN 패딩/절사
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from data_utils import load_angles_csv

MAX_LEN  = 400
K_FRAMES = 3
IMG_SIZE = 224

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _compute_kinematics(angles: np.ndarray) -> np.ndarray:
    vel = np.diff(angles, axis=0, prepend=angles[:1])
    acc = np.diff(vel,    axis=0, prepend=vel[:1])
    return np.concatenate([angles, vel, acc], axis=1).astype(np.float32)


def _pad_or_crop(arr: np.ndarray, max_len: int) -> np.ndarray:
    T, D = arr.shape
    if T >= max_len:
        return arr[:max_len]
    return np.concatenate([arr, np.zeros((max_len - T, D), dtype=np.float32)], axis=0)


class MultiModalDataset(Dataset):
    def __init__(self, subjects, angle_stats=None, kine_stats=None, augment=False):
        """
        subjects    : list of (subject_dir_path, label)
        angle_stats : (mean(3,), std(3,)) — None 이면 train 기준 계산
        kine_stats  : (mean(9,), std(9,)) — None 이면 train 기준 계산
        augment     : True 이면 시계열 augmentation 적용 (train only)
        """
        self.subjects = subjects
        self.augment  = augment

        self.img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

        if angle_stats is None:
            all_angles = np.concatenate(
                [load_angles_csv(p) for p, _ in subjects], axis=0)
            self.angle_mean = all_angles.mean(axis=0)
            self.angle_std  = all_angles.std(axis=0) + 1e-8
        else:
            self.angle_mean, self.angle_std = angle_stats

        if kine_stats is None:
            all_kine = np.concatenate(
                [_compute_kinematics(load_angles_csv(p)) for p, _ in subjects], axis=0)
            self.kine_mean = all_kine.mean(axis=0)
            self.kine_std  = all_kine.std(axis=0) + 1e-8
        else:
            self.kine_mean, self.kine_std = kine_stats

    # ── 내부 로더 ──────────────────────────────────────────────────────────────

    def _load_angle_seq(self, subject_path: str) -> np.ndarray:
        angles = load_angles_csv(subject_path)
        angles = (angles - self.angle_mean) / self.angle_std
        if self.augment:
            angles += np.random.normal(0, 0.05, angles.shape).astype(np.float32)
            angles *= np.random.uniform(0.9, 1.1)
        return _pad_or_crop(angles.astype(np.float32), MAX_LEN)

    def _load_kine_seq(self, subject_path: str) -> np.ndarray:
        angles = load_angles_csv(subject_path)
        kine   = _compute_kinematics(angles)
        kine   = (kine - self.kine_mean) / self.kine_std
        if self.augment:
            kine += np.random.normal(0, 0.05, kine.shape).astype(np.float32)
            kine *= np.random.uniform(0.9, 1.1)
        return _pad_or_crop(kine.astype(np.float32), MAX_LEN)

    def _load_video_frames(self, subject_path: str, suffix: str) -> torch.Tensor:
        files = [f for f in os.listdir(subject_path) if f.endswith(suffix)]
        if not files:
            raise FileNotFoundError(f"{suffix} 영상 없음: {subject_path}")

        cap   = cv2.VideoCapture(os.path.join(subject_path, files[0]))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(total - 1, 0), K_FRAMES, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.img_transform(frame))
        cap.release()

        return torch.stack(frames)   # (K, 3, H, W)

    # ── __getitem__ ────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        path, label = self.subjects[idx]

        angle_seq    = torch.from_numpy(self._load_angle_seq(path))          # (MAX_LEN, 3)
        dense_frames = self._load_video_frames(path, '_optflow_dense.mp4')   # (K, 3, H, W)
        sparse_frames= self._load_video_frames(path, '_optflow_sparse.mp4')  # (K, 3, H, W)
        kine_seq     = torch.from_numpy(self._load_kine_seq(path))           # (MAX_LEN, 9)
        y            = torch.tensor(label, dtype=torch.long)

        return angle_seq, dense_frames, sparse_frames, kine_seq, y

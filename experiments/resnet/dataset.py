"""
ResNet용 데이터셋
입력: alpha, beta, gamma 그래프 이미지 각각을 grayscale로 변환 후
      3채널(R=alpha, G=beta, B=gamma)로 합쳐서 224x224 이미지 생성
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from data_utils import get_graph_paths

IMG_SIZE = 224

# ImageNet 정규화 (pretrained ResNet 사용)
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    NORMALIZE,
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    NORMALIZE,
])


class GraphImageDataset(Dataset):
    def __init__(self, subjects, is_train=True):
        """
        subjects: list of (path, label)
        각 피험자의 alpha/beta/gamma 그래프를 grayscale로 읽어
        RGB 3채널 이미지로 합성.
        """
        self.subjects  = subjects
        self.transform = TRAIN_TRANSFORM if is_train else VAL_TRANSFORM

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        path, label = self.subjects[idx]
        alpha_p, beta_p, gamma_p = get_graph_paths(path)

        # 각 그래프를 grayscale로 로딩 후 RGB 합성
        alpha_img = Image.open(alpha_p).convert('L')  # (W, H)
        beta_img  = Image.open(beta_p).convert('L')
        gamma_img = Image.open(gamma_p).convert('L')

        # 3채널 RGB 이미지 합성
        img = Image.merge('RGB', [alpha_img, beta_img, gamma_img])

        x = self.transform(img)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

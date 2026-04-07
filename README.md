# Glute Bridge Video Analysis

MediaPipe Pose를 활용해 글루트 브릿지 동작 영상에서 관절 각도를 자동으로 추출하고 시각화하는 분석 파이프라인입니다.
추출된 각도 데이터를 바탕으로 동작 품질(best/worst)을 분류하는 머신러닝 실험도 포함합니다.

---

## 측정 각도

| 각도 | 관절 | 설명 |
|------|------|------|
| **alpha** | 발목 | KNEE→ANKLE 벡터와 FOOT_INDEX→HEEL 벡터 사이 각도 (발목·발 정렬) |
| **beta**  | 고관절 | KNEE→HIP→SHOULDER (꼭짓점: HIP) |
| **gamma** | 무릎 | ANKLE→KNEE→HIP (꼭짓점: KNEE) |

> 매 프레임마다 MediaPipe의 `visibility` 점수를 비교해 카메라에 더 잘 보이는 쪽(좌/우)을 자동으로 선택합니다.

---

## 실행 환경

- Python 3.10
- conda 환경 사용 권장

---

## 필수 패키지

| 패키지 | 버전 | 용도 |
|--------|------|------|
| mediapipe | 0.10.14 | 포즈 추정 |
| opencv-python | 4.10.0.84 | 영상 처리 |
| numpy | 1.26.4 | 수치 연산 |
| pandas | 2.2.3 | 각도 데이터 처리 |
| matplotlib | 3.9.2 | 그래프 시각화 |
| tqdm | 4.67.1 | 진행률 표시 |
| torch | 2.11.0 | LSTM / ResNet / Transformer 학습 |
| torchvision | 0.26.0 | ResNet18 pretrained 모델 |
| scikit-learn | 1.7.2 | Logistic Regression, K-Fold, class weight |
| Pillow | 12.1.1 | 그래프 이미지 로딩 (ResNet 입력) |

---

## 설치

```bash
conda env create -f environment.yml
conda activate mediapipe_video
```

> GPU 사용 시 torch/torchvision은 [https://pytorch.org](https://pytorch.org) 에서 CUDA 버전에 맞는 명령으로 별도 설치하세요.

---

## 프로젝트 구조

```
mediapipe_glute bridge_video_analysis/
├── input/                        # 분석할 영상 파일 위치
│   ├── best/                     # 우수 동작 영상
│   └── worst/                    # 불량 동작 영상
├── output/                       # 분석 결과 저장 (자동 생성)
│   ├── best/
│   │   └── gb01_best/
│   │       ├── gb01_best_angles.csv
│   │       ├── gb01_best_skeleton.mp4
│   │       ├── gb01_best_optflow.mp4
│   │       └── graphs/
│   │           ├── gb01_best_alpha.png
│   │           ├── gb01_best_beta.png
│   │           └── gb01_best_gamma.png
│   └── worst/
│       └── (동일 구조)
├── config/
│   └── angles.json               # 각도 정의 (랜드마크 인덱스)
├── src/
│   ├── angle_calculator.py       # 관절 각도 계산
│   ├── video_analyzer.py         # 영상 프레임 분석
│   ├── graph_plotter.py          # 그래프 시각화
│   ├── skeleton_renderer.py      # 스켈레톤 오버레이 영상 생성
│   └── optical_flow.py           # Optical Flow 영상 생성
├── scripts/
│   └── run_pipeline.py           # 분석 파이프라인 실행 진입점
├── experiments/
│   ├── data_utils.py             # 데이터 로딩, Stratified 5-Fold 분할
│   ├── rule_based/train.py       # Logistic Regression (통계 피처 15개)
│   ├── lstm/                     # Bidirectional LSTM 분류기
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── train.py
│   ├── resnet/                   # ResNet18 기반 분류기
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── train.py
│   └── transformer/              # Transformer 기반 분류기
│       ├── dataset.py
│       ├── model.py
│       └── train.py
└── environment.yml
```

---

## 사용 방법

### 1. 영상 파일 준비

분석할 영상을 `input/best/`, `input/worst/` 폴더에 넣습니다.

```
input/
├── best/
│   ├── gb01_best.mp4
│   └── gb02_best.mp4
└── worst/
    ├── gb01_worst.mp4
    └── gb02_worst.mp4
```

### 2. 파이프라인 실행

```bash
python scripts/run_pipeline.py
```

옵션:

```bash
# 모델 복잡도 지정 (0=빠름, 1=기본(default), 2=정확)
python scripts/run_pipeline.py --complexity 2

# CSV + 그래프만 생성 (skeleton/optflow 영상 생략 — 속도 빠름)
python scripts/run_pipeline.py --no-video

# 이미 처리된 영상 건너뜀
python scripts/run_pipeline.py --only-new

# 입출력 경로 직접 지정
python scripts/run_pipeline.py --input input --output output
```

### 3. 결과 확인

```
output/
└── best/
    └── gb01_best/
        ├── gb01_best_angles.csv     # 프레임별 side, alpha, beta, gamma 각도
        ├── gb01_best_skeleton.mp4   # 스켈레톤 오버레이 영상
        ├── gb01_best_optflow.mp4    # Optical Flow 시각화 영상
        └── graphs/
            ├── gb01_best_alpha.png
            ├── gb01_best_beta.png
            └── gb01_best_gamma.png
```

---

## 출력 예시

- **CSV**: 프레임 번호, 시간(초), 좌우 선택(side), alpha/beta/gamma 각도 (랜드마크 미검출 프레임은 NaN)
- **그래프**: X축 시간(초), Y축 각도(°)
- **skeleton 영상**: 원본 영상에 MediaPipe 랜드마크와 연결선 오버레이
- **optflow 영상**: 키포인트 기반 Optical Flow 벡터 시각화

---

## 머신러닝 실험 (experiments/)

`output/` 의 분석 결과를 입력으로 사용해 동작 품질(best=1 / worst=0)을 분류합니다.
모든 실험은 **Stratified 5-Fold Cross Validation**으로 평가합니다.

> `gb13_worst`와 `gb27_worst`는 동일한 validation fold에 배정되지 않도록 제약이 적용됩니다.

### 공통 실행 방법

```bash
cd experiments

# Rule-based: Logistic Regression (통계 피처 15개)
python rule_based/train.py

# LSTM: Bidirectional LSTM (각도 시계열 입력)
python lstm/train.py

# ResNet18: 그래프 이미지 3채널 합성 입력
python resnet/train.py

# Transformer
python transformer/train.py
```

### 모델별 입력 방식

| 모델 | 입력 | 특징 |
|------|------|------|
| Rule-based | alpha/beta/gamma 각각 mean, std, min, max, range → 15개 피처 | 해석 용이 |
| LSTM | 각도 시계열 (T, 3) → MAX_LEN=400 패딩 | 시간 흐름 반영 |
| ResNet18 | alpha/beta/gamma 그래프 이미지를 RGB 3채널로 합성 (224×224) | pretrained 활용 |
| Transformer | 각도 시계열 (T, 3) | 장거리 의존성 모델링 |

### LSTM 학습 주요 설정

| 항목 | 값 |
|------|-----|
| Seed | 42 (재현성 보장) |
| Epochs | 100 (Early Stopping 적용) |
| Early Stopping | val macro-F1 기준, patience=15 |
| Batch size | 8 |
| Learning rate | 1e-3 (Adam, weight_decay=1e-4) |
| Class weight | balanced (sklearn compute_class_weight) |
| Augmentation | Gaussian noise + random scale (train only) |
| Best model 기준 | val macro-F1 |
| 체크포인트 저장 | `experiments/lstm/checkpoints/fold{N}_best.pt` |

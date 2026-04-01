# Glute Bridge Video Analysis

MediaPipe Pose를 활용해 글루트 브릿지 동작 영상에서 관절 각도를 자동으로 추출하고 시각화하는 분석 파이프라인입니다.

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

| 패키지 | 버전 |
|--------|------|
| mediapipe | 0.10.14 |
| opencv-python | 4.10.0.84 |
| numpy | 1.26.4 |
| pandas | 2.2.3 |
| matplotlib | 3.9.2 |
| tqdm | 4.67.1 |

---

## 설치

```bash
conda env create -f environment.yml
conda activate mediapipe_video
```

---

## 프로젝트 구조

```
mediapipe_glute bridge_video_analysis/
├── input/                  # 분석할 영상 파일 위치
├── output/                 # 분석 결과 저장 (자동 생성)
├── config/
│   └── angles.json         # 각도 정의
├── src/
│   ├── angle_calculator.py # 관절 각도 계산
│   ├── video_analyzer.py   # 영상 프레임 분석
│   └── graph_plotter.py    # 그래프 시각화
├── scripts/
│   └── run_pipeline.py     # 실행 진입점
└── environment.yml
```

---

## 사용 방법

### 1. 영상 파일 준비

분석할 영상을 `input/` 폴더에 넣습니다. 파일 이름이 결과 폴더명(sample ID)이 됩니다.

```
input/
    gb01.mp4
    gb02.mp4
```

### 2. 파이프라인 실행

```bash
python scripts/run_pipeline.py
```

옵션:

```bash
# 모델 복잡도 지정 (0=빠름, 1=기본(default), 2=정확)
python scripts/run_pipeline.py --complexity 2

# 입출력 경로 직접 지정
python scripts/run_pipeline.py --input input --output output
```

### 3. 결과 확인

```
output/
    gb01/
        gb01_angles.csv         # 프레임별 alpha, beta, gamma 각도
        graphs/
            gb01_alpha.png
            gb01_beta.png
            gb01_gamma.png
```

---

## 출력 예시

- **CSV**: 프레임 번호, 시간(초), alpha/beta/gamma 각도 (랜드마크 미검출 프레임은 NaN)
- **그래프**: X축 시간(초), Y축 각도(°), 최댓값·최솟값 자동 표시

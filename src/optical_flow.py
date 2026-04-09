"""
Optical Flow 영상 생성 — 두 가지 방식

render_optflow_dense:
    Farneback Dense Optical Flow + HSV 인코딩
    - Hue: 이동 방향, Value: 이동 크기
    - 전체 픽셀에 flow 정보 → ViT 입력에 적합

render_optflow_sparse:
    Lucas-Kanade Sparse Optical Flow
    - 관절 6개 위치에서만 flow 계산 → 화살표로 시각화
    - 어두운 배경 + skeleton 오버레이
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ── Dense (Farneback) 파라미터 ─────────────────────────────────────────────────
_FB_PARAMS = dict(
    pyr_scale  = 0.5,
    levels     = 3,
    winsize    = 15,
    iterations = 3,
    poly_n     = 5,
    poly_sigma = 1.2,
    flags      = 0,
)

# ── Sparse (Lucas-Kanade) 파라미터 ────────────────────────────────────────────
_LK_PARAMS = dict(
    winSize  = (21, 21),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# 관절별 BGR 색상
KEYPOINT_COLORS = {
    "SHOULDER":   (100, 100, 255),
    "HIP":        (100, 255, 100),
    "KNEE":       (255, 100, 100),
    "ANKLE":      (100, 255, 255),
    "HEEL":       (255, 100, 255),
    "FOOT_INDEX": (255, 255, 100),
}

CONNECTIONS = [
    ("SHOULDER", "HIP"),
    ("HIP",      "KNEE"),
    ("KNEE",     "ANKLE"),
    ("ANKLE",    "HEEL"),
    ("ANKLE",    "FOOT_INDEX"),
]

_MIN_FLOW_MAG  = 0.3
_ARROW_SCALE   = 4.0   # 화살표를 실제 displacement보다 크게 그려 미세 이동도 가시화


def _draw_skeleton(frame: np.ndarray, kps: dict) -> None:
    for a, b in CONNECTIONS:
        if a in kps and b in kps:
            cv2.line(frame, kps[a], kps[b], (220, 220, 220), 2, cv2.LINE_AA)
    for name, pt in kps.items():
        color = KEYPOINT_COLORS.get(name, (255, 255, 255))
        cv2.circle(frame, pt, 6, color,          -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 6, (255, 255, 255),  1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# Dense Optical Flow (Farneback + HSV)
# ══════════════════════════════════════════════════════════════════════════════

def render_optflow_dense(
    video_path: Path,
    output_path: Path,
    frame_keypoints: list | None = None,
) -> None:
    """
    Dense Farneback Optical Flow + HSV 인코딩 영상 저장.
    frame_keypoints 가 주어지면 skeleton 오버레이 추가.
    """
    video_path  = Path(video_path)
    output_path = Path(output_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    prev_gray = None

    for i in tqdm(range(total), desc=f"dense flow: {video_path.name}", leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow       = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **_FB_PARAMS)
            mag, ang   = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv        = np.zeros((height, width, 3), dtype=np.uint8)
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            annotated  = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            annotated = np.zeros((height, width, 3), dtype=np.uint8)

        if frame_keypoints is not None:
            curr_kps = frame_keypoints[i] if i < len(frame_keypoints) else None
            if curr_kps is not None:
                _draw_skeleton(annotated, curr_kps)

        prev_gray = gray
        writer.write(annotated)

    cap.release()
    writer.release()


# ══════════════════════════════════════════════════════════════════════════════
# Sparse Optical Flow (Lucas-Kanade 화살표)
# ══════════════════════════════════════════════════════════════════════════════

def render_optflow_sparse(
    video_path: Path,
    output_path: Path,
    frame_keypoints: list,
) -> None:
    """
    Lucas-Kanade Sparse Optical Flow 영상 저장.
    관절 6개 위치에서 flow 계산 → 컬러 화살표 + skeleton 오버레이.
    """
    video_path  = Path(video_path)
    output_path = Path(output_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    prev_gray = None
    prev_kps  = None

    for i in tqdm(range(total), desc=f"sparse flow: {video_path.name}", leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        annotated = (frame * 0.6).astype(np.uint8)   # 배경 어둡게

        curr_kps = frame_keypoints[i] if i < len(frame_keypoints) else None

        if curr_kps is not None:
            # ── LK 화살표 ──────────────────────────────────────────────────
            if prev_gray is not None and prev_kps is not None:
                for name in KEYPOINT_COLORS:
                    if name not in prev_kps or name not in curr_kps:
                        continue

                    prev_pt              = np.array([[prev_kps[name]]], dtype=np.float32)
                    next_pts, status, _  = cv2.calcOpticalFlowPyrLK(
                                               prev_gray, gray, prev_pt, None, **_LK_PARAMS)

                    px, py = prev_kps[name]

                    # LK 추적 성공 시 LK 결과, 실패 시 MediaPipe 좌표 차이 사용
                    if status is not None and status[0][0] == 1:
                        lx, ly    = next_pts[0][0]
                        dx, dy    = lx - px, ly - py
                        lk_ok     = True
                    else:
                        cx, cy    = curr_kps[name]
                        dx, dy    = cx - px, cy - py
                        lk_ok     = False

                    mag = np.hypot(dx, dy)
                    if mag < _MIN_FLOW_MAG:
                        continue

                    # 화살표 끝점: displacement를 스케일링해서 시각적으로 강조
                    ex = int(px + dx * _ARROW_SCALE)
                    ey = int(py + dy * _ARROW_SCALE)

                    color     = KEYPOINT_COLORS[name]
                    thickness = 2 if lk_ok else 1
                    tip_len   = min(0.4, 10.0 / max(mag * _ARROW_SCALE, 1.0))

                    cv2.arrowedLine(annotated, (px, py), (ex, ey),
                                    (0, 0, 0), thickness + 2, cv2.LINE_AA, tipLength=tip_len)
                    cv2.arrowedLine(annotated, (px, py), (ex, ey),
                                    color, thickness, cv2.LINE_AA, tipLength=tip_len)

            # ── skeleton ───────────────────────────────────────────────────
            _draw_skeleton(annotated, curr_kps)

            # ── 범례 ───────────────────────────────────────────────────────
            for j, (name, color) in enumerate(KEYPOINT_COLORS.items()):
                pos = (10, 30 + j * 22)
                cv2.putText(annotated, name, pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),   3, cv2.LINE_AA)
                cv2.putText(annotated, name, pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color,       1, cv2.LINE_AA)

        prev_gray = gray
        prev_kps  = curr_kps
        writer.write(annotated)

    cap.release()
    writer.release()

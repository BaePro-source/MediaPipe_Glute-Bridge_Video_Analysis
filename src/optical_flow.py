import cv2
import numpy as np
from collections import deque
from pathlib import Path
from tqdm import tqdm


# Lucas-Kanade 파라미터
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# 관절별 BGR 색상
KEYPOINT_COLORS = {
    "SHOULDER":   (100, 100, 255),  # red
    "HIP":        (100, 255, 100),  # green
    "KNEE":       (255, 100, 100),  # blue
    "ANKLE":      (100, 255, 255),  # yellow
    "HEEL":       (255, 100, 255),  # magenta
    "FOOT_INDEX": (255, 255, 100),  # cyan
}

# 관심 관절 연결 (skeleton_renderer와 동일 6개 관절)
CONNECTIONS = [
    ("SHOULDER", "HIP"),
    ("HIP", "KNEE"),
    ("KNEE", "ANKLE"),
    ("ANKLE", "HEEL"),
    ("ANKLE", "FOOT_INDEX"),
]

TRAIL_LENGTH = 20   # 궤적 표시 프레임 수
MIN_FLOW_MAG = 1.5  # 화살표 표시 최소 이동량 (px)


def _draw_skeleton(frame: np.ndarray, kps: dict) -> None:
    """저장된 키포인트로 간단한 skeleton 연결선 + 관절 점 그리기."""
    for a, b in CONNECTIONS:
        if a in kps and b in kps:
            cv2.line(frame, kps[a], kps[b], (180, 180, 180), 2, cv2.LINE_AA)
    for name, pt in kps.items():
        color = KEYPOINT_COLORS.get(name, (255, 255, 255))
        cv2.circle(frame, pt, 6, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_trails(frame: np.ndarray, trails: dict) -> None:
    """각 관절의 이동 궤적(trail)을 시간 흐름에 따라 페이드-인 선으로 그리기."""
    for name, trail in trails.items():
        pts = list(trail)
        color = KEYPOINT_COLORS.get(name, (255, 255, 255))
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            c = tuple(int(ch * alpha) for ch in color)
            cv2.line(frame, pts[i - 1], pts[i], c, 2, cv2.LINE_AA)


def _draw_flow_arrows(
    frame: np.ndarray,
    prev_kps: dict,
    curr_kps: dict,
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> None:
    """
    이전 프레임 → 현재 프레임 사이 각 관절의 LK optical flow 벡터를 화살표로 표시.
    MediaPipe 검출 위치를 ground-truth로 사용하고, LK가 추적에 성공한 경우에만 표시.
    """
    for name in KEYPOINT_COLORS:
        if name not in prev_kps or name not in curr_kps:
            continue

        prev_pt = np.array([[prev_kps[name]]], dtype=np.float32)
        lk_pt, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pt, None, **LK_PARAMS
        )

        color = KEYPOINT_COLORS[name]
        px, py = prev_kps[name]
        cx, cy = curr_kps[name]
        dx, dy = cx - px, cy - py
        mag = np.hypot(dx, dy)

        if mag < MIN_FLOW_MAG:
            continue

        # LK 추적 성공 여부에 따라 화살표 두께 변경
        thickness = 2 if (status is not None and status[0][0] == 1) else 1
        tip_len = min(0.4, 10.0 / max(mag, 1.0))

        # 검정 외곽선 → 컬러 화살표 (가독성)
        cv2.arrowedLine(frame, (px, py), (cx, cy), (0, 0, 0), thickness + 2,
                        cv2.LINE_AA, tipLength=tip_len)
        cv2.arrowedLine(frame, (px, py), (cx, cy), color, thickness,
                        cv2.LINE_AA, tipLength=tip_len)


def render_optflow_video(
    video_path: Path,
    output_path: Path,
    frame_keypoints: list,
) -> None:
    """
    Skeleton 키포인트 기반 Sparse Optical Flow (Lucas-Kanade) 영상 저장.

    Args:
        video_path:       원본 영상 경로
        output_path:      출력 영상 경로
        frame_keypoints:  skeleton_renderer.render_skeleton_video()의 반환값
                          List[dict | None] — 프레임별 {name: (x_px, y_px)}
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    trails: dict[str, deque] = {
        name: deque(maxlen=TRAIL_LENGTH) for name in KEYPOINT_COLORS
    }

    prev_gray = None
    prev_kps = None

    for frame_idx in tqdm(range(total_frames), desc=f"optflow:  {video_path.name}", leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 배경을 약간 어둡게 (flow/trail 가독성 향상)
        annotated = (frame * 0.6).astype(np.uint8)

        curr_kps = frame_keypoints[frame_idx] if frame_idx < len(frame_keypoints) else None

        if curr_kps is not None:
            # 궤적 업데이트
            for name, pt in curr_kps.items():
                if name in trails:
                    trails[name].append(pt)

            # 궤적 그리기
            _draw_trails(annotated, trails)

            # optical flow 화살표
            if prev_gray is not None and prev_kps is not None:
                _draw_flow_arrows(annotated, prev_kps, curr_kps, prev_gray, gray)

            # skeleton 그리기
            _draw_skeleton(annotated, curr_kps)

            # 범례
            for i, (name, color) in enumerate(KEYPOINT_COLORS.items()):
                cv2.putText(
                    annotated, name,
                    (10, 30 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA,
                )
                cv2.putText(
                    annotated, name,
                    (10, 30 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 1, cv2.LINE_AA,
                )

        prev_gray = gray
        prev_kps = curr_kps
        writer.write(annotated)

    cap.release()
    writer.release()

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.angle_calculator import compute_all_angles, LANDMARKS

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

KEYPOINT_NAMES = ["SHOULDER", "HIP", "KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]

# 각도 텍스트 색상
ANGLE_COLORS_BGR = {
    "alpha": (54, 76, 231),   # red
    "beta":  (49, 204, 46),   # green
    "gamma": (219, 152, 52),  # blue
}


def render_skeleton_video(
    video_path: Path,
    output_path: Path,
    model_complexity: int = 1,
) -> list:
    """
    MediaPipe skeleton을 오버레이한 영상을 저장하고, 프레임별 keypoint 픽셀 좌표를 반환.

    Returns:
        frame_keypoints: List[dict | None]
            프레임마다 {name: (x_px, y_px)} 딕셔너리. 랜드마크 미검출 시 None.
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

    frame_keypoints = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        for frame_idx in tqdm(range(total_frames), desc=f"skeleton: {video_path.name}", leave=False):
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            annotated = frame.copy()

            if result.pose_landmarks:
                # MediaPipe 전신 skeleton 그리기
                mp_drawing.draw_landmarks(
                    annotated,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=4
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 200, 255), thickness=2
                    ),
                )

                # 각도 계산
                angles = compute_all_angles(result.pose_landmarks.landmark)
                side = angles["side"]

                # 각도 + 사이드 텍스트 오버레이
                lines = [
                    (f"alpha : {angles['alpha']:6.1f} deg", ANGLE_COLORS_BGR["alpha"]),
                    (f"beta  : {angles['beta']:6.1f} deg",  ANGLE_COLORS_BGR["beta"]),
                    (f"gamma : {angles['gamma']:6.1f} deg", ANGLE_COLORS_BGR["gamma"]),
                    (f"side  : {side}",                     (200, 200, 200)),
                ]
                for i, (text, color) in enumerate(lines):
                    cv2.putText(
                        annotated, text,
                        (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 0, 0), 4, cv2.LINE_AA,
                    )
                    cv2.putText(
                        annotated, text,
                        (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        color, 2, cv2.LINE_AA,
                    )

                # 관심 키포인트 픽셀 좌표 저장
                kps = {}
                for name in KEYPOINT_NAMES:
                    idx = LANDMARKS[side][name]
                    lm = result.pose_landmarks.landmark[idx]
                    kps[name] = (int(lm.x * width), int(lm.y * height))
                frame_keypoints.append(kps)
            else:
                frame_keypoints.append(None)

            writer.write(annotated)

    cap.release()
    writer.release()
    return frame_keypoints

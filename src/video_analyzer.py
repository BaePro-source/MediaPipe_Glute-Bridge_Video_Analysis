import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.angle_calculator import compute_all_angles

mp_pose = mp.solutions.pose


def analyze_video(video_path: str | Path, model_complexity: int = 1) -> pd.DataFrame:
    """
    동영상을 프레임 단위로 분석하여 alpha, beta, gamma 각도를 DataFrame으로 반환.

    Returns:
        DataFrame columns: [frame, time_sec, alpha, beta, gamma]
        랜드마크 미검출 프레임은 NaN으로 채움.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    records = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        for frame_idx in tqdm(range(total_frames), desc=video_path.name, leave=False):
            ret, frame = cap.read()
            if not ret:
                break

            time_sec = frame_idx / fps if fps > 0 else 0.0

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                angles = compute_all_angles(result.pose_landmarks.landmark)
            else:
                angles = {"alpha": float("nan"), "beta": float("nan"), "gamma": float("nan")}

            records.append({
                "frame": frame_idx,
                "time_sec": round(time_sec, 4),
                **angles,
            })

    cap.release()

    df = pd.DataFrame(records)
    return df

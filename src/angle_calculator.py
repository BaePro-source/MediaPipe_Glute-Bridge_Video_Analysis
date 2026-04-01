import numpy as np


LANDMARK_INDEX = {
    "LEFT_SHOULDER": 11,
    "LEFT_HIP": 23,
    "LEFT_KNEE": 25,
    "LEFT_ANKLE": 27,
    "LEFT_HEEL": 29,
    "LEFT_FOOT_INDEX": 31,
}


def get_coord(landmarks, name: str) -> np.ndarray:
    idx = LANDMARK_INDEX[name]
    lm = landmarks[idx]
    return np.array([lm.x, lm.y, lm.z])


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """두 벡터 사이의 각도를 degree로 반환 (0~180)."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_val = np.dot(v1, v2) / (norm1 * norm2)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def three_point_angle(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """꼭짓점(vertex)에서 a-vertex-b 각도를 degree로 반환."""
    v1 = a - vertex
    v2 = b - vertex
    return angle_between_vectors(v1, v2)


def compute_alpha(landmarks) -> float:
    """
    alpha: L_KNEE->L_ANKLE 벡터와 L_FOOT_INDEX->L_HEEL 벡터 사이 각도 (작은 값).
    발목과 발의 정렬 각도.
    """
    knee = get_coord(landmarks, "LEFT_KNEE")
    ankle = get_coord(landmarks, "LEFT_ANKLE")
    foot = get_coord(landmarks, "LEFT_FOOT_INDEX")
    heel = get_coord(landmarks, "LEFT_HEEL")

    v1 = ankle - knee          # L_KNEE -> L_ANKLE
    v2 = heel - foot           # L_FOOT_INDEX -> L_HEEL

    angle = angle_between_vectors(v1, v2)
    return min(angle, 180.0 - angle)


def compute_beta(landmarks) -> float:
    """
    beta: L_KNEE -> L_HIP -> L_SHOULDER (꼭짓점: L_HIP).
    고관절 각도.
    """
    knee = get_coord(landmarks, "LEFT_KNEE")
    hip = get_coord(landmarks, "LEFT_HIP")
    shoulder = get_coord(landmarks, "LEFT_SHOULDER")
    return three_point_angle(knee, hip, shoulder)


def compute_gamma(landmarks) -> float:
    """
    gamma: L_ANKLE -> L_KNEE -> L_HIP (꼭짓점: L_KNEE).
    무릎 굴곡 각도.
    """
    ankle = get_coord(landmarks, "LEFT_ANKLE")
    knee = get_coord(landmarks, "LEFT_KNEE")
    hip = get_coord(landmarks, "LEFT_HIP")
    return three_point_angle(ankle, knee, hip)


def compute_all_angles(landmarks) -> dict:
    """한 프레임의 landmarks에서 alpha, beta, gamma를 모두 계산."""
    return {
        "alpha": compute_alpha(landmarks),
        "beta": compute_beta(landmarks),
        "gamma": compute_gamma(landmarks),
    }

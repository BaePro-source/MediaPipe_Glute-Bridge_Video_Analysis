import numpy as np


LANDMARKS = {
    "LEFT": {
        "SHOULDER":   11,
        "HIP":        23,
        "KNEE":       25,
        "ANKLE":      27,
        "HEEL":       29,
        "FOOT_INDEX": 31,
    },
    "RIGHT": {
        "SHOULDER":   12,
        "HIP":        24,
        "KNEE":       26,
        "ANKLE":      28,
        "HEEL":       30,
        "FOOT_INDEX": 32,
    },
}


def detect_side(landmarks) -> str:
    """
    LEFT_HIP vs RIGHT_HIP의 visibility를 비교해 카메라에 더 잘 보이는 쪽을 반환.
    """
    left_vis  = landmarks[LANDMARKS["LEFT"]["HIP"]].visibility
    right_vis = landmarks[LANDMARKS["RIGHT"]["HIP"]].visibility
    return "LEFT" if left_vis >= right_vis else "RIGHT"


def get_coord(landmarks, side: str, name: str) -> np.ndarray:
    idx = LANDMARKS[side][name]
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
    return angle_between_vectors(a - vertex, b - vertex)


def compute_alpha(landmarks, side: str) -> float:
    """
    alpha: KNEE->ANKLE 벡터와 FOOT_INDEX->HEEL 벡터 사이 각도.
    발목과 발의 정렬 각도.
    """
    knee  = get_coord(landmarks, side, "KNEE")
    ankle = get_coord(landmarks, side, "ANKLE")
    foot  = get_coord(landmarks, side, "FOOT_INDEX")
    heel  = get_coord(landmarks, side, "HEEL")

    angle = angle_between_vectors(ankle - knee, heel - foot)
    return min(angle, 180.0 - angle)


def compute_beta(landmarks, side: str) -> float:
    """
    beta: KNEE -> HIP -> SHOULDER (꼭짓점: HIP).
    pose-estimation 기반 근사 각도
    """
    knee     = get_coord(landmarks, side, "KNEE")
    hip      = get_coord(landmarks, side, "HIP")
    shoulder = get_coord(landmarks, side, "SHOULDER")
    return three_point_angle(knee, hip, shoulder)


def compute_gamma(landmarks, side: str) -> float:
    """
    gamma: ANKLE -> KNEE -> HIP (꼭짓점: KNEE).
    무릎 내부각(0-180).
    """
    ankle = get_coord(landmarks, side, "ANKLE")
    knee  = get_coord(landmarks, side, "KNEE")
    hip   = get_coord(landmarks, side, "HIP")
    return three_point_angle(ankle, knee, hip)


def compute_all_angles(landmarks) -> dict:
    """
    한 프레임의 landmarks에서 alpha, beta, gamma를 모두 계산.
    visibility 기반으로 좌/우 중 카메라에 잘 보이는 쪽을 자동 선택.
    """
    side = detect_side(landmarks)
    return {
        "side": side,
        "alpha": compute_alpha(landmarks, side),
        "beta":  compute_beta(landmarks, side),
        "gamma": compute_gamma(landmarks, side),
    }

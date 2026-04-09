"""
기존 optflow 영상을 두 가지 방식으로 재생성.

  _optflow_dense.mp4  — Farneback Dense + HSV  (MediaPipe 불필요)
  _optflow_sparse.mp4 — Lucas-Kanade 화살표    (MediaPipe 재실행 필요)

실행:
    python scripts/regen_optflow.py             # dense + sparse 모두
    python scripts/regen_optflow.py --dense-only  # dense만 (빠름)
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optical_flow import render_optflow_dense, render_optflow_sparse
from src.skeleton_renderer import render_skeleton_video

INPUT_DIR  = PROJECT_ROOT / 'input'
OUTPUT_DIR = PROJECT_ROOT / 'output'
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}


def find_input_video(sample_id: str) -> Path | None:
    for class_dir in INPUT_DIR.iterdir():
        if not class_dir.is_dir():
            continue
        for v in class_dir.iterdir():
            if v.suffix.lower() in VIDEO_EXTENSIONS and v.stem == sample_id:
                return v
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dense-only', action='store_true',
                        help='Dense optical flow만 재생성 (MediaPipe 실행 없음, 빠름)')
    parser.add_argument('--complexity', type=int, default=1, choices=[0, 1, 2],
                        help='MediaPipe 복잡도 (sparse 재생성 시 사용)')
    args = parser.parse_args()

    tasks = []
    for class_dir in sorted(OUTPUT_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        for subj_dir in sorted(class_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            sample_id   = subj_dir.name
            input_video = find_input_video(sample_id)
            if input_video is None:
                print(f"[SKIP] 원본 영상 없음: {sample_id}")
                continue
            tasks.append((sample_id, input_video, subj_dir))

    if not tasks:
        print("[ERROR] 재생성할 영상이 없습니다.")
        sys.exit(1)

    mode = "dense only" if args.dense_only else "dense + sparse"
    print(f"Optical Flow 재생성: {len(tasks)}개  ({mode})\n")

    for idx, (sample_id, input_video, subj_dir) in enumerate(tasks, 1):
        print(f"[{idx:02d}/{len(tasks)}] {sample_id}")

        # ── Dense (MediaPipe 불필요) ───────────────────────────────────────
        dense_path = subj_dir / f"{sample_id}_optflow_dense.mp4"
        render_optflow_dense(input_video, dense_path, frame_keypoints=None)

        if args.dense_only:
            continue

        # ── Sparse (skeleton renderer로 keypoints 재획득) ──────────────────
        skeleton_path = subj_dir / f"{sample_id}_skeleton.mp4"
        print(f"  skeleton 재렌더링 (MediaPipe)...")
        frame_keypoints = render_skeleton_video(input_video, skeleton_path, args.complexity)

        sparse_path = subj_dir / f"{sample_id}_optflow_sparse.mp4"
        render_optflow_sparse(input_video, sparse_path, frame_keypoints)

    print("\n전체 재생성 완료!")


if __name__ == '__main__':
    main()

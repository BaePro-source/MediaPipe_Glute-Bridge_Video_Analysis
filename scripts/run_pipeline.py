"""
글루트 브릿지 영상 분석 파이프라인

입력 구조:
    input/
        gb01.mp4
        gb02.mp4
        ...

출력 구조:
    output/
        gb01/
            gb01_angles.csv
            graphs/
                gb01_alpha.png
                gb01_beta.png
                gb01_gamma.png
        ...

실행:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --input input --output output --complexity 1
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.video_analyzer import analyze_video
from src.graph_plotter import plot_single_angle

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def process_video(
    video_path: Path,
    output_dir: Path,
    model_complexity: int,
) -> None:
    sample_id = video_path.stem
    print(f"\n[{sample_id}] {video_path.name} 분석 중...")

    out_sample = output_dir / sample_id
    out_sample.mkdir(parents=True, exist_ok=True)
    graph_dir = out_sample / "graphs"

    df = analyze_video(video_path, model_complexity=model_complexity)

    # CSV 저장
    csv_path = out_sample / f"{sample_id}_angles.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV 저장: {csv_path.name}")

    # 각도 그래프 (3개)
    for angle in ["alpha", "beta", "gamma"]:
        out_png = graph_dir / f"{sample_id}_{angle}.png"
        plot_single_angle(df, angle, sample_id, out_png)
        print(f"  그래프 저장: {out_png.name}")


def main():
    parser = argparse.ArgumentParser(description="글루트 브릿지 영상 각도 분석 파이프라인")
    parser.add_argument("--input", type=str, default="input",
                        help="입력 디렉토리 (기본값: input)")
    parser.add_argument("--output", type=str, default="output",
                        help="출력 디렉토리 (기본값: output)")
    parser.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                        help="MediaPipe 모델 복잡도 (0=빠름, 1=기본, 2=정확, 기본값: 1)")
    args = parser.parse_args()

    input_dir = PROJECT_ROOT / args.input
    output_dir = PROJECT_ROOT / args.output

    if not input_dir.exists():
        print(f"[ERROR] 입력 디렉토리가 없습니다: {input_dir}")
        sys.exit(1)

    video_files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_files:
        print(f"[ERROR] 입력 디렉토리에 영상 파일이 없습니다: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"총 {len(video_files)}개 영상 처리 시작")
    print(f"입력: {input_dir}")
    print(f"출력: {output_dir}")
    print(f"모델 복잡도: {args.complexity}")

    for video_path in video_files:
        process_video(video_path, output_dir, args.complexity)

    print("\n전체 처리 완료!")


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.video_analyzer import analyze_video
from src.graph_plotter import plot_single_angle
from src.skeleton_renderer import render_skeleton_video
from src.optical_flow import render_optflow_video

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def is_done(video_path: Path, out_video_dir: Path, with_video: bool) -> bool:
    """해당 영상의 결과물이 이미 존재하는지 확인."""
    sample_id = video_path.stem
    csv = out_video_dir / f"{sample_id}_angles.csv"
    graphs = [out_video_dir / "graphs" / f"{sample_id}_{angle}.png" for angle in ["alpha", "beta", "gamma"]]
    base_done = csv.exists() and all(g.exists() for g in graphs)
    if not with_video:
        return base_done
    skeleton_mp4 = out_video_dir / f"{sample_id}_skeleton.mp4"
    optflow_mp4  = out_video_dir / f"{sample_id}_optflow.mp4"
    return base_done and skeleton_mp4.exists() and optflow_mp4.exists()


def process_video(
    video_path: Path,
    out_video_dir: Path,
    model_complexity: int,
    with_video: bool,
) -> None:
    sample_id = video_path.stem
    print(f"\n[{sample_id}] {video_path.name} 분석 중...")

    out_video_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = out_video_dir / "graphs"

    df = analyze_video(video_path, model_complexity=model_complexity)

    # CSV 저장
    csv_path = out_video_dir / f"{sample_id}_angles.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV 저장: {csv_path.name}")

    # 각도 그래프 (3개)
    for angle in ["alpha", "beta", "gamma"]:
        out_png = graph_dir / f"{sample_id}_{angle}.png"
        plot_single_angle(df, angle, sample_id, out_png)
        print(f"  그래프 저장: {out_png.name}")

    if not with_video:
        return

    # Skeleton 영상 + 키포인트 좌표 획득
    skeleton_path = out_video_dir / f"{sample_id}_skeleton.mp4"
    frame_keypoints = render_skeleton_video(video_path, skeleton_path, model_complexity)
    print(f"  Skeleton 영상 저장: {skeleton_path.name}")

    # Optical Flow 영상
    optflow_path = out_video_dir / f"{sample_id}_optflow.mp4"
    render_optflow_video(video_path, optflow_path, frame_keypoints)
    print(f"  Optical Flow 영상 저장: {optflow_path.name}")


def main():
    parser = argparse.ArgumentParser(description="글루트 브릿지 영상 각도 분석 파이프라인")
    parser.add_argument("--input", type=str, default="input",
                        help="입력 디렉토리 (기본값: input)")
    parser.add_argument("--output", type=str, default="output",
                        help="출력 디렉토리 (기본값: output)")
    parser.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                        help="MediaPipe 모델 복잡도 (0=빠름, 1=기본, 2=정확, 기본값: 1)")
    parser.add_argument("--only-new", action="store_true",
                        help="결과물이 이미 있는 영상은 건너뜀")
    parser.add_argument("--no-video", action="store_true",
                        help="skeleton/optical flow 영상 생성 건너뜀 (CSV + 그래프만)")
    args = parser.parse_args()

    input_dir = PROJECT_ROOT / args.input
    output_dir = PROJECT_ROOT / args.output

    if not input_dir.exists():
        print(f"[ERROR] 입력 디렉토리가 없습니다: {input_dir}")
        sys.exit(1)

    # input/best/, input/worst/ 클래스 폴더들을 순회
    class_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())

    if not class_dirs:
        print(f"[ERROR] 입력 디렉토리에 하위 폴더가 없습니다: {input_dir}")
        sys.exit(1)

    # 각 클래스 폴더 안의 영상 파일 수집: (video_path, out_video_dir) 쌍
    tasks = []
    for class_dir in class_dirs:
        video_files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        for video_path in video_files:
            out_video_dir = output_dir / class_dir.name / video_path.stem
            tasks.append((video_path, out_video_dir))

    if not tasks:
        print(f"[ERROR] best/worst 폴더에 영상 파일이 없습니다: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    with_video = not args.no_video

    print(f"총 {len(tasks)}개 영상 처리 시작")
    print(f"입력: {input_dir}")
    print(f"출력: {output_dir}")
    print(f"모델 복잡도: {args.complexity}")
    print(f"영상 출력: {'OFF (--no-video)' if not with_video else 'ON (skeleton + optflow)'}")

    for video_path, out_video_dir in tasks:
        if args.only_new and is_done(video_path, out_video_dir, with_video):
            print(f"\n[{video_path.stem}] 결과물 존재 — 건너뜀")
            continue
        process_video(video_path, out_video_dir, args.complexity, with_video)

    print("\n전체 처리 완료!")


if __name__ == "__main__":
    main()

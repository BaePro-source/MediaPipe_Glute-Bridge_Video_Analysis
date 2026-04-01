import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


ANGLE_COLORS = {
    "alpha": "#E74C3C",
    "beta":  "#2ECC71",
    "gamma": "#3498DB",
}


def plot_single_angle(
    df: pd.DataFrame,
    angle: str,
    sample_id: str,
    output_path: Path,
) -> None:
    """
    단일 각도 그래프 저장.
    X축: 시간(초), Y축: 각도(degree)
    최댓값/최솟값 표시.
    """
    series = df[angle].dropna()
    if series.empty:
        print(f"  [SKIP] {sample_id}/{angle}: 유효 데이터 없음")
        return

    time = df.loc[series.index, "time_sec"]
    if "time_sec" not in df.columns:
        raise ValueError("DataFrame must contain 'time_sec' column")
    
    max_val = series.max()
    min_val = series.min()
    max_t = time[series.idxmax()]
    min_t = time[series.idxmin()]

    color = ANGLE_COLORS[angle]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(time, series, color=color, linewidth=1.5, label=angle)

    # 최댓값 마커
    ax.annotate(
        f"Max: {max_val:.1f}°",
        xy=(max_t, max_val),
        xytext=(max_t, max_val + 3),
        ha="center", fontsize=8, color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
    )
    # 최솟값 마커
    ax.annotate(
        f"Min: {min_val:.1f}°",
        xy=(min_t, min_val),
        xytext=(min_t, min_val - 5),
        ha="center", fontsize=8, color="blue",
        arrowprops=dict(arrowstyle="->", color="blue", lw=0.8),
    )

    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Angle (°)", fontsize=10)
    ax.set_title(
        f"{sample_id} | {angle.upper()}\n"
        f"Max: {max_val:.1f}°  Min: {min_val:.1f}°  Range: {max_val - min_val:.1f}°",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

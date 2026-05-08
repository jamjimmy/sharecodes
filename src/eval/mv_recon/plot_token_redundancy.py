import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def _collect_series(records, metric_key):
    x = []
    y_full = []
    y_filtered = []
    for rec in records:
        full_v = rec["full"].get(metric_key)
        filtered_v = rec["filtered"].get(metric_key)
        if full_v is None or filtered_v is None:
            continue
        x.append(rec["frame_idx_curr"])
        y_full.append(float(full_v))
        y_filtered.append(float(filtered_v))
    return np.asarray(x), np.asarray(y_full), np.asarray(y_filtered)


def moving_average(y, window=5):
    if window <= 1 or len(y) < window:
        return y.copy()
    pad = window // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y_pad, kernel, mode="valid")
    return y_smooth[: len(y)]


def setup_plot_style():
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.2,
            "legend.frameon": False,
        }
    )


def beautify_metric_name(metric):
    mapping = {
        "cos_mean_all_pairs": "Mean Cosine of All Pairs",
        "cos_mean_of_curr_max": "Mean Max Cosine Similarity of Tokens",
        "cos_p95_of_curr_max": "95th Percentile of Max Cosine",
    }
    return mapping.get(metric, metric)
def compute_local_stats(y, window=5):
    if len(y) < window:
        return y, np.zeros_like(y)

    pad = window // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")

    means = []
    stds = []

    for i in range(len(y)):
        segment = y_pad[i:i + window]
        means.append(segment.mean())
        stds.append(segment.std())

    return np.array(means), np.array(stds)
def plot_curve_with_ci(
    x,
    y_full,
    y_filtered,
    metric,
    out_path,
    window=5,
):
    color_full = "#4A90E2"
    color_filtered = "#7DAE7D"
    # 7DAE7D

    mean_full, std_full = compute_local_stats(y_full, window)
    mean_filtered, std_filtered = compute_local_stats(y_filtered, window)

    plt.figure(figsize=(7.0, 4.2))

    # ===== Full tokens =====
    plt.plot(
        x,
        mean_full,
        color=color_full,
        label="Full tokens",
        linewidth=2.5,
    )
    plt.fill_between(
        x,
        mean_full - std_full,
        mean_full + std_full,
        color=color_full,
        alpha=0.18,
    )

    # ===== Filtered tokens =====
    plt.plot(
        x,
        mean_filtered,
        color=color_filtered,
        label="Filtered tokens",
        linewidth=2.5,
    )
    plt.fill_between(
        x,
        mean_filtered - std_filtered,
        mean_filtered + std_filtered,
        color=color_filtered,
        alpha=0.18,
    )

    plt.xlabel("Current Frame Index")
    plt.ylabel("Cosine Similarity")
    plt.title(beautify_metric_name(metric))

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    # 去掉右上边框（关键！）
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
def plot_curve(
    x,
    y_full,
    y_filtered,
    metric,
    out_path,
    smooth_window=5,
    show_raw=True,
):
    # 论文风格配色
    color_full = "#4C78A8"      # muted blue
    color_filtered = "#E45756"  # muted red

    y_full_s = moving_average(y_full, smooth_window)
    y_filtered_s = moving_average(y_filtered, smooth_window)

    plt.figure(figsize=(7.0, 4.2))

    if show_raw:
        plt.plot(
            x,
            y_full,
            color=color_full,
            alpha=0.22,
            linewidth=1.2,
        )
        plt.plot(
            x,
            y_filtered,
            color=color_filtered,
            alpha=0.22,
            linewidth=1.2,
        )

    plt.plot(
        x,
        y_full_s,
        color=color_full,
        label="Full tokens",
        linewidth=2.4,
    )
    plt.plot(
        x,
        y_filtered_s,
        color=color_filtered,
        label="Filtered tokens",
        linewidth=2.4,
    )

    plt.xlabel("Current Frame Index")
    plt.ylabel("Cosine Similarity")
    plt.title(beautify_metric_name(metric))
    plt.grid(True, linestyle="--")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_summary_bar(summary, out_path):
    labels = list(summary.keys())
    full_vals = [summary[k]["full_mean"] for k in labels]
    filtered_vals = [summary[k]["filtered_mean"] for k in labels]

    labels_pretty = [beautify_metric_name(k) for k in labels]

    x = np.arange(len(labels))
    width = 0.34

    color_full = "#4C78A8"
    color_filtered = "#E45756"

    plt.figure(figsize=(7.2, 4.6))
    bars1 = plt.bar(x - width / 2, full_vals, width, label="Full tokens", color=color_full)
    bars2 = plt.bar(
        x + width / 2, filtered_vals, width, label="Filtered tokens", color=color_filtered
    )

    plt.xticks(x, labels_pretty, rotation=12, ha="right")
    plt.ylabel("Sequence Mean")
    plt.title("Adjacent-frame Cosine Statistics Summary")
    plt.grid(axis="y", linestyle="--")

    # 数值标注
    for bars in [bars1, bars2]:
        for b in bars:
            h = b.get_height()
            plt.text(
                b.get_x() + b.get_width() / 2,
                h,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot adjacent-frame token cosine similarity statistics."
    )
    parser.add_argument(
        "--stats-json",
        type=str,
        required=True,
        help="Path to token_redundancy_stats.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for figures (default: same as stats json dir).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Window size for moving average smoothing.",
    )
    parser.add_argument(
        "--hide-raw",
        action="store_true",
        help="Hide raw curves and only show smoothed curves.",
    )
    args = parser.parse_args()

    setup_plot_style()

    with open(args.stats_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    if len(records) == 0:
        raise ValueError("No records found in stats json.")

    out_dir = args.out_dir or os.path.dirname(args.stats_json)
    os.makedirs(out_dir, exist_ok=True)

    metrics = [
        "cos_mean_all_pairs",
        "cos_mean_of_curr_max",
        "cos_p95_of_curr_max",
    ]

    for metric in metrics:
        x, y_full, y_filtered = _collect_series(records, metric)
        if x.size == 0:
            continue
        plot_curve_with_ci(
            x,
            y_full,
            y_filtered,
            metric,
            os.path.join(out_dir, f"{metric}_curve.png"),
            window=args.smooth_window,
        )
        
        # plot_curve(
        #     x=x,
        #     y_full=y_full,
        #     y_filtered=y_filtered,
        #     metric=metric,
        #     out_path=os.path.join(out_dir, f"{metric}_curve.png"),
        #     smooth_window=args.smooth_window,
        #     show_raw=not args.hide_raw,
        # )

    summary = {}
    for metric in metrics:
        _, y_full, y_filtered = _collect_series(records, metric)
        if y_full.size == 0:
            continue
        summary[metric] = {
            "full_mean": float(y_full.mean()),
            "filtered_mean": float(y_filtered.mean()),
        }

    if summary:
        plot_summary_bar(
            summary,
            os.path.join(out_dir, "token_redundancy_summary_bar.png"),
        )

        with open(
            os.path.join(out_dir, "token_redundancy_summary.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Done. Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
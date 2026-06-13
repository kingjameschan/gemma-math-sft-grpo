"""Plot per-question pass@1 distribution with Easy/Medium/Hard bucket overlay.

Shows:
  - histogram of base IT model's per-question pass@1 (5% bins)
  - colored regions for Easy / Medium / Hard cuts
  - cumulative distribution overlay

Usage:
  python3 v3/tools/_plot_difficulty_buckets.py path/to/pass_at_k.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pass_at_k_json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fp = Path(args.pass_at_k_json)
    d = json.load(open(fp))
    samples = d["samples"]
    K = d["config"]["K"]
    pass1 = np.array([s["any_correct_per_K"] / K for s in samples])
    n = len(pass1)

    n_easy = int((pass1 >= 0.9).sum())
    n_hard = int((pass1 <= 0.1).sum())
    n_med = n - n_easy - n_hard

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # === Left: histogram with bucket coloring ===
    ax = axes[0]
    bins = np.linspace(0, 1.0001, 21)  # 5% bins
    centers = (bins[:-1] + bins[1:]) / 2
    width = bins[1] - bins[0]
    counts, _ = np.histogram(pass1, bins=bins)

    bar_colors = []
    for c in centers:
        if c >= 0.9:
            bar_colors.append("#16a34a")  # Easy green
        elif c <= 0.1:
            bar_colors.append("#dc2626")  # Hard red
        else:
            bar_colors.append("#f59e0b")  # Medium amber

    ax.bar(centers, counts, width=width * 0.95, color=bar_colors, alpha=0.85,
           edgecolor="white", linewidth=0.4)

    # Bucket boundaries
    ax.axvspan(0, 0.1, color="#dc2626", alpha=0.08, zorder=0)
    ax.axvspan(0.1, 0.9, color="#f59e0b", alpha=0.05, zorder=0)
    ax.axvspan(0.9, 1.0, color="#16a34a", alpha=0.08, zorder=0)
    ax.axvline(0.1, color="#666", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(0.9, color="#666", linestyle="--", linewidth=0.8, alpha=0.6)

    # Annotations
    ymax = ax.get_ylim()[1]
    ax.text(0.05, ymax * 0.85, f"Hard\n{n_hard}\n({n_hard/n*100:.1f}%)",
            ha="center", fontsize=10, color="#7c2d12", fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#dc2626"))
    ax.text(0.5, ymax * 0.55, f"Medium\n{n_med}\n({n_med/n*100:.1f}%)",
            ha="center", fontsize=10, color="#92400e", fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#f59e0b"))
    ax.text(0.95, ymax * 0.85, f"Easy\n{n_easy}\n({n_easy/n*100:.1f}%)",
            ha="center", fontsize=10, color="#14532d", fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#16a34a"))

    ax.set_xlabel("base IT pass@1 (per question, K=64)")
    ax.set_ylabel("# questions")
    ax.set_title(f"(a) per-question pass@1 distribution + bucket cuts (n={n})",
                 loc="left", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 460)   # match SFT difficulty grid scale for cross-plot comparison
    ax.grid(axis="y", alpha=0.3)

    # === Right: cumulative distribution ===
    ax = axes[1]
    sorted_p1 = np.sort(pass1)
    cum = np.arange(1, n + 1) / n * 100
    ax.plot(sorted_p1, cum, "-", color="#2563eb", linewidth=2)
    ax.fill_between(sorted_p1, 0, cum, color="#2563eb", alpha=0.15)

    # Bucket region shading
    ax.axvspan(0, 0.1, color="#dc2626", alpha=0.08)
    ax.axvspan(0.1, 0.9, color="#f59e0b", alpha=0.05)
    ax.axvspan(0.9, 1.0, color="#16a34a", alpha=0.08)
    ax.axvline(0.1, color="#666", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(0.9, color="#666", linestyle="--", linewidth=0.8, alpha=0.6)

    # Mark the cumulative% at thresholds
    cum_at_0_1 = (pass1 <= 0.1).sum() / n * 100
    cum_at_0_9 = (pass1 < 0.9).sum() / n * 100
    ax.scatter([0.1, 0.9], [cum_at_0_1, cum_at_0_9], color="#2563eb",
               s=60, zorder=5, edgecolor="white")
    ax.annotate(f"  {cum_at_0_1:.1f}%", xy=(0.1, cum_at_0_1), fontsize=10,
                ha="left", va="bottom", color="#2563eb")
    ax.annotate(f"  {cum_at_0_9:.1f}%", xy=(0.9, cum_at_0_9), fontsize=10,
                ha="left", va="bottom", color="#2563eb")

    # Median + mean reference
    median = np.median(pass1)
    mean = pass1.mean()
    ax.axhline(50, color="#aaa", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.text(0.02, 52, f"median = {median:.3f}", fontsize=9, color="#666")
    ax.text(0.02, 47, f"mean = {mean:.3f}", fontsize=9, color="#666")

    ax.set_xlabel("base IT pass@1")
    ax.set_ylabel("cumulative % of questions ≤ x")
    ax.set_title("(b) cumulative distribution (CDF)", loc="left", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)

    fig.suptitle(f"v3 difficulty bucketing — Gemma2-2B-IT base on GSM8K test (K=64)",
                 fontsize=12, fontweight="semibold", y=1.0)
    plt.tight_layout()

    out_path = Path(args.out) if args.out else fp.parent / f"{fp.stem}_difficulty_buckets.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

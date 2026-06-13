"""Plot truncation rates by difficulty bucket from e1_baseline_summary.json.

Shows the key finding: Hard questions hit max_new_tokens ceiling much more often
than Easy questions (Gemma2-IT whitespace-tail bug), with both response-level
and question-level breakdowns.

Usage:
  python3 v3/tools/_plot_truncation_rates.py
  python3 v3/tools/_plot_truncation_rates.py --summary v3/outputs/e1_baseline_summary.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUMMARY = ROOT / "v3" / "E1_baseline" / "outputs" / "e1_baseline_summary.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fp = Path(args.summary)
    d = json.load(open(fp))
    tr = d["truncation_rates"]

    buckets = ["Easy", "Medium", "Hard"]
    bucket_colors = {"Easy": "#16a34a", "Medium": "#f59e0b", "Hard": "#dc2626",
                     "overall": "#2563eb"}

    # Per-bucket numbers
    resp_rates = [tr["by_bucket"][b]["response_truncation_rate"] * 100 for b in buckets]
    q_rates = [tr["by_bucket"][b]["question_truncation_rate"] * 100 for b in buckets]
    n_resp_trunc = [tr["by_bucket"][b]["n_truncated_responses"] for b in buckets]
    n_resp_total = [tr["by_bucket"][b]["n_responses"] for b in buckets]
    n_q_trunc = [tr["by_bucket"][b]["n_questions_with_any_truncated"] for b in buckets]
    n_q_total = [tr["by_bucket"][b]["n_questions"] for b in buckets]
    overall_resp_rate = tr["overall"]["response_truncation_rate"] * 100
    overall_q_rate = tr["overall"]["question_truncation_rate"] * 100

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # === Panel (a): Response-level truncation rate ===
    ax = axes[0]
    x = np.arange(len(buckets))
    bars = ax.bar(x, resp_rates,
                  color=[bucket_colors[b] for b in buckets],
                  alpha=0.85, edgecolor="white", linewidth=0.6)
    # Annotate bars with absolute counts
    for i, (rate, nt, n) in enumerate(zip(resp_rates, n_resp_trunc, n_resp_total)):
        ax.text(i, rate + 0.012, f"{rate:.3f}%\n({nt}/{n})",
                ha="center", va="bottom", fontsize=9, fontweight="semibold")
    # Overall reference line
    ax.axhline(overall_resp_rate, color="#2563eb", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(2.4, overall_resp_rate + 0.005, f"overall {overall_resp_rate:.3f}%",
            color="#2563eb", fontsize=9, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylabel("response-level truncation rate (%)")
    ax.set_title("(a) response-level: % of K=64 responses hitting 1024 ceiling",
                 loc="left", fontsize=11)
    ax.set_ylim(0, max(resp_rates) * 1.45)
    ax.grid(axis="y", alpha=0.3)

    # === Panel (b): Question-level truncation rate ===
    ax = axes[1]
    bars = ax.bar(x, q_rates,
                  color=[bucket_colors[b] for b in buckets],
                  alpha=0.85, edgecolor="white", linewidth=0.6)
    for i, (rate, nt, n) in enumerate(zip(q_rates, n_q_trunc, n_q_total)):
        ax.text(i, rate + 0.4, f"{rate:.2f}%\n({nt}/{n})",
                ha="center", va="bottom", fontsize=9, fontweight="semibold")
    ax.axhline(overall_q_rate, color="#2563eb", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(2.4, overall_q_rate + 0.3, f"overall {overall_q_rate:.2f}%",
            color="#2563eb", fontsize=9, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylabel("question-level truncation rate (%)")
    ax.set_title("(b) question-level: % of questions with ≥ 1/64 hitting ceiling",
                 loc="left", fontsize=11)
    ax.set_ylim(0, max(q_rates) * 1.25)
    ax.grid(axis="y", alpha=0.3)

    # Cross-bucket gradient annotation
    if q_rates[0] > 0:
        ratio_hard_easy = q_rates[2] / q_rates[0]
        ax.annotate(
            f"Hard / Easy ratio = {ratio_hard_easy:.1f}x\n(Hard questions trigger whitespace-tail bug more)",
            xy=(2, q_rates[2]), xytext=(0.5, q_rates[2] * 0.7),
            fontsize=9.5, color="#7c2d12", fontweight="semibold",
            arrowprops=dict(arrowstyle="->", color="#7c2d12", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef2f2",
                      edgecolor="#dc2626", linewidth=1),
        )

    fig.suptitle(f"E1 baseline — truncation rate by difficulty bucket "
                 f"(K={d['K']}, n_q={d['n_questions']}, max_new_tokens=1024)",
                 fontsize=12, fontweight="semibold", y=1.0)
    plt.tight_layout()

    out_path = Path(args.out) if args.out else fp.parent / "e1_truncation_rates.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

"""SFT K=64 D_test: per-question pass@K=64 distribution + difficulty bucket cuts.

For each LR (1e-3, 1e-4, 5e-4): 10 panels (one per ckpt step) with E/M/H overlay.
Renders 3 separate PNG files.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "v3" / "E2_sft" / "outputs" / "test_eval_k64"
LRS = ["1e-4", "5e-4", "1e-3"]
STEPS = [10, 30, 50, 70, 90, 110, 130, 150, 170, 186]
K = 64

# Bucket thresholds (E1 baseline convention)
EASY_T = 0.9   # >= 90% of K samples correct
HARD_T = 0.1   # <= 10% of K samples correct


def load_ckpt(lr, step):
    fp = EVAL_DIR / f"sft_lr{lr}_r64_checkpoint-{step}.json"
    return json.load(open(fp))


def render_one_lr(lr):
    out_file = ROOT / "v3" / "E2_sft" / "outputs" / f"lr{lr.replace('-','-')}_difficulty_grid.png"
    out_file = ROOT / "v3" / "E2_sft" / "outputs" / f"lr{lr}_difficulty_grid.png"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)

    table_rows = []
    for ax, step in zip(axes.flat, STEPS):
        d = load_ckpt(lr, step)
        cc = np.array(d["per_question_correct_count"])
        p1 = cc / K   # per-question pass@K (frac of K correct)
        n = len(p1)
        n_easy = int((p1 >= EASY_T).sum())
        n_hard = int((p1 <= HARD_T).sum())
        n_med = n - n_easy - n_hard

        bins = np.linspace(0, 1.0001, 21)  # 5% bins
        centers = (bins[:-1] + bins[1:]) / 2
        width = bins[1] - bins[0]
        counts, _ = np.histogram(p1, bins=bins)
        bar_colors = ["#dc2626" if c <= HARD_T
                      else "#16a34a" if c >= EASY_T
                      else "#f59e0b" for c in centers]
        ax.bar(centers, counts, width=width * 0.95, color=bar_colors, alpha=0.85,
               edgecolor="white", linewidth=0.4)

        ax.axvspan(0, HARD_T, color="#dc2626", alpha=0.06, zorder=0)
        ax.axvspan(EASY_T, 1.0, color="#16a34a", alpha=0.06, zorder=0)
        ax.axvline(HARD_T, color="#666", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.axvline(EASY_T, color="#666", linestyle="--", linewidth=0.6, alpha=0.5)

        # Inset stats
        ax.set_title(f"step={step}  E:{n_easy} M:{n_med} H:{n_hard}",
                     loc="left", fontsize=9, fontweight="semibold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 460)   # match base IT max bin (440 @ pass@K≈1.0) for cross-plot comparison
        ax.grid(axis="y", alpha=0.25)

        table_rows.append((step, n_easy, n_med, n_hard,
                           p1.mean()*100, np.percentile(p1, 50)*100))

    for ax in axes[1, :]:
        ax.set_xlabel("per-question pass@64 (correct/K)")
    for ax in axes[:, 0]:
        ax.set_ylabel("# questions")

    fig.suptitle(f"lr={lr} r=64 K=64 D_test — per-question pass@64 distribution + buckets (Hard ≤ 0.1, Easy ≥ 0.9)",
                 fontsize=12, fontweight="semibold")
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"saved: {out_file}\n")

    # Print table
    print(f"=== lr={lr} ===")
    print(f"{'step':>4} | {'Easy':>4} {'Med':>4} {'Hard':>4}  | {'mean p1':>7} {'med p1':>6}")
    print("-" * 50)
    for r in table_rows:
        print(f"{r[0]:>4} | {r[1]:>4} {r[2]:>4} {r[3]:>4}  | {r[4]:>6.1f}% {r[5]:>5.1f}%")
    print()
    plt.close()


def main():
    # Print BASE IT once
    bf = ROOT / "v3" / "E1_baseline" / "outputs" / "pass_at_k_20260427_222954" / "base_gemma-2-2b-it_k64.json"
    if bf.exists():
        bd = json.load(open(bf))
        if "samples" in bd:
            bp1 = np.array([s["any_correct_per_K"] / K for s in bd["samples"]])
            n_b = len(bp1)
            ne_b = (bp1 >= EASY_T).sum()
            nh_b = (bp1 <= HARD_T).sum()
            nm_b = n_b - ne_b - nh_b
            print(f"=== BASE IT (reference) ===")
            print(f"BASE | {int(ne_b):>4} {int(nm_b):>4} {int(nh_b):>4}  | {bp1.mean()*100:>6.1f}% {np.percentile(bp1,50)*100:>5.1f}%\n")
    for lr in LRS:
        render_one_lr(lr)


if __name__ == "__main__":
    main()

"""Per-question answer entropy: DAPO ck-15 vs base IT.

Verifies the "sharpening" hypothesis by comparing the entropy of the
empirical answer distribution per question (computed from K=64 samples).

If RL sharpens the output distribution around the correct mode, we expect
DAPO entropy to be systematically lower than base IT entropy.

base entropy: from v3/E1_baseline/outputs/.../base_gemma-2-2b-it_k64_curves.json
DAPO any_preds: from v3/E5_grpo/outputs/k64_dapo_ck15/r15_dapo_checkpoint-15_k64.json

Same formula as E1 _compute_pass_maj_curves.py line 172-175 (raw any_preds).
"""
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE_CURVES = ROOT / "v3" / "E1_baseline" / "outputs" / "pass_at_k_20260427_222954" / "base_gemma-2-2b-it_k64_curves.json"
DAPO_K64 = ROOT / "v3" / "E5_grpo" / "outputs" / "k64_dapo_ck15" / "r15_dapo_checkpoint-15_k64.json"
OUT_FILE = ROOT / "v3" / "E5_grpo" / "outputs" / "k64_dapo_ck15" / "entropy_dapo_vs_base.png"


def entropy_from_preds(preds):
    """Same formula as E1 line 172-175. Raw any_preds, no normalization."""
    cnt = Counter(preds)
    n = sum(cnt.values())
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log(c / n) for c in cnt.values() if c > 0)


def main():
    # base IT: per-question H already saved in curves JSON
    base = json.load(open(BASE_CURVES))
    base_H = base["entropy"]["per_question_H_any"]

    # DAPO: compute from samples
    dapo = json.load(open(DAPO_K64))
    dapo_H = [entropy_from_preds(s["any_preds"]) for s in dapo["samples"]]

    n_base = len(base_H)
    n_dapo = len(dapo_H)
    print(f"n_base={n_base}  n_dapo={n_dapo}")

    print("\n=== entropy stats (nats, per-question H over 64 sample any_preds) ===")
    for name, H in [("base IT", base_H), ("DAPO ck-15", dapo_H)]:
        print(f"  {name:>12s}: mean={statistics.mean(H):.3f}  median={statistics.median(H):.3f}  "
              f"std={statistics.stdev(H):.3f}  max={max(H):.3f}  "
              f"frac(H=0)={sum(1 for h in H if h < 1e-6)/len(H):.3f}")

    delta_mean = statistics.mean(dapo_H) - statistics.mean(base_H)
    delta_median = statistics.median(dapo_H) - statistics.median(base_H)
    print(f"\n  Δmean   = {delta_mean:+.3f} nats  ({delta_mean/statistics.mean(base_H)*100:+.1f}%)")
    print(f"  Δmedian = {delta_median:+.3f} nats")
    if delta_mean < -0.05:
        print(f"  → DAPO entropy systematically lower → SHARPENING confirmed ✓")

    # Per-question paired comparison (same question id in both)
    if n_base == n_dapo:
        per_q_delta = [d - b for d, b in zip(dapo_H, base_H)]
        n_lower = sum(1 for d in per_q_delta if d < 0)
        n_higher = sum(1 for d in per_q_delta if d > 0)
        n_same = sum(1 for d in per_q_delta if d == 0)
        print(f"\n  per-question (paired): DAPO H < base H  in {n_lower}/{n_dapo} ({n_lower/n_dapo*100:.1f}%)")
        print(f"                          DAPO H > base H  in {n_higher}/{n_dapo} ({n_higher/n_dapo*100:.1f}%)")
        print(f"                          DAPO H = base H  in {n_same}/{n_dapo} ({n_same/n_dapo*100:.1f}%)")

    # Plot histogram comparison
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: overlapping histograms
    ax = axes[0]
    bins = np.linspace(0, max(max(base_H), max(dapo_H)) + 0.1, 40)
    ax.hist(base_H, bins=bins, color="black", alpha=0.45, label=f"base IT  (mean={statistics.mean(base_H):.3f})", density=True)
    ax.hist(dapo_H, bins=bins, color="#dc2626", alpha=0.55, label=f"DAPO ck-15  (mean={statistics.mean(dapo_H):.3f})", density=True)
    ax.axvline(statistics.mean(base_H), color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(statistics.mean(dapo_H), color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("per-question answer entropy H (nats)")
    ax.set_ylabel("density")
    ax.set_title("Per-question entropy distribution (over 64 sample any_preds)",
                 fontsize=10.5, loc="left", fontweight="semibold")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", fontsize=9)

    # Right: paired delta H per question (sorted)
    if n_base == n_dapo:
        ax = axes[1]
        sorted_delta = sorted(per_q_delta)
        ax.plot(range(len(sorted_delta)), sorted_delta, "-", color="#2563eb", linewidth=1.2)
        ax.axhline(0, color="black", linewidth=1.0, alpha=0.6)
        ax.fill_between(range(len(sorted_delta)),
                        sorted_delta, 0,
                        where=[d < 0 for d in sorted_delta],
                        color="#16a34a", alpha=0.30, label=f"H ↓ (sharpened, {n_lower})")
        ax.fill_between(range(len(sorted_delta)),
                        sorted_delta, 0,
                        where=[d > 0 for d in sorted_delta],
                        color="#dc2626", alpha=0.30, label=f"H ↑ (diffused, {n_higher})")
        ax.set_xlabel("question rank (sorted by ΔH)")
        ax.set_ylabel("ΔH = H_DAPO − H_base  (nats)")
        ax.set_title(f"Paired per-question ΔH  (mean={delta_mean:+.3f})",
                     fontsize=10.5, loc="left", fontweight="semibold")
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"\nsaved: {OUT_FILE}")


if __name__ == "__main__":
    main()

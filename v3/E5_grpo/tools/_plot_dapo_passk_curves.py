"""DAPO ck-15 vs base IT — pass@K + maj@K curves on D_test (K=64 sampling).

base IT data:  v3/E1_baseline/outputs/pass_at_k_20260427_222954/base_gemma-2-2b-it_k64_curves.json
DAPO ck-15:    v3/E5_grpo/outputs/k64_dapo_ck15/r15_dapo_checkpoint-15_k64.json
Sampling:      T=0.7, top_p=0.95, n=64 (Codex unbiased pass@k)
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
BASE_CURVES = ROOT / "v3" / "E1_baseline" / "outputs" / "pass_at_k_20260427_222954" / "base_gemma-2-2b-it_k64_curves.json"
DAPO_K64 = ROOT / "v3" / "E5_grpo" / "outputs" / "k64_dapo_ck15" / "r15_dapo_checkpoint-15_k64.json"
OUT_FILE = ROOT / "v3" / "E5_grpo" / "outputs" / "k64_dapo_ck15" / "passk_majk_dapo_vs_base.png"

KS = [1, 2, 4, 8, 16, 32, 64]


def load_dapo_curves(jf):
    """Extract pass@k for k in KS from DAPO metrics dict."""
    j = json.load(open(jf))
    m = j["metrics"]
    passk = {k: m[f"pass_at_{k}_numeric"] * 100 for k in KS}
    # maj@k only at K=64 (single-trial); approximate maj@1=pass@1
    majk = {1: m["pass_at_1_numeric"] * 100,
            64: m["maj_at_64_numeric"] * 100}
    return passk, majk


def main():
    base = json.load(open(BASE_CURVES))
    base_passk = {int(k): v * 100 for k, v in base["pass_at_k_numeric"].items()}
    base_majk = {int(k): v * 100 for k, v in base["maj_at_k_numeric"].items()}

    dapo_passk, dapo_majk = load_dapo_curves(DAPO_K64)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    # Base IT (black)
    ks_b = sorted(base_passk.keys())
    ax.plot(ks_b, [base_passk[k] for k in ks_b], "-",
            color="black", markersize=6, linewidth=1.8, label="base IT  pass@K", zorder=3)
    ax.plot(ks_b, [base_majk[k] for k in ks_b], "--",
            color="black", markersize=5, linewidth=1.4, alpha=0.7,
            label="base IT  maj@K", zorder=3)

    # DAPO ck-15 (red)
    ks_d = sorted(dapo_passk.keys())
    ax.plot(ks_d, [dapo_passk[k] for k in ks_d], "-",
            color="#dc2626", markersize=6, linewidth=1.8, label="DAPO ck-15  pass@K", zorder=4)
    # maj@K only at k=1 and k=64 → just points
    maj_xs = sorted(dapo_majk.keys())
    ax.plot(maj_xs, [dapo_majk[k] for k in maj_xs], "--",
            color="#dc2626", markersize=5, linewidth=1.4, alpha=0.7,
            label="DAPO ck-15  maj@K", zorder=4)

    ax.set_xscale("log", base=2)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("K (samples per question)")
    ax.set_ylabel("accuracy on D_test (%)")
    ax.set_title("DAPO ck-15 vs base IT — pass@K & maj@K (T=0.7, top_p=0.95, n=64)",
                 fontsize=11, fontweight="semibold", loc="left")
    ax.grid(alpha=0.3, linestyle=":")
    ax.set_ylim(55, 95)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)

    # Annotate key deltas
    txt = (f"Δpass@1  : {dapo_passk[1] - base_passk[1]:+.2f}pp  "
           f"({base_passk[1]:.2f}% → {dapo_passk[1]:.2f}%)\n"
           f"Δpass@64 : {dapo_passk[64] - base_passk[64]:+.2f}pp  "
           f"({base_passk[64]:.2f}% → {dapo_passk[64]:.2f}%)\n"
           f"Δmaj@64  : {dapo_majk[64] - base_majk[64]:+.2f}pp  "
           f"({base_majk[64]:.2f}% → {dapo_majk[64]:.2f}%)")
    ax.text(1.05, 92, txt, fontsize=8.5, family="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9))

    plt.tight_layout()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"saved: {OUT_FILE}")

    print("\n=== summary ===")
    print(f"  base IT     pass@1={base_passk[1]:5.2f}%  pass@64={base_passk[64]:5.2f}%  maj@64={base_majk[64]:5.2f}%")
    print(f"  DAPO ck-15  pass@1={dapo_passk[1]:5.2f}%  pass@64={dapo_passk[64]:5.2f}%  maj@64={dapo_majk[64]:5.2f}%")
    print(f"  Δpass@1   = {dapo_passk[1] - base_passk[1]:+.2f}pp")
    print(f"  Δpass@64  = {dapo_passk[64] - base_passk[64]:+.2f}pp")
    print(f"  Δmaj@64   = {dapo_majk[64] - base_majk[64]:+.2f}pp")


if __name__ == "__main__":
    main()

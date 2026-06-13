"""MATH K=64 OOD preview — base GSM8K vs base MATH (DAPO MATH 未完成).

跑完 DAPO MATH 后用 _plot_dapo_ck15_combined.py 更新.
"""
import json
import os
from pathlib import Path

import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
import numpy as np

for p in ["/mnt/c/Windows/Fonts/msyh.ttc"]:
    if os.path.exists(p):
        _fm.fontManager.addfont(p)
        break

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

ROOT = Path("/mnt/d/fine-tuning")
GSM8K_BASE = ROOT / "v3/E1_baseline/outputs/pass_at_k_20260427_222954/base_gemma-2-2b-it_k64.json"
GSM8K_DAPO = ROOT / "v3/E5_grpo/outputs/k64_dapo_ck15/r15_dapo_checkpoint-15_k64.json"
MATH_BASE  = ROOT / "v3/E1_baseline/outputs/pass_at_k_math_20260513_092839/base_gemma-2-2b-it_k64.json"
OUT_DIR    = ROOT / "v3/E5_grpo/outputs"
OUT        = OUT_DIR / "math_k64_preview.png"


def load_passk(path):
    d = json.load(open(path))
    m = d["metrics"]
    ks = [1, 2, 4, 8, 16, 32, 64]
    return {
        "pass": [m[f"pass_at_{k}_numeric"] * 100 for k in ks],
        "maj":  m["maj_at_64_numeric"] * 100,
        "ks":   ks,
        "n":    d["config"]["samples"],
    }


def main():
    gsm_b = load_passk(GSM8K_BASE)
    gsm_d = load_passk(GSM8K_DAPO)
    math_b = load_passk(MATH_BASE)
    # MATH DAPO 还在跑

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # L1: pass@k curves
    ax = axes[0]
    ax.plot(gsm_b["ks"], gsm_b["pass"], "o-", color="#3b82f6", linewidth=2, markersize=6,
            label=f"GSM8K base (n={gsm_b['n']})")
    ax.plot(gsm_d["ks"], gsm_d["pass"], "s-", color="#dc2626", linewidth=2, markersize=6,
            label=f"GSM8K DAPO ck-15 (n={gsm_d['n']})")
    ax.plot(math_b["ks"], math_b["pass"], "o--", color="#3b82f6", linewidth=2, markersize=6,
            alpha=0.55, label=f"MATH base (n={math_b['n']}, OOD)")
    # placeholder for DAPO MATH
    ax.plot([], [], "s--", color="#dc2626", linewidth=2, markersize=6, alpha=0.55,
            label="MATH DAPO ck-15 (TBD, running)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(gsm_b["ks"])
    ax.set_xticklabels([str(k) for k in gsm_b["ks"]])
    ax.set_xlabel("K (samples per question)")
    ax.set_ylabel("pass@k (numeric, %)")
    ax.set_ylim(20, 95)
    ax.set_title("pass@k 曲线 — GSM8K (in-domain) vs MATH numeric (OOD)\n"
                 "实线 = GSM8K, 虚线 = MATH (n=2927)",
                 loc="left", fontsize=10, fontweight="semibold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3, linestyle=":")

    # L2: pass@1 + pass@64 + maj@64 bar
    ax = axes[1]
    metrics = ["pass@1", "pass@64", "maj@64"]
    gsm_b_vals = [gsm_b["pass"][0], gsm_b["pass"][-1], gsm_b["maj"]]
    gsm_d_vals = [gsm_d["pass"][0], gsm_d["pass"][-1], gsm_d["maj"]]
    math_b_vals = [math_b["pass"][0], math_b["pass"][-1], math_b["maj"]]
    x = np.arange(len(metrics))
    w = 0.22
    ax.bar(x - 1.5*w, gsm_b_vals, w, label="GSM8K base", color="#3b82f6")
    ax.bar(x - 0.5*w, gsm_d_vals, w, label="GSM8K DAPO", color="#dc2626")
    ax.bar(x + 0.5*w, math_b_vals, w, label="MATH base", color="#3b82f6", alpha=0.55, edgecolor="black", hatch="//")
    ax.bar(x + 1.5*w, [0]*3, w, label="MATH DAPO (TBD)", color="#dc2626", alpha=0.55, edgecolor="black", hatch="//")
    for i, (b, d, m) in enumerate(zip(gsm_b_vals, gsm_d_vals, math_b_vals)):
        ax.text(i - 1.5*w, b + 1, f"{b:.1f}", ha="center", fontsize=8)
        ax.text(i - 0.5*w, d + 1, f"{d:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.5*w, m + 1, f"{m:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("metric (%)")
    ax.set_ylim(0, 100)
    ax.set_title("base vs DAPO 在两个 domain\n"
                 "MATH DAPO 完成后会填灰红 bar (等 ~3h)",
                 loc="left", fontsize=10, fontweight="semibold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    # Headline annotation
    base_drop_p1 = gsm_b["pass"][0] - math_b["pass"][0]
    base_drop_p64 = gsm_b["pass"][-1] - math_b["pass"][-1]
    fig.suptitle(
        f"MATH numeric (n=2927) K=64 OOD eval — base IT 起点比 GSM8K 低很多\n"
        f"pass@1: GSM8K {gsm_b['pass'][0]:.1f}% → MATH {math_b['pass'][0]:.1f}% (Δ -{base_drop_p1:.1f}pp)  ·  "
        f"pass@64: {gsm_b['pass'][-1]:.1f}% → {math_b['pass'][-1]:.1f}% (Δ -{base_drop_p64:.1f}pp)",
        fontsize=11, fontweight="semibold", y=1.02
    )

    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved: {OUT}")

    # Print summary
    print()
    print("=== base 跨 domain 对比 ===")
    print(f"  pass@1   : GSM8K {gsm_b['pass'][0]:>5.2f}%  →  MATH {math_b['pass'][0]:>5.2f}%  Δ = -{base_drop_p1:.2f}pp")
    print(f"  pass@2   : GSM8K {gsm_b['pass'][1]:>5.2f}%  →  MATH {math_b['pass'][1]:>5.2f}%")
    print(f"  pass@8   : GSM8K {gsm_b['pass'][3]:>5.2f}%  →  MATH {math_b['pass'][3]:>5.2f}%")
    print(f"  pass@32  : GSM8K {gsm_b['pass'][5]:>5.2f}%  →  MATH {math_b['pass'][5]:>5.2f}%")
    print(f"  pass@64  : GSM8K {gsm_b['pass'][-1]:>5.2f}%  →  MATH {math_b['pass'][-1]:>5.2f}%  Δ = -{base_drop_p64:.2f}pp")
    print(f"  maj@64   : GSM8K {gsm_b['maj']:>5.2f}%  →  MATH {math_b['maj']:>5.2f}%")


if __name__ == "__main__":
    main()

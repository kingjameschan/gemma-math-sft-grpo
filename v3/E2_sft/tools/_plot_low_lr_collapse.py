"""lr=1e-5 + 5e-5 D_test pass@1 trajectory plot.

Single panel:
  X = train step (0 to 186, with step=0 = base IT anchor)
  Y = pass@1 (%)
  2 lines (lr=1e-5, lr=5e-5)
  + base IT horizontal reference at 61.94%
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "v3" / "E2_sft" / "outputs" / "sft_full_test_pass1"
OUT_FILE = ROOT / "v3" / "E2_sft" / "outputs" / "low_lr_collapse.png"
BASE = 61.94

LRS = ["1e-5", "5e-5"]
LR_COLORS = {"1e-5": "#1f77b4", "5e-5": "#d62728"}


def parse(jf):
    parts = jf.stem.split("_")
    return parts[1][2:], int(parts[3].split("-")[1])


def main():
    data = defaultdict(dict)  # data[lr][step] = pass@1
    for jf in EVAL_DIR.glob("sft_lr*_checkpoint-*.json"):
        try:
            lr, step = parse(jf)
        except Exception:
            continue
        if lr not in LRS:
            continue
        d = json.load(open(jf))
        data[lr][step] = d["pass_at_1"] * 100

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for lr in LRS:
        row = data.get(lr, {})
        if not row:
            continue
        steps = sorted(row.keys())
        vals = [row[s] for s in steps]
        # Prepend base IT as step=0
        steps = [0] + steps
        vals = [BASE] + vals
        ax.plot(steps, vals, "o-", color=LR_COLORS[lr],
                markersize=5, linewidth=1.6, alpha=0.95, label=f"lr={lr}")

    ax.axhline(BASE, color="#888", linestyle=":", linewidth=1.2, alpha=0.7,
               label=f"base IT = {BASE:.1f}%")

    ax.set_xlabel("checkpoint step (~16 samples per step, eff_batch=16)")
    ax.set_ylabel("pass@1 (%) on D_test (1319)")
    ax.set_title("Low-LR SFT collapse trajectory — lr=1e-5 / 5e-5 also crash, just slower",
                 loc="left", fontsize=12, fontweight="semibold")
    ax.set_xlim(left=0)
    all_steps = sorted({0, 10, 30, 50, 70, 90, 110, 130, 150, 170, 186})
    ax.set_xticks(all_steps)
    ax.set_xticklabels([str(s) for s in all_steps], rotation=45, fontsize=9)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="best", fontsize=10)
    ax.set_ylim(35, 70)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"saved: {OUT_FILE}")


if __name__ == "__main__":
    main()

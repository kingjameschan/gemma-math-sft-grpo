"""Stage 1 v2 D_dev trajectory plot — 4×3 grid (LR × β), 5 ckpts each.

Updates as each config completes its eval. Empty panels show "pending".
Run after every new config eval.

Output: v3/E5_grpo/outputs/fastgrid/stage1_v2_trajectories.png
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "v3" / "E5_grpo" / "outputs" / "fastgrid" / "stage1_v2_eval"
OUT = ROOT / "v3" / "E5_grpo" / "outputs" / "fastgrid" / "stage1_v2_trajectories.png"

LRS = ["1e-6", "5e-6", "1e-5", "5e-5"]
BETAS = ["0.01", "0.04", "0.1"]
STEPS = [20, 40, 60, 80, 100]

# Reference baselines (approximated from earlier base IT runs)
# D_dev is systematically ~9pp higher than test (per E5 dev-vs-test analysis)
BASE_DEV_PASS1 = 70    # base IT estimated D_dev pass@1
BASE_DEV_BOXED = 46    # base IT estimated boxed_rate (E1 reports test boxed=46.17%, similar)


def load_results():
    out = {}
    for f in sorted(EVAL_DIR.glob("*.json")):
        m = re.match(r"lr(.+?)_b(.+?)_checkpoint-(\d+)", f.stem)
        if not m: continue
        lr, beta, step = m.group(1), m.group(2), int(m.group(3))
        d = json.loads(f.read_text())
        out[(lr, beta, step)] = {
            "pass1": d["pass_at_1"] * 100,
            "boxed": d["boxed_rate"] * 100,
            "len":   d["mean_response_length"],
        }
    return out


def main():
    results = load_results()
    print(f"[load] {len(results)} ckpt evals from {EVAL_DIR.name}")

    # Determine which configs are fully done (all 5 ckpts)
    done_configs = []
    for lr in LRS:
        for beta in BETAS:
            n = sum(1 for s in STEPS if (lr, beta, s) in results)
            if n == len(STEPS):
                done_configs.append((lr, beta))
    print(f"[plot] {len(done_configs)}/12 configs fully evaluated")

    fig, axes = plt.subplots(len(LRS), len(BETAS), figsize=(11, 11),
                             sharex=True, sharey=True)
    for i, lr in enumerate(LRS):
        for j, beta in enumerate(BETAS):
            ax = axes[i, j]
            ps, bs = [], []
            for s in STEPS:
                d = results.get((lr, beta, s))
                ps.append(d["pass1"] if d else np.nan)
                bs.append(d["boxed"] if d else np.nan)
            n_done = sum(1 for p in ps if not np.isnan(p))

            if n_done > 0:
                ax.plot(STEPS, ps, "o-", color="C0", lw=2, ms=8, label="pass@1")
                ax.plot(STEPS, bs, "s--", color="C1", lw=2, ms=8, label="boxed_rate")
                ax.axhline(BASE_DEV_PASS1, color="C0", ls=":", alpha=0.5)
                ax.axhline(BASE_DEV_BOXED, color="C1", ls=":", alpha=0.5)
                for x, y in zip(STEPS, ps):
                    if not np.isnan(y):
                        ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 6),
                                    textcoords="offset points", ha="center",
                                    fontsize=7, color="C0")
                for x, y in zip(STEPS, bs):
                    if not np.isnan(y):
                        ax.annotate(f"{y:.0f}", (x, y), xytext=(0, -12),
                                    textcoords="offset points", ha="center",
                                    fontsize=7, color="C1")
                ax.set_title(f"lr={lr}  β={beta}  ({n_done}/{len(STEPS)})", fontsize=10)
            else:
                ax.text(0.5, 0.5, "pending", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=12)
                ax.set_title(f"lr={lr}  β={beta}", fontsize=10, color="gray")

            ax.set_ylim(20, 100)
            ax.grid(alpha=0.3)
            if i == len(LRS) - 1: ax.set_xlabel("step")
            if j == 0: ax.set_ylabel("%")
            if i == 0 and j == 0 and n_done > 0:
                ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        f"Stage 1 v2 D_dev trajectories: pass@1 (blue) + boxed_rate (orange) "
        f"— {len(done_configs)}/12 configs done\n"
        f"G=16  T=1.0  max_steps=100  save_steps=20  "
        f"(dashed gray = base IT ≈{BASE_DEV_PASS1}%/{BASE_DEV_BOXED}%)",
        fontsize=11, y=0.995,
    )
    plt.tight_layout()
    plt.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()

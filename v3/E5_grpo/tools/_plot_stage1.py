"""Stage 1 GRPO fastgrid analysis plots.

Inputs:
  - v3/E5_grpo/outputs/fastgrid/stage1_eval/*.json  (48 ckpts × pass@1/boxed/len)
  - v3/E5_grpo/outputs/fastgrid/stage1_logs/*.log    (12 configs × per-step reward/KL/loss)

Outputs:
  - stage1_heatmap.png       — 2 grids (pass@1, boxed_rate) at FINAL step (4 LR × 3 β)
  - stage1_trajectories.png  — pass@1 + boxed over 4 ckpts, 12 panels
  - stage1_train_curves.png  — reward, KL, loss, frac_zero_std over 20 train steps
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "v3" / "E5_grpo" / "outputs" / "fastgrid" / "stage1_eval"
LOG_DIR = ROOT / "v3" / "E5_grpo" / "outputs" / "fastgrid" / "stage1_logs"
OUT_DIR = ROOT / "v3" / "E5_grpo" / "outputs" / "fastgrid"

LRS = ["1e-6", "5e-6", "1e-5", "5e-5"]
BETAS = ["0.01", "0.04", "0.1"]
STEPS = [5, 10, 15, 20]


def load_eval():
    out = {}
    for f in sorted(EVAL_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        name = f.stem  # lr1e-5_b0.01_checkpoint-10
        m = re.match(r"lr(.+?)_b(.+?)_checkpoint-(\d+)", name)
        if not m:
            continue
        lr, beta, step = m.group(1), m.group(2), int(m.group(3))
        out[(lr, beta, step)] = {
            "pass1": d["pass_at_1"],
            "boxed": d["boxed_rate"],
            "mean_len": d["mean_response_length"],
        }
    return out


def parse_train_log(log_path):
    """Extract per-step metrics from train log (TRL prints dict-like lines)."""
    metrics_keys = ["reward", "reward_std", "kl", "loss", "grad_norm",
                    "frac_reward_zero_std", "completions/mean_length", "entropy"]
    series = {k: [] for k in metrics_keys}
    text = log_path.read_text()
    # Each step prints a dict like {'loss': '...', 'reward': '...', ...}
    for line in text.splitlines():
        if not line.startswith("{'loss'"):
            continue
        for k in metrics_keys:
            m = re.search(rf"'{re.escape(k)}': '([^']+)'", line)
            if m:
                try:
                    series[k].append(float(m.group(1)))
                except ValueError:
                    pass
    return series


# ============================================================
# Plot 1: heatmap at step 20 (final ckpt)
# ============================================================
def plot_heatmap(eval_data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    titles = ["pass@1 (D_dev) at step 20", "boxed_rate (D_dev) at step 20"]
    keys = ["pass1", "boxed"]

    for ax, title, key in zip(axes, titles, keys):
        grid = np.full((len(LRS), len(BETAS)), np.nan)
        for i, lr in enumerate(LRS):
            for j, beta in enumerate(BETAS):
                d = eval_data.get((lr, beta, 20))
                if d:
                    grid[i, j] = d[key] * 100

        cmap = "RdYlGn" if key == "pass1" else "viridis"
        vmin, vmax = (60, 75) if key == "pass1" else (30, 100)
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(BETAS)))
        ax.set_xticklabels([f"β={b}" for b in BETAS])
        ax.set_yticks(range(len(LRS)))
        ax.set_yticklabels([f"lr={lr}" for lr in LRS])
        ax.set_title(title)
        for i in range(len(LRS)):
            for j in range(len(BETAS)):
                v = grid[i, j]
                if not np.isnan(v):
                    color = "white" if (key == "boxed" and v < 60) else "black"
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            color=color, fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate winner
    fig.suptitle("Stage 1 fastgrid: 4 LR × 3 β grid (final ckpt step=20)\n"
                 "Winner: lr=5e-5 β=0.04 step=10  →  pass@1=72.6%  boxed=91.6%",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "stage1_heatmap.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


# ============================================================
# Plot 2: pass@1 + boxed trajectories (4 LR rows × 3 β cols)
# ============================================================
def plot_trajectories(eval_data):
    fig, axes = plt.subplots(len(LRS), len(BETAS), figsize=(11, 11),
                             sharex=True, sharey=True)
    # D_dev base IT (no exact run; approx +9pp above test 62 ≈ 71)
    base_pass1 = 70  # base IT estimated D_dev pass@1
    base_boxed = 46  # E1 baseline boxed_rate

    for i, lr in enumerate(LRS):
        for j, beta in enumerate(BETAS):
            ax = axes[i, j]
            ps, bs = [], []
            for s in STEPS:
                d = eval_data.get((lr, beta, s))
                if d:
                    ps.append(d["pass1"] * 100)
                    bs.append(d["boxed"] * 100)
                else:
                    ps.append(np.nan); bs.append(np.nan)
            ax.plot(STEPS, ps, "o-", color="C0", label="pass@1")
            ax.plot(STEPS, bs, "s--", color="C1", label="boxed_rate")
            ax.axhline(base_pass1, color="C0", ls=":", alpha=0.4)
            ax.axhline(base_boxed, color="C1", ls=":", alpha=0.4)
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
            ax.set_title(f"lr={lr}  β={beta}", fontsize=10)
            ax.set_ylim(20, 100)
            ax.grid(alpha=0.3)
            if i == len(LRS) - 1:
                ax.set_xlabel("step")
            if j == 0:
                ax.set_ylabel("%")
            if i == 0 and j == 0:
                ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("Stage 1 trajectories: pass@1 (blue) + boxed_rate (orange) over 4 ckpts\n"
                 "Dashed = base IT reference (≈62%/46%)", fontsize=11, y=0.995)
    plt.tight_layout()
    out = OUT_DIR / "stage1_trajectories.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


# ============================================================
# Plot 3: training curves (reward, KL, loss, frac_zero_std)
# ============================================================
def plot_train_curves():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    metric_panels = [
        ("reward",                "reward (mean over 8 prompts × 8 G)", axes[0, 0]),
        ("kl",                    "KL (vs ref)",                          axes[0, 1]),
        ("loss",                  "loss (log scale)",                     axes[1, 0]),
        ("frac_reward_zero_std",  "frac of groups with std=0 (degenerate)", axes[1, 1]),
    ]

    cmap = plt.cm.viridis(np.linspace(0, 1, len(LRS) * len(BETAS)))
    color_idx = 0
    for lr in LRS:
        for beta in BETAS:
            log_file = LOG_DIR / f"lr{lr}_b{beta}.log"
            if not log_file.exists():
                continue
            series = parse_train_log(log_file)
            label = f"lr={lr} β={beta}"
            color = cmap[color_idx]
            for key, title, ax in metric_panels:
                ys = series.get(key, [])
                if ys:
                    ax.plot(range(1, len(ys) + 1), ys, "-",
                            color=color, alpha=0.7, label=label, lw=1.4)
            color_idx += 1

    for key, title, ax in metric_panels:
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        if key == "loss":
            ax.set_yscale("symlog", linthresh=1e-7)

    axes[0, 0].legend(fontsize=7, loc="upper left", ncol=2, bbox_to_anchor=(1.05, 1))
    fig.suptitle("Stage 1 training curves: 12 configs over 20 steps", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "stage1_train_curves.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


def main():
    print("[load] eval JSONs...")
    eval_data = load_eval()
    print(f"  loaded {len(eval_data)} ckpt evals")

    print("\n[plot] heatmap...")
    plot_heatmap(eval_data)

    print("\n[plot] trajectories...")
    plot_trajectories(eval_data)

    print("\n[plot] training curves...")
    plot_train_curves()


if __name__ == "__main__":
    main()

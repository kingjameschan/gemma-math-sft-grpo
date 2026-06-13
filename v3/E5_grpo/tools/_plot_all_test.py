"""All Stage 1 + Pilot results visualized on TEST set (1319).

Outputs:
  - stage1_heatmap_test.png      — 4×3 LR×β heatmap (test pass@1 final ckpt)
  - stage1_trajectories_test.png — 12 panels × test pass@1 over step 5/10/15/20
  - stage1_vs_dev_test.png       — D_dev vs test scatter for all 48 ckpts
  - pilot_g8_vs_g16_test.png     — G=8 vs G=16 trajectories (test 1319)
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
S1_DEV  = ROOT / "v3" / "E5_grpo" / "outputs" / "fastgrid" / "stage1_eval"
S1_TEST = ROOT / "v3" / "E5_grpo" / "outputs" / "fastgrid" / "stage1_eval_test"
PILOT_TEST = ROOT / "v3" / "E5_grpo" / "outputs" / "pilot" / "eval_test_g16"
OUT = ROOT / "v3" / "E5_grpo" / "outputs" / "fastgrid"

LRS = ["1e-6", "5e-6", "1e-5", "5e-5"]
BETAS = ["0.01", "0.04", "0.1"]
STEPS = [5, 10, 15, 20]
BASE_TEST_PASS1 = 61.94  # E1 baseline


def load_grid(d):
    out = {}
    for f in sorted(Path(d).glob("*.json")):
        m = re.match(r"lr(.+?)_b(.+?)_checkpoint-(\d+)", f.stem)
        if not m: continue
        lr, beta, step = m.group(1), m.group(2), int(m.group(3))
        d_ = json.loads(f.read_text())
        out[(lr, beta, step)] = {
            "pass1": d_["pass_at_1"] * 100,
            "boxed": d_["boxed_rate"] * 100,
            "len":   d_["mean_response_length"],
        }
    return out


def plot_heatmap_test(s1_test):
    fig, ax = plt.subplots(figsize=(8, 5))
    grid = np.full((len(LRS), len(BETAS)), np.nan)
    for i, lr in enumerate(LRS):
        for j, beta in enumerate(BETAS):
            d = s1_test.get((lr, beta, 20))
            if d:
                grid[i, j] = d["pass1"]

    im = ax.imshow(grid, cmap="RdYlGn", vmin=58, vmax=66, aspect="auto")
    ax.set_xticks(range(len(BETAS))); ax.set_xticklabels([f"β={b}" for b in BETAS])
    ax.set_yticks(range(len(LRS)));   ax.set_yticklabels([f"lr={lr}" for lr in LRS])
    for i in range(len(LRS)):
        for j in range(len(BETAS)):
            v = grid[i, j]
            if not np.isnan(v):
                delta = v - BASE_TEST_PASS1
                color = "white" if v < 61 else "black"
                ax.text(j, i, f"{v:.2f}\nΔ{delta:+.2f}", ha="center", va="center",
                        color=color, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, label="pass@1 (test, %)")
    ax.set_title(f"Stage 1: TEST pass@1 at step=20 (vs base IT={BASE_TEST_PASS1}%)\n"
                 f"All 12 configs in 60.5-62.2%, Δ within ±1.4pp (noise floor)")
    plt.tight_layout()
    out = OUT / "stage1_heatmap_test.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


def plot_trajectories_test(s1_test):
    fig, axes = plt.subplots(len(LRS), len(BETAS), figsize=(11, 11),
                             sharex=True, sharey=True)
    for i, lr in enumerate(LRS):
        for j, beta in enumerate(BETAS):
            ax = axes[i, j]
            ps = []
            for s in STEPS:
                d = s1_test.get((lr, beta, s))
                ps.append(d["pass1"] if d else np.nan)
            ax.plot(STEPS, ps, "o-", color="C0", lw=2, ms=8)
            ax.axhline(BASE_TEST_PASS1, color="gray", ls="--", alpha=0.6,
                       label=f"base={BASE_TEST_PASS1}%")
            for x, y in zip(STEPS, ps):
                if not np.isnan(y):
                    ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 6),
                                textcoords="offset points", ha="center", fontsize=8)
            ax.set_title(f"lr={lr}  β={beta}", fontsize=10)
            ax.set_ylim(58, 66)
            ax.grid(alpha=0.3)
            if i == len(LRS) - 1: ax.set_xlabel("step")
            if j == 0: ax.set_ylabel("test pass@1 (%)")
            if i == 0 and j == 0: ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Stage 1 TEST pass@1 trajectories (12 configs × 4 ckpts)\n"
                 f"Dashed = base IT {BASE_TEST_PASS1}%", fontsize=12, y=0.995)
    plt.tight_layout()
    out = OUT / "stage1_trajectories_test.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


def plot_dev_vs_test(s1_dev, s1_test):
    fig, ax = plt.subplots(figsize=(8, 7))
    devs, tests = [], []
    for k, td in s1_test.items():
        dd = s1_dev.get(k)
        if dd:
            devs.append(dd["pass1"]); tests.append(td["pass1"])

    ax.scatter(devs, tests, s=40, alpha=0.7, color="C0")
    # 1:1 line
    lo, hi = 58, 75
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="1:1")
    # Best fit
    z = np.polyfit(devs, tests, 1)
    xs = np.array([lo, hi])
    ax.plot(xs, np.polyval(z, xs), "r-", alpha=0.7,
            label=f"fit: test = {z[0]:.2f}·dev + {z[1]:.1f}")
    # Reference lines
    ax.axhline(BASE_TEST_PASS1, color="gray", ls=":", alpha=0.5,
               label=f"base IT test={BASE_TEST_PASS1}%")
    # Mean offset
    offsets = [d - t for d, t in zip(devs, tests)]
    mean_off = np.mean(offsets)
    ax.set_title(f"Stage 1: D_dev vs TEST pass@1 (48 ckpts)\n"
                 f"D_dev mean = {np.mean(devs):.2f}%, "
                 f"Test mean = {np.mean(tests):.2f}% — D_dev systematically +{mean_off:.2f}pp higher")
    ax.set_xlabel("D_dev pass@1 (%)")
    ax.set_ylabel("TEST pass@1 (%)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT / "stage1_dev_vs_test.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


def plot_pilot_g8_vs_g16(pilot_test):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: pass@1 trajectory G=8 vs G=16
    ax = axes[0]
    pilot_steps = [30, 60, 90, 120, 150]
    g8_pass1, g16_pass1 = [], []
    g8_boxed, g16_boxed = [], []
    for s in pilot_steps:
        g8  = json.loads((PILOT_TEST / f"lr5e-5_b0.04_G8_s150_checkpoint-{s}.json").read_text())
        g16 = json.loads((PILOT_TEST / f"lr5e-5_b0.04_G16_s150_checkpoint-{s}.json").read_text())
        g8_pass1.append(g8["pass_at_1"]*100); g16_pass1.append(g16["pass_at_1"]*100)
        g8_boxed.append(g8["boxed_rate"]*100); g16_boxed.append(g16["boxed_rate"]*100)

    ax.plot(pilot_steps, g8_pass1, "o-", color="C0", lw=2, ms=8, label="G=8")
    ax.plot(pilot_steps, g16_pass1, "s-", color="C1", lw=2, ms=8, label="G=16")
    ax.axhline(BASE_TEST_PASS1, color="gray", ls="--", alpha=0.6,
               label=f"base={BASE_TEST_PASS1}%")
    for x, y in zip(pilot_steps, g8_pass1):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8, color="C0")
    for x, y in zip(pilot_steps, g16_pass1):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=8, color="C1")
    ax.set_title("Pilot: TEST pass@1 (lr=5e-5 β=0.04 × 150 step)")
    ax.set_xlabel("step"); ax.set_ylabel("TEST pass@1 (%)")
    ax.set_ylim(58, 66)
    ax.legend(); ax.grid(alpha=0.3)

    # Panel B: boxed_rate trajectory
    ax = axes[1]
    ax.plot(pilot_steps, g8_boxed, "o-", color="C0", lw=2, ms=8, label="G=8")
    ax.plot(pilot_steps, g16_boxed, "s-", color="C1", lw=2, ms=8, label="G=16")
    ax.axhline(46, color="gray", ls="--", alpha=0.6, label="base boxed≈46%")
    ax.set_title("Pilot: TEST boxed_rate (format learning is unstable)")
    ax.set_xlabel("step"); ax.set_ylabel("boxed_rate (%)")
    ax.set_ylim(0, 100)
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("Pilot G=8 vs G=16 — both ±3pp pass@1 noise, no clear winner",
                 fontsize=12)
    plt.tight_layout()
    out = OUT / "pilot_g8_vs_g16_test.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


def main():
    print("[load] s1 dev + test ...")
    s1_dev  = load_grid(S1_DEV)
    s1_test = load_grid(S1_TEST)
    print(f"  s1_dev  {len(s1_dev)} ckpts")
    print(f"  s1_test {len(s1_test)} ckpts")
    pilot_test = load_grid(PILOT_TEST)
    print(f"  pilot_test {len(pilot_test)} ckpts")

    print("\n[plot] heatmap test ..."); plot_heatmap_test(s1_test)
    print("[plot] trajectories test ..."); plot_trajectories_test(s1_test)
    print("[plot] dev vs test scatter ..."); plot_dev_vs_test(s1_dev, s1_test)
    print("[plot] pilot G=8 vs G=16 ..."); plot_pilot_g8_vs_g16(pilot_test)


if __name__ == "__main__":
    main()

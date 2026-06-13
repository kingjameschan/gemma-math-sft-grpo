"""Fastgrid D_test heatmap: LR × rank × ckpt step.

3 panels (step=3, 6, 9), each panel is an 8×4 heatmap (LR × rank).
Color = pass@1 (%), cells annotated with numeric values.
Diverging colormap centered on base IT pass@1.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "v3" / "E2_sft" / "outputs" / "fastgrid_eval_test"
OUT_FILE = ROOT / "v3" / "E2_sft" / "outputs" / "fastgrid_heatmap.png"
BASE = 61.94   # base IT D_test pass@1

LR_ORDER = ["1e-5", "5e-5", "1e-4", "2.5e-4", "5e-4", "7.5e-4", "1e-3", "2.5e-3"]
RANKS = [8, 16, 32, 64]
STEPS = [3, 6, 9]


def parse(jf):
    parts = jf.stem.split("_")
    return parts[1][2:], int(parts[2][1:]), int(parts[3].split("-")[1])


def main():
    data = defaultdict(dict)  # data[step][(rank, lr)] = pass@1
    for jf in EVAL_DIR.glob("sft_lr*_checkpoint-*.json"):
        try:
            lr, r, step = parse(jf)
        except Exception:
            continue
        d = json.load(open(jf))
        data[step][(r, lr)] = d["pass_at_1"] * 100

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
    })
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    # Color limits: span from 0 to (base + max headroom). Center diverging at base.
    vmin, vmax = 0, 75
    cmap = plt.cm.RdYlGn
    norm = plt.cm.colors.TwoSlopeNorm(vmin=vmin, vcenter=BASE, vmax=vmax)

    im = None
    for ax, step in zip(axes, STEPS):
        # Build matrix: rows = ranks (low to high), cols = LRs (low to high)
        mat = np.full((len(RANKS), len(LR_ORDER)), np.nan)
        for i, r in enumerate(RANKS):
            for j, lr in enumerate(LR_ORDER):
                v = data[step].get((r, lr))
                if v is not None:
                    mat[i, j] = v

        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

        # Cell annotations
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                # Pick text color based on cell brightness
                color = "black" if abs(v - BASE) < 15 else "white"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="medium")

        ax.set_xticks(range(len(LR_ORDER)))
        ax.set_xticklabels(LR_ORDER, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(RANKS)))
        ax.set_yticklabels([f"r={r}" for r in RANKS])
        ax.set_xlabel("learning rate")
        if step == STEPS[0]:
            ax.set_ylabel("LoRA rank")
        ax.set_title(f"checkpoint step = {step}", loc="left", fontsize=11, fontweight="semibold")

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.025,
                        pad=0.02, ticks=[0, 20, 40, BASE, 70])
    cbar.set_label("pass@1 (%)")
    cbar.ax.axhline(BASE, color="black", linewidth=1.0)
    cbar.ax.set_yticklabels([f"{int(t)}" if t != BASE else f"base={BASE}" for t in [0, 20, 40, BASE, 70]])

    fig.suptitle(f"Fastgrid D_test (N=1319) — pass@1 heatmap × 3 ckpt steps  (base IT = {BASE}%)",
                 fontsize=12, fontweight="semibold", y=1.02)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"saved: {OUT_FILE}")


if __name__ == "__main__":
    main()

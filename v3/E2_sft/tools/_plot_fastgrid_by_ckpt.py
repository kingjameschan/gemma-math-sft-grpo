"""Fastgrid plot with ckpt as primary dimension.

Layout: 3 panels (step=3, 6, 9), each panel:
  X = LR (log scale, 8 values)
  Lines = rank (4 values: 8/16/32/64)
  Y = pass@1 (%)

Reads BOTH D_dev and D_test fastgrid_eval dirs and generates two output files.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]

CONFIGS = [
    {
        "name": "D_dev",
        "eval_dir": ROOT / "v3" / "E2_sft" / "outputs" / "fastgrid_eval",
        "base_path": ROOT / "v3" / "E2_sft" / "outputs" / "dev_eval" / "base_gemma-2-2b-it.json",
        "base_key": "pass_at_1",   # already in 0..1
        "out_file": ROOT / "v3" / "E2_sft" / "outputs" / "fastgrid_by_ckpt_dev.png",
        "n": 500,
    },
    {
        "name": "D_test",
        "eval_dir": ROOT / "v3" / "E2_sft" / "outputs" / "fastgrid_eval_test",
        "base_anchor": 61.94,      # from E1 baseline summary (numeric_accuracy)
        "out_file": ROOT / "v3" / "E2_sft" / "outputs" / "fastgrid_by_ckpt_test.png",
        "n": 1319,
    },
]

LR_ORDER = ["1e-5", "5e-5", "1e-4", "2.5e-4", "5e-4", "7.5e-4", "1e-3", "2.5e-3"]
LR_NUMERIC = {"1e-5": 1e-5, "5e-5": 5e-5, "1e-4": 1e-4, "2.5e-4": 2.5e-4,
              "5e-4": 5e-4, "7.5e-4": 7.5e-4, "1e-3": 1e-3, "2.5e-3": 2.5e-3}
RANKS = [8, 16, 32, 64]
STEPS = [3, 6, 9]
# 4 high-contrast colors (Wong palette, colorblind-safe, well-separated)
RANK_COLORS = {8: "#000000", 16: "#E69F00", 32: "#56B4E9", 64: "#009E73"}
#                black       orange       sky blue     teal-green


def parse(jf):
    parts = jf.stem.split("_")
    return parts[1][2:], int(parts[2][1:]), int(parts[3].split("-")[1])


def render(cfg):
    eval_dir = cfg["eval_dir"]
    if not eval_dir.exists():
        print(f"[skip] {cfg['name']}: {eval_dir} missing")
        return

    # Resolve base anchor
    if "base_anchor" in cfg:
        base = cfg["base_anchor"]
    else:
        d = json.load(open(cfg["base_path"]))
        base = d[cfg["base_key"]] * 100

    data = defaultdict(dict)  # data[(step, rank)][lr] = pass@1
    for jf in eval_dir.glob("*.json"):
        try:
            lr, r, step = parse(jf)
        except Exception:
            continue
        d = json.load(open(jf))
        data[(step, r)][lr] = d["pass_at_1"] * 100

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, step in zip(axes, STEPS):
        for r in RANKS:
            row = data.get((step, r), {})
            if not row:
                continue
            xs = [LR_NUMERIC[lr] for lr in LR_ORDER if lr in row]
            ys = [row[lr] for lr in LR_ORDER if lr in row]
            ax.plot(xs, ys, "-", color=RANK_COLORS[r], label=f"r={r}",
                    linewidth=1.0, alpha=0.95)
        # Base IT reference
        ax.axhline(base, color="#999", linestyle=":", linewidth=1.0, alpha=0.7,
                   label=f"base IT = {base:.1f}%")
        ax.set_xscale("log")
        ax.set_xticks([lr for lr in LR_NUMERIC.values()])
        ax.set_xticklabels(LR_ORDER, rotation=45, fontsize=8)
        ax.set_xlabel("learning rate")
        ax.set_title(f"checkpoint step = {step}", loc="left", fontsize=11)
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("pass@1 (%)")

    fig.suptitle(f"Fastgrid {cfg['name']} (N={cfg['n']}) — pass@1 vs LR, by ckpt step",
                 fontsize=12, fontweight="semibold")
    plt.tight_layout()

    cfg["out_file"].parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cfg["out_file"], dpi=180, bbox_inches="tight", facecolor="white")
    print(f"saved: {cfg['out_file']}")
    plt.close()


def main():
    for cfg in CONFIGS:
        render(cfg)


if __name__ == "__main__":
    main()

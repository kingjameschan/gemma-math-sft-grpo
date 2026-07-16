"""Regenerate the eight-panel same-chain PPL figure from compact results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = REPO_ROOT / "v3" / "E5_grpo" / "outputs" / "yue_ppl_analysis"

LABELS = [
    r"$PPL_{base}(Y_{base})$",
    r"$PPL_{base}(Y_{DAPO})$",
    r"$\mathbf{PPL_{DAPO}(Y_{DAPO})}$",
    r"$PPL_{base}(Y_{GRPO})$",
    r"$\mathbf{PPL_{GRPO}(Y_{GRPO})}$",
    r"$PPL_{base}(Y_{SFT})$",
    r"$PPL_{base}(Y_{Claude})$",
    r"$PPL_{base}(Y_{Gemini})$",
]
COLORS = [
    "#4a9b8e",
    "#c64646",
    "#a13030",
    "#8b5fbf",
    "#5d3f87",
    "#3d8b3d",
    "#cc7ab8",
    "#e2a857",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=DEFAULT_DIR / "ppl_8panel_selfppl_results.json"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_DIR / "yue_8panel_selfppl.png"
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.results.open(encoding="utf-8") as handle:
        panels = json.load(handle)

    panel_map = {
        (panel["problem_id"], panel["filter"]): panel for panel in panels
    }
    figure, axes = plt.subplots(4, 2, figsize=(13, 17), facecolor="white")
    figure.subplots_adjust(
        wspace=0.30, hspace=0.62, top=0.95, bottom=0.09, left=0.07, right=0.97
    )

    for row, problem_id in enumerate(range(1, 5)):
        for column, filter_label in enumerate(("correct", "wrong")):
            axis = axes[row, column]
            panel = panel_map[(problem_id, filter_label)]
            data = [bar["ppls"] for bar in panel["bars"]]
            tick_labels = [
                f"{label}\nN={len(values)}" for label, values in zip(LABELS, data)
            ]
            boxplot = axis.boxplot(
                data,
                tick_labels=tick_labels,
                patch_artist=True,
                widths=0.55,
                showfliers=True,
                flierprops={
                    "marker": "*",
                    "markersize": 2.5,
                    "markerfacecolor": "black",
                    "markeredgecolor": "black",
                },
                medianprops={"color": "black", "linewidth": 1.0},
                whiskerprops={"linewidth": 0.8},
                capprops={"linewidth": 0.8},
            )
            for patch, color in zip(boxplot["boxes"], COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.65)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.5)

            for index, values in enumerate(data, start=1):
                if not values:
                    continue
                jitter = np.random.RandomState(42).uniform(-0.10, 0.10, len(values))
                axis.scatter(
                    index + jitter,
                    values,
                    s=1.2,
                    color="black",
                    alpha=0.5,
                    zorder=10,
                )

            flattened = [value for values in data for value in values]
            if flattened:
                axis.set_ylim(
                    max(0.95, min(flattened) * 0.95), max(flattened) * 1.05
                )
            axis.set_ylabel("Perplexity")
            axis.tick_params(axis="x", labelrotation=45, labelsize=6)
            axis.grid(axis="y", alpha=0.3, linestyle=":")
            axis.set_title(
                f'P{problem_id} ({panel["base_label"]}, gold={panel["gold"]}) '
                f'— RL filter: {filter_label}',
                fontsize=10,
            )

    figure.text(
        0.5,
        0.015,
        "Bold bars are RL self-PPL; compare 2–3 (DAPO) and 4–5 (GRPO).",
        ha="center",
        fontsize=9,
        fontstyle="italic",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    pooled = {
        index: [
            value
            for panel in panels
            for value in panel["bars"][index]["ppls"]
        ]
        for index in range(len(LABELS))
    }
    print(f"Saved {args.output}")
    print(
        "Aggregate medians: "
        f"Base(DAPO)={median(pooled[1]):.3f}, DAPO(DAPO)={median(pooled[2]):.3f}, "
        f"Base(GRPO)={median(pooled[3]):.3f}, GRPO(GRPO)={median(pooled[4]):.3f}, "
        f"Base(Claude)={median(pooled[6]):.3f}, Base(Gemini)={median(pooled[7]):.3f}"
    )


if __name__ == "__main__":
    main()

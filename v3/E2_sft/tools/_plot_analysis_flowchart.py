"""Top-down analysis flowchart.

Starting observation (L1): pass@K stable, maj@K -5pp, pass@1 -18pp.
For each, traces which deeper layer answers WHY.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_FILE = Path(__file__).resolve().parents[3] / "v3" / "E2_sft" / "outputs" / "analysis_flowchart.png"


def box(ax, x, y, w, h, text, fc="white", ec="black", fs=9.5, fw="normal", lw=1.2):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          fc=fc, ec=ec, linewidth=lw)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=fw, wrap=True)


def arrow(ax, x1, y1, x2, y2, color="#444", lw=1.4, style="->"):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                          mutation_scale=14, color=color, linewidth=lw,
                          connectionstyle="arc3,rad=0")
    ax.add_patch(arr)


def main():
    fig, ax = plt.subplots(figsize=(18, 13))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # ============ Row 0: Title ============
    ax.text(9, 13.5, "SFT lr=5e-4 step=130 vs base IT — analysis decomposition",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(9, 13.0, "Starting question: why did pass@K / maj@K / pass@1 each change differently?",
            ha="center", fontsize=11, color="#444", fontstyle="italic")

    # ============ Row 1: L1 Observations (3 columns) ============
    obs_y = 11.3
    obs_h = 1.1
    obs_w = 5.0
    cols = [(0.5, "#fee2e2", "#dc2626"), (6.5, "#fef3c7", "#f59e0b"), (12.5, "#dcfce7", "#16a34a")]

    box(ax, cols[0][0], obs_y, obs_w, obs_h,
        "(L1) pass@K=64\nbase 93%  →  SFT 93%\nΔ ≈ 0",
        fc=cols[0][1], ec=cols[0][2], fs=11, fw="bold")
    box(ax, cols[1][0], obs_y, obs_w, obs_h,
        "(L1) maj@K=64\nbase 70%  →  SFT 65%\nΔ = -5pp",
        fc=cols[1][1], ec=cols[1][2], fs=11, fw="bold")
    box(ax, cols[2][0], obs_y, obs_w, obs_h,
        "(L1) pass@1\nbase 62%  →  SFT 44%\nΔ = -18pp",
        fc=cols[2][1], ec=cols[2][2], fs=11, fw="bold")

    # ============ Row 2: Semantic interpretation ============
    sem_y = 9.3
    sem_h = 1.4

    box(ax, cols[0][0], sem_y, obs_w, sem_h,
        "what it measures:\nDISTRIBUTION SUPPORT\n(can correct ever appear in 64 samples?)\n→ ceiling capability",
        fs=9, fc="white", ec=cols[0][2])
    box(ax, cols[1][0], sem_y, obs_w, sem_h,
        "what it measures:\nDISTRIBUTION MODE\n(is correct the most-frequent answer?)\n→ top-1 consensus",
        fs=9, fc="white", ec=cols[1][2])
    box(ax, cols[2][0], sem_y, obs_w, sem_h,
        "what it measures:\nDISTRIBUTION MASS\n(correct's frequency = c/64)\n→ sampling efficiency",
        fs=9, fc="white", ec=cols[2][2])

    # ============ Row 3: Open questions (4 boxes, some span) ============
    q_y = 7.3
    q_h = 1.4

    box(ax, cols[0][0], q_y, obs_w, q_h,
        "Q: was capability fully preserved\nor did some Q lose it entirely?\n→ check per-Q pass@K shape",
        fs=9, fc="white", ec=cols[0][2])
    box(ax, cols[1][0], q_y, obs_w, q_h,
        "Q1: which 5% of Q had mode flip?\n(distribution: bucket?)\nQ2: is mode wrong because of\nattractor or dilution? (shape?)",
        fs=9, fc="white", ec=cols[1][2])
    box(ax, cols[2][0], q_y, obs_w, q_h,
        "Q1: which bucket lost most mass?\nQ2: is the mass loss real capability lost\nor just K=1 sampling drift?",
        fs=9, fc="white", ec=cols[2][2])

    # ============ Row 4: Layers that answer ============
    l_y = 5.0
    l_h = 1.6

    # pass@K column → L3 confirms ceiling
    box(ax, cols[0][0], l_y, obs_w, l_h,
        "→ L3 (per-Q pass@K dist)\nfile: lr5e-4_difficulty_grid.png\n+ migration matrix\nFINDING: 0% Q fully lost (C=7%↔7%)\nceiling preserved",
        fs=8.5, fc="#f0fdf4", ec=cols[0][2])

    # maj@K column → L2.2 (bucket) + L4 (shape)
    box(ax, cols[1][0], l_y + 0.85, 2.4, 0.7,
        "→ L2.2 per-bucket maj@K\n(new findings png panel b)",
        fs=8, fc="#fffbeb", ec=cols[1][2])
    box(ax, cols[1][0] + 2.6, l_y + 0.85, 2.4, 0.7,
        "→ L4 mode/wrong conc\n(scatter + wrong_conc figs)",
        fs=8, fc="#fffbeb", ec=cols[1][2])
    box(ax, cols[1][0], l_y, obs_w, 0.75,
        "FINDING (L2.2): Easy -10pp / Medium -10pp / Hard +12pp\nFINDING (L4): TBD — attractor vs dilution",
        fs=8.5, fc="white", ec=cols[1][2])

    # pass@1 column → L2.1 (bucket) + L3 (capability vs sampling)
    box(ax, cols[2][0], l_y + 0.85, 2.4, 0.7,
        "→ L2.1 per-bucket pass@1\n(sft_per_bucket_trajectory)",
        fs=8, fc="#f0fdf4", ec=cols[2][2])
    box(ax, cols[2][0] + 2.6, l_y + 0.85, 2.4, 0.7,
        "→ L3 base→SFT migration\n(new findings png panel c)",
        fs=8, fc="#f0fdf4", ec=cols[2][2])
    box(ax, cols[2][0], l_y, obs_w, 0.75,
        "FINDING (L2.1): Easy -27pp dominant; FINDING (L3): real capability lost\n(Easy pass@K 0.98→0.71, not just K=1 drift)",
        fs=8.5, fc="white", ec=cols[2][2])

    # ============ Arrows: Row 1 → Row 2 → Row 3 → Row 4 ============
    for x_left, _, _ in cols:
        arrow(ax, x_left + obs_w / 2, obs_y, x_left + obs_w / 2, sem_y + sem_h)
        arrow(ax, x_left + obs_w / 2, sem_y, x_left + obs_w / 2, q_y + q_h)
        arrow(ax, x_left + obs_w / 2, q_y, x_left + obs_w / 2, l_y + l_h)

    # ============ Row 5: Cross-cutting layer (L5) ============
    box(ax, 0.5, 2.8, 17.0, 1.0,
        "L5 (physical mechanism, cross-cutting): how do response token length / step count changes "
        "implement the above mass redistribution?\n"
        "files: lr5e-4_length_grid.png, length_e1style/lr5e-4_step130_length_e1style.png, base length_classA.png",
        fs=9.5, fc="#f3e8ff", ec="#9333ea", fw="bold")
    arrow(ax, 3, l_y, 3, 3.8)
    arrow(ax, 9, l_y, 9, 3.8)
    arrow(ax, 15, l_y, 15, 3.8)

    # ============ Row 6: Synthesis ============
    box(ax, 0.5, 0.5, 17.0, 1.7,
        "SYNTHESIS (SFT lr=5e-4 step 130 behavior):\n"
        "1) capability ceiling preserved (pass@K nearly unchanged) — model still 'knows' all questions\n"
        "2) mass primarily attenuated on Easy bucket (pass@1: Easy -27pp dominant; this is REAL capability not sampling drift)\n"
        "3) mode flips spread Easy + Medium (-10pp each); Hard maj@K gains +12pp (format gain)\n"
        "→ SFT does mass redistribution NOT capability deletion; Easy bucket is the main victim",
        fs=10, fc="#fef9c3", ec="#a16207", fw="bold")
    arrow(ax, 3, 2.8, 3, 2.2)
    arrow(ax, 9, 2.8, 9, 2.2)
    arrow(ax, 15, 2.8, 15, 2.2)

    plt.tight_layout()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"saved: {OUT_FILE}")


if __name__ == "__main__":
    main()

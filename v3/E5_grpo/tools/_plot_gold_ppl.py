"""Gold-answer PPL comparison: base vs R15 ck-15 (DAPO) vs R16 step_42 (GRPO).

L1: PPL bar chart with deltas
L2: per-Q ΔPPL histogram (R16 - base, R15 - base) showing per-question drift
L3: per-Q PPL scatter (base x-axis, trained y-axis) — diagonal = no change
L4: Summary text + interpretation
"""
import json, math
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
_fm.fontManager.addfont("/mnt/c/Windows/Fonts/msyh.ttc")
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path("/mnt/d/fine-tuning")
DIR = ROOT / "v3/E5_grpo/outputs/gold_ppl"
OUT = DIR / "gold_ppl_comparison.png"

base = json.load(open(DIR / "base.json"))
sft = json.load(open(DIR / "sft_ck130.json"))
r15 = json.load(open(DIR / "r15_ck15.json"))
r16 = json.load(open(DIR / "r16_step42.json"))

# Aligned per-Q PPL arrays (index by q_idx)
def per_q_dict(d):
    return {r["q_idx"]: r for r in d["per_q"]}
b_dict, s_dict, r1_dict, r2_dict = per_q_dict(base), per_q_dict(sft), per_q_dict(r15), per_q_dict(r16)
common = sorted(set(b_dict) & set(s_dict) & set(r1_dict) & set(r2_dict))
ppl_b = np.array([b_dict[i]["ppl"] for i in common])
ppl_s = np.array([s_dict[i]["ppl"] for i in common])
ppl_r1 = np.array([r1_dict[i]["ppl"] for i in common])
ppl_r2 = np.array([r2_dict[i]["ppl"] for i in common])
print(f"n_aligned = {len(common)}")

COLOR_BASE = "black"
COLOR_SFT = "#16a34a"
COLOR_R15 = "#dc2626"
COLOR_R16 = "#7c3aed"

fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")
plt.subplots_adjust(hspace=0.45, wspace=0.30)

# ====== L1: mean PPL bar ======
ax = axes[0, 0]
methods = ["base IT", "SFT lr5e-4 ck-130", "R15 DAPO ck-15", "R16 GRPO step_42"]
ppls = [base["mean_ppl"], sft["mean_ppl"], r15["mean_ppl"], r16["mean_ppl"]]
colors = [COLOR_BASE, COLOR_SFT, COLOR_R15, COLOR_R16]
bars = ax.bar(range(4), ppls, color=colors, edgecolor="black", linewidth=0.7)
for i, p in enumerate(ppls):
    ax.text(i, p + 0.04, f"{p:.3f}", ha="center", fontsize=10, fontweight="semibold")
for i in [1, 2, 3]:
    delta = ppls[i] - ppls[0]
    col = "#15803d" if delta < 0 else "#b91c1c"
    ax.text(i, ppls[i] - 0.4, f"Δ={delta:+.3f}", ha="center", fontsize=9, color=col, fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(methods, fontsize=8.5)
ax.set_ylabel("mean PPL (lower better)")
ax.set_title("L1 — Gold-answer PPL (GSM8K test n=1319, token-weighted mean)\n"
             "SFT minimizes gold NLL → PPL ↓ ; RL optimizes reward → PPL ↑",
             loc="left", fontsize=9, fontweight="semibold")
ax.set_ylim(0, max(ppls) * 1.15)
ax.grid(axis="y", alpha=0.3, linestyle=":")

# ====== L2: per-Q ΔPPL histogram (SFT, R15, R16 − base) ======
ax = axes[0, 1]
delta_sft = ppl_s - ppl_b
delta_r15 = ppl_r1 - ppl_b
delta_r16 = ppl_r2 - ppl_b
bins = np.linspace(-3, 3, 61)
def clip(arr): return np.clip(arr, bins[0], bins[-1])
ax.hist(clip(delta_sft), bins=bins, alpha=0.55, color=COLOR_SFT, label=f"SFT ΔPPL (mean {delta_sft.mean():+.3f})", edgecolor="white", linewidth=0.3)
ax.hist(clip(delta_r15), bins=bins, alpha=0.55, color=COLOR_R15, label=f"R15 ΔPPL (mean {delta_r15.mean():+.3f})", edgecolor="white", linewidth=0.3)
ax.hist(clip(delta_r16), bins=bins, alpha=0.55, color=COLOR_R16, label=f"R16 ΔPPL (mean {delta_r16.mean():+.3f})", edgecolor="white", linewidth=0.3)
ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
n_sft_up, n_sft_dn = int((delta_sft > 0).sum()), int((delta_sft < 0).sum())
n_r15_up, n_r15_dn = int((delta_r15 > 0).sum()), int((delta_r15 < 0).sum())
n_r16_up, n_r16_dn = int((delta_r16 > 0).sum()), int((delta_r16 < 0).sum())
ax.set_xlabel("ΔPPL per question (trained − base, ←better)")
ax.set_ylabel("# questions")
ax.set_title(f"L2 — Per-Q ΔPPL distribution\n"
             f"SFT ↓PPL: {n_sft_dn} ({n_sft_dn/len(common)*100:.1f}%)  ↑PPL: {n_sft_up} ({n_sft_up/len(common)*100:.1f}%)\n"
             f"R15 ↑PPL: {n_r15_up} ({n_r15_up/len(common)*100:.1f}%)  ↓PPL: {n_r15_dn} ({n_r15_dn/len(common)*100:.1f}%)\n"
             f"R16 ↑PPL: {n_r16_up} ({n_r16_up/len(common)*100:.1f}%)  ↓PPL: {n_r16_dn} ({n_r16_dn/len(common)*100:.1f}%)",
             loc="left", fontsize=8.5, fontweight="semibold")
ax.legend(fontsize=8); ax.grid(alpha=0.25, linestyle=":")

# ====== L3: scatter base PPL vs trained PPL (3 methods) ======
ax = axes[1, 0]
xmax = max(np.percentile(ppl_b, 99), np.percentile(ppl_r2, 99)) * 1.05
ax.scatter(ppl_b, ppl_r2, s=4, alpha=0.25, color=COLOR_R16, label="R16 GRPO")
ax.scatter(ppl_b, ppl_r1, s=4, alpha=0.25, color=COLOR_R15, label="R15 DAPO")
ax.scatter(ppl_b, ppl_s, s=4, alpha=0.35, color=COLOR_SFT, label="SFT lr5e-4 ck-130")
ax.plot([0, xmax], [0, xmax], "--", color="black", linewidth=1, alpha=0.7, label="y=x (no change)")
ax.set_xlim(0, xmax); ax.set_ylim(0, xmax)
ax.set_xlabel("base PPL"); ax.set_ylabel("trained PPL")
ax.set_title("L3 — Per-Q PPL scatter (above diag = trained worse than base on gold;\n"
             "below diag = trained better)",
             loc="left", fontsize=9, fontweight="semibold")
ax.legend(fontsize=8, markerscale=3); ax.grid(alpha=0.25, linestyle=":")

# ====== L4: summary text ======
ax = axes[1, 1]; ax.axis("off")
summary = (
    "Gold-answer PPL interpretation:\n\n"
    f"• mean PPL:\n"
    f"    base IT       3.235\n"
    f"    SFT ck-130   {sft['mean_ppl']:.3f}  Δ={sft['mean_ppl']-base['mean_ppl']:+.3f}  ← 直接 minimize gold NLL\n"
    f"    R15 DAPO     {r15['mean_ppl']:.3f}  Δ={r15['mean_ppl']-base['mean_ppl']:+.3f}\n"
    f"    R16 GRPO     {r16['mean_ppl']:.3f}  Δ={r16['mean_ppl']-base['mean_ppl']:+.3f}\n\n"
    "对比训练目标:\n"
    "  SFT:  minimize -log P(gold | q)            → PPL ↓ 大幅 (训练目标 = 评测目标)\n"
    "  RL:   maximize E[reward(boxed-correct)]   → 跟 gold token 无关\n"
    "         模型 reroute prob 给自己偏好的 path → PPL ↑\n\n"
    "R16 GRPO drift 比 R15 DAPO 强:\n"
    f"  • SFT: {n_sft_dn}/{len(common)} = {n_sft_dn/len(common)*100:.1f}% Q PPL 下降 (好)\n"
    f"  • R15: {n_r15_up}/{len(common)} = {n_r15_up/len(common)*100:.1f}% Q PPL 上升 (差)\n"
    f"  • R16: {n_r16_up}/{len(common)} = {n_r16_up/len(common)*100:.1f}% Q PPL 上升 (差)\n\n"
    "DSMath claim 1 解释: RL re-rank 概率而非 expand capability.\n"
    "  pass@K 没动 (capability), 但 mode 偏离 gold path → gold PPL ↑\n"
    "  Eval target (boxed-correct accuracy) ↑ 但 reference matching ↓ 同时发生.\n\n"
    "SFT 是 reference-matching 优化 (PPL ↓), RL 是 outcome 优化 (PPL ↑ 副作用)."
)
ax.text(0.02, 0.98, summary, transform=ax.transAxes, ha="left", va="top",
        family="monospace", fontsize=9)

fig.suptitle("Gold-answer PPL — base IT / SFT / R15 DAPO / R16 GRPO (GSM8K test n=1319, forward-pass NLL)",
             fontsize=11, fontweight="semibold", y=0.995)
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved: {OUT}")

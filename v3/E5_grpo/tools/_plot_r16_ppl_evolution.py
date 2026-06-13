"""R16 GRPO Gold PPL evolution scatter — L3-style across step 32/42/58.

Layout: 3 scatter panels (one per ckpt) + 1 summary panel.
  X = base PPL, Y = R16 step_N PPL. Above diag = drift away from gold.
"""
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
_fm.fontManager.addfont("/mnt/c/Windows/Fonts/msyh.ttc")
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path("/mnt/d/fine-tuning")
DIR = ROOT / "v3/E5_grpo/outputs/gold_ppl"
OUT = DIR / "r16_ppl_evolution.png"

base = json.load(open(DIR / "base.json"))
r32 = json.load(open(DIR / "r16_step32.json"))
r42 = json.load(open(DIR / "r16_step42.json"))
r58 = json.load(open(DIR / "r16_step58.json"))

# Build q_idx → ppl dict for each
def per_q_dict(d):
    return {r["q_idx"]: r for r in d["per_q"]}
b_dict = per_q_dict(base)
dicts = [per_q_dict(r32), per_q_dict(r42), per_q_dict(r58)]
common = sorted(set(b_dict) & set(dicts[0]) & set(dicts[1]) & set(dicts[2]))
ppl_b = np.array([b_dict[i]["ppl"] for i in common])
ppl_32 = np.array([dicts[0][i]["ppl"] for i in common])
ppl_42 = np.array([dicts[1][i]["ppl"] for i in common])
ppl_58 = np.array([dicts[2][i]["ppl"] for i in common])
print(f"n_aligned = {len(common)}")

COLOR_BASE = "black"
COLOR_PURPLE = "#7c3aed"
xmax = max(np.percentile(ppl_b, 99), np.percentile(ppl_58, 99)) * 1.05

fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor="white")
plt.subplots_adjust(hspace=0.42, wspace=0.30)

panels = [(axes[0, 0], r32, ppl_32, 32, "#a78bfa"),
          (axes[0, 1], r42, ppl_42, 42, "#7c3aed"),
          (axes[1, 0], r58, ppl_58, 58, "#5b21b6")]

for ax, rd, ppl, step, color in panels:
    ax.scatter(ppl_b, ppl, s=4, alpha=0.30, color=color, edgecolor="none")
    ax.plot([0, xmax], [0, xmax], "--", color="black", linewidth=1, alpha=0.7, label="y=x (no change)")
    ax.set_xlim(0, xmax); ax.set_ylim(0, xmax)
    ax.set_xlabel("base PPL"); ax.set_ylabel(f"R16 step{step} PPL")
    n_up = int((ppl > ppl_b).sum()); n_dn = int((ppl < ppl_b).sum())
    delta = rd["mean_ppl"] - base["mean_ppl"]
    ax.set_title(f"R16 step{step}  PPL={rd['mean_ppl']:.3f}  Δ={delta:+.3f}\n"
                 f"↑PPL: {n_up} ({n_up/len(common)*100:.1f}%)  ↓PPL: {n_dn} ({n_dn/len(common)*100:.1f}%)",
                 fontsize=10, fontweight="semibold", loc="left")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.25, linestyle=":")

# Summary: mean PPL evolution bar
ax = axes[1, 1]
labels = ["base", "step32", "step42(peak)", "step58"]
ppls = [base["mean_ppl"], r32["mean_ppl"], r42["mean_ppl"], r58["mean_ppl"]]
colors = ["black", "#a78bfa", "#7c3aed", "#5b21b6"]
bars = ax.bar(range(4), ppls, color=colors, edgecolor="black", linewidth=0.7)
for i, p in enumerate(ppls):
    ax.text(i, p + 0.06, f"{p:.3f}", ha="center", fontsize=11, fontweight="semibold")
    if i > 0:
        ax.text(i, p - 0.35, f"Δ={p-ppls[0]:+.3f}", ha="center", fontsize=9, color="#b91c1c", fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("mean gold PPL (lower = closer to gold)")
ax.set_ylim(0, max(ppls) * 1.15)
ax.set_title("R16 GRPO gold-PPL monotonic drift through training\n"
             "step→ : RL re-routes probability mass AWAY from gold reasoning path",
             fontsize=10, fontweight="semibold", loc="left")
ax.grid(axis="y", alpha=0.3, linestyle=":")

# Dev curve overlay (right y-axis)
ax2 = ax.twinx()
dev_steps = [0, 32, 42, 58]
dev_acc = [None, 75.4, 77.2, 75.2]  # from r16_dev_eval (step 0 is base — set None or 32.2 from step 1)
dev_acc_clean = [v for v in dev_acc if v is not None]
ax2.plot(range(1, 4), dev_acc_clean, "o-", color="#16a34a", linewidth=2, markersize=10, label="dev acc%")
for i, v in enumerate(dev_acc_clean):
    ax2.text(i+1, v + 0.4, f"{v:.1f}", ha="center", fontsize=9, color="#16a34a")
ax2.set_ylabel("dev acc % (n=500)", color="#16a34a")
ax2.tick_params(axis="y", labelcolor="#16a34a")
ax2.set_ylim(70, 80)
ax2.legend(loc="lower right", fontsize=8)

fig.suptitle("R16 GRPO Gold-PPL evolution — base / step 32 / step 42 (dev-peak) / step 58 "
             f"(GSM8K test n={len(common)}, forward-pass NLL)",
             fontsize=11.5, fontweight="semibold", y=0.995)
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved → {OUT}")

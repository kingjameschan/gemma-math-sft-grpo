"""DAPO ck-15 — per-question c shift analysis (pass@1 mechanism).

Pass@1 dynamics 在 A/B/C bucket transition 里看不到 (桶内 c 值变化 invisible).
这个图直接显示**每题 c (out of 64) 的 base→DAPO 变化**, 揭示 pass@1 +4.22pp 的来源.
"""
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
_fm.fontManager.addfont("/mnt/c/Windows/Fonts/msyh.ttc")
import numpy as np

ROOT = Path("/mnt/d/fine-tuning")
DAPO_K64 = ROOT / "v3/E5_grpo/outputs/k64_dapo_ck15/r15_dapo_checkpoint-15_k64.json"
BASE_K64 = ROOT / "v3/E1_baseline/outputs/pass_at_k_20260427_222954/base_gemma-2-2b-it_k64.json"
TEST_PC = ROOT / "v3/shared/data/gsm8k/test_pc.jsonl"
OUT_FILE = ROOT / "v3/E5_grpo/outputs/k64_dapo_ck15/dapo_pass1_mechanism.png"


def normalize(s):
    if s is None: return None
    s = str(s).strip().replace(",", "").replace("$", "").replace(" ", "")
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v): return s
        return str(int(v)) if v == int(v) else str(v)
    except: return s


def bucket(c, mode_correct):
    if c == 0: return "C"
    return "A" if mode_correct else "B"


def main():
    golds = []
    for line in open(TEST_PC):
        ex = json.loads(line); txt = ex["completion"][0]["content"]
        if "\\boxed{" in txt:
            e = txt.rfind("}"); s = txt.rfind("\\boxed{") + len("\\boxed{")
            golds.append(normalize(txt[s:e].strip()))
        else: golds.append(None)

    bd = json.load(open(BASE_K64))
    base_ans = [[normalize(a) for a in s["any_preds"]] for s in bd["samples"]]
    dd = json.load(open(DAPO_K64))
    dapo_ans = [[normalize(a) for a in s["any_preds"]] for s in dd["samples"]]

    n_q = len(golds)
    rows = []
    for i in range(n_q):
        b_cnt = Counter(base_ans[i]); d_cnt = Counter(dapo_ans[i])
        b_c = b_cnt.get(golds[i], 0); d_c = d_cnt.get(golds[i], 0)
        b_mode = max(b_cnt.items(), key=lambda kv: kv[1])[0] if b_cnt else None
        d_mode = max(d_cnt.items(), key=lambda kv: kv[1])[0] if d_cnt else None
        rows.append(dict(
            qi=i, base_c=b_c, dapo_c=d_c,
            base_bk=bucket(b_c, b_mode == golds[i]),
            dapo_bk=bucket(d_c, d_mode == golds[i]),
        ))

    delta_c = np.array([r["dapo_c"] - r["base_c"] for r in rows])
    base_c = np.array([r["base_c"] for r in rows])
    dapo_c = np.array([r["dapo_c"] for r in rows])

    # Net Δsum_c check
    sum_dc = delta_c.sum()
    pass1_diff_pp = sum_dc / (64 * n_q) * 100
    print(f"=== pass@1 mechanism ===")
    print(f"  Total Δsum_c = {sum_dc:+d} samples  (={sum_dc/64*100/n_q:+.2f}pp pass@1)")
    print(f"  mean Δc per question = {delta_c.mean():+.3f}")
    print(f"  # Q with Δc>0: {(delta_c > 0).sum()}  ({(delta_c>0).mean()*100:.1f}%)")
    print(f"  # Q with Δc<0: {(delta_c < 0).sum()}  ({(delta_c<0).mean()*100:.1f}%)")
    print(f"  # Q with Δc=0: {(delta_c == 0).sum()}  ({(delta_c==0).mean()*100:.1f}%)")
    print()

    # Per-bucket-transition Δsum_c
    print(f"=== Δsum_c by transition (sum of c shifts within each transition cell) ===")
    trans_dc = {}
    for t in [("A","A"),("A","B"),("A","C"),("B","A"),("B","B"),("B","C"),("C","A"),("C","B"),("C","C")]:
        sel = [(r["dapo_c"] - r["base_c"]) for r in rows if r["base_bk"] == t[0] and r["dapo_bk"] == t[1]]
        if sel:
            trans_dc[t] = (sum(sel), len(sel), np.mean(sel))
            print(f"  {t[0]}→{t[1]}: n={len(sel):4d}  Σ Δc={sum(sel):+6d}  mean Δc={np.mean(sel):+.2f}")

    # ============== Plot ==============
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (TL) Per-Q Δc histogram
    ax = axes[0, 0]
    bins = np.arange(-65, 66, 2)
    ax.hist(delta_c, bins=bins, color="#3b82f6", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.0, alpha=0.6)
    ax.axvline(delta_c.mean(), color="#dc2626", linestyle="--", linewidth=1.5,
               label=f"mean Δc = {delta_c.mean():+.2f}")
    ax.set_xlabel("Δc = c_DAPO - c_base  (out of 64)")
    ax.set_ylabel("# questions")
    ax.set_title(f"L1 — per-Q Δc 直方图\n"
                 f"+Δc: {(delta_c>0).sum()} 题 ({(delta_c>0).mean()*100:.1f}%) | "
                 f"-Δc: {(delta_c<0).sum()} | 0: {(delta_c==0).sum()}\n"
                 f"Σ Δc = {sum_dc:+d} = +{pass1_diff_pp:.2f}pp pass@1",
                 loc="left", fontsize=10, fontweight="semibold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle=":")

    # (TR) base c vs DAPO c scatter
    ax = axes[0, 1]
    # color by transition type
    colors_t = {
        ("A","A"): "#16a34a", ("A","B"): "#f59e0b", ("A","C"): "#dc2626",
        ("B","A"): "#2563eb", ("B","B"): "#9ca3af", ("B","C"): "#dc2626",
        ("C","A"): "#16a34a", ("C","B"): "#3b82f6", ("C","C"): "#1f2937",
    }
    for t in [("C","C"),("A","A"),("B","B"),("A","B"),("B","A"),("B","C"),("A","C"),("C","A"),("C","B")]:
        sel_rows = [r for r in rows if (r["base_bk"], r["dapo_bk"]) == t]
        if not sel_rows: continue
        xs = [r["base_c"] for r in sel_rows]
        ys = [r["dapo_c"] for r in sel_rows]
        ax.scatter(xs, ys, c=colors_t[t], s=12, alpha=0.5, label=f"{t[0]}→{t[1]} ({len(sel_rows)})", edgecolor="none")
    ax.plot([0, 64], [0, 64], "--", color="black", alpha=0.4, linewidth=1, label="y=x (no change)")
    ax.set_xlabel("base c (correct in 64)")
    ax.set_ylabel("DAPO c (correct in 64)")
    ax.set_xlim(-2, 66); ax.set_ylim(-2, 66)
    ax.set_title("L2 — per-Q c scatter (base → DAPO)\n"
                 "上方 = c 涨 (DAPO better); 下方 = c 跌 (regression)",
                 loc="left", fontsize=10, fontweight="semibold")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=0.3, linestyle=":")

    # (BL) Within stay-A bucket: c_base vs c_DAPO distribution overlay
    ax = axes[1, 0]
    a_rows = [r for r in rows if r["base_bk"] == "A" and r["dapo_bk"] == "A"]
    a_base = np.array([r["base_c"] for r in a_rows])
    a_dapo = np.array([r["dapo_c"] for r in a_rows])
    bins_c = np.arange(0, 66, 2)
    ax.hist(a_base, bins=bins_c, alpha=0.5, color="black", label=f"base c (mean={a_base.mean():.1f}, n={len(a_base)})", edgecolor="white")
    ax.hist(a_dapo, bins=bins_c, alpha=0.5, color="#dc2626", label=f"DAPO c (mean={a_dapo.mean():.1f}, Δ={a_dapo.mean()-a_base.mean():+.2f})", edgecolor="white")
    ax.axvline(a_base.mean(), color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(a_dapo.mean(), color="#dc2626", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlabel("c (correct count in 64 sample)")
    ax.set_ylabel("# questions")
    ax.set_title(f"L3 — Stay-A 桶 (873 题) 内的 c 分布对比\n"
                 f"DAPO 把 c 分布整体推右 (sharpen toward 64) → pass@1 主战场",
                 loc="left", fontsize=10, fontweight="semibold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle=":")

    # (BR) Contribution breakdown — Σ Δc by transition
    ax = axes[1, 1]
    cells = [("A","A"),("B","A"),("C","A"),("C","B"),("A","B"),("B","B"),("C","C"),("A","C"),("B","C")]
    contribs = [trans_dc.get(t, (0, 0, 0))[0] for t in cells]
    counts = [trans_dc.get(t, (0, 0, 0))[1] for t in cells]
    colors_bar = ["#16a34a" if v > 0 else "#9ca3af" if v == 0 else "#dc2626" for v in contribs]
    x = np.arange(len(cells))
    bars = ax.bar(x, contribs, color=colors_bar, edgecolor="white", linewidth=1)
    for i, (v, n) in enumerate(zip(contribs, counts)):
        if v != 0:
            ax.text(i, v + (60 if v > 0 else -80), f"{v:+d}\n(n={n})", ha="center",
                    fontsize=8, fontweight="semibold")
        else:
            ax.text(i, 100, f"0\n(n={n})", ha="center", fontsize=8, color="#666")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{t[0]}→{t[1]}" for t in cells], fontsize=9)
    ax.set_ylabel("Σ Δc (sum of correct count gain)")
    ax.set_title(f"L4 — Σ Δc by transition (pass@1 净 +{sum_dc} 来自哪)\n"
                 f"绿 = 净 c 涨 (pass@1 贡献), 红 = 净 c 跌, 灰 = 0 题",
                 loc="left", fontsize=10, fontweight="semibold")
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    fig.suptitle(
        f"DAPO ck-15 vs base IT — per-question c shift 分析 (pass@1 +{pass1_diff_pp:.2f}pp 的 mechanism)\n"
        f"A/B/C transition matrix 看不到桶内 c 变化, 这里直接看 c shift",
        fontsize=12, fontweight="semibold", y=1.00,
    )
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nsaved: {OUT_FILE}")


if __name__ == "__main__":
    main()

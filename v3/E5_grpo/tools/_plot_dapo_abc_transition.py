"""DAPO ck-15 — A/B/C bucket transition heatmap (base → DAPO).

L1.2 analysis 配套图. A/B/C 分桶基于 mode-correctness:
  A: pass@64=1 且 mode 是 correct (= maj@64=1 的题)
  B: pass@64=1 但 mode 是 wrong (correct 在 support 里但投错)
  C: pass@64=0 (64 sample 全错)

矩阵每个 cell: 从 base bucket → DAPO bucket 的题数.
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
OUT_FILE = ROOT / "v3/E5_grpo/outputs/k64_dapo_ck15/dapo_abc_transition.png"


def normalize(s):
    if s is None: return None
    s = str(s).strip().replace(",", "").replace("$", "").replace(" ", "")
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v): return s
        return str(int(v)) if v == int(v) else str(v)
    except: return s


def bucket(ans, gold):
    cnt = Counter(ans)
    c = cnt.get(gold, 0)
    if c == 0:
        return "C"
    mode = max(cnt.items(), key=lambda kv: kv[1])[0]
    return "A" if mode == gold else "B"


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
    M = np.zeros((3, 3), dtype=int)
    idx = {"A": 0, "B": 1, "C": 2}
    for i in range(n_q):
        b = bucket(base_ans[i], golds[i])
        d = bucket(dapo_ans[i], golds[i])
        M[idx[b], idx[d]] += 1

    base_sz = M.sum(axis=1)
    dapo_sz = M.sum(axis=0)

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, (ax, ax_txt) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    im = ax.imshow(M, cmap="Blues", aspect="auto")

    bucket_labels = ["A\n(mode=对)", "B\n(≥1对,mode 错)", "C\n(0/64 对)"]
    bucket_desc = {
        "A": "polish OK (maj@64=correct)",
        "B": "knows but votes wrong",
        "C": "capability gap (in 64 sample)",
    }

    # Cell annotations
    for i in range(3):
        row_total = M[i].sum()
        for j in range(3):
            cnt = M[i, j]
            pct_row = cnt / row_total * 100 if row_total else 0
            color = "white" if cnt > M.max() * 0.5 else "black"
            ax.text(j, i, f"{cnt}\n({pct_row:.0f}%)", ha="center", va="center",
                    color=color, fontsize=13, fontweight="semibold")

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([f"→ DAPO {l}\n[{dapo_sz[idx[l[0]]]}]" for l in ["A", "B", "C"]], fontsize=10)
    ax.set_yticklabels([f"base {bucket_labels[i]}\n[{base_sz[i]}]" for i in range(3)], fontsize=10)
    ax.set_title("A/B/C bucket transition (base IT → DAPO ck-15) on 1319 GSM8K test\n"
                 f"  A={base_sz[0]}→{dapo_sz[0]} (Δ{dapo_sz[0]-base_sz[0]:+d}) | "
                 f"B={base_sz[1]}→{dapo_sz[1]} (Δ{dapo_sz[1]-base_sz[1]:+d}) | "
                 f"C={base_sz[2]}→{dapo_sz[2]} (Δ{dapo_sz[2]-base_sz[2]:+d})",
                 loc="left", fontsize=11, fontweight="semibold", pad=15)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="# questions")

    # Annotation: key transitions
    annot_lines = [
        "关键 transitions:",
        f"  A→A: {M[0,0]} (95%, 安全)",
        f"  A→C: {M[0,2]} ★ 0 题! (DAPO 不杀 base 好题)",
        f"  B→A: {M[1,0]} (mode 修对, 成功 polish)",
        f"  B→C: {M[1,2]} ★ catastrophic backfire (锐化错方向)",
        f"  C→A: {M[2,0]} ★ 0 题! (DAPO 救不出 stable correct)",
        f"  C→B: {M[2,1]} (support 扩张但 mode 不稳)",
        f"  C→C: {M[2,2]} (persistent capability gap)",
        "",
        "Net 效应:",
        f"  maj@64 flip: +{M[1,0]-M[0,1]} 题 (B→A {M[1,0]} - A→B {M[0,1]})",
        f"  pass@64 net: {M[2,1]-M[1,2]:+d} 题 (recovered {M[2,1]} - newly_lost {M[1,2]})",
    ]
    ax_txt.axis("off")
    ax_txt.text(0.0, 1.0, "\n".join(annot_lines),
                fontsize=11, family="monospace",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#fef9c3", edgecolor="gray"))

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved: {OUT_FILE}")


if __name__ == "__main__":
    main()

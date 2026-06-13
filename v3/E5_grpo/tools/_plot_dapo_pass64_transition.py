"""DAPO ck-15 — pass@64 per-question transition analysis.

Pass@64 是 binary (= 1 iff c≥1, = 0 iff c=0), 所以 transition 只有 4 类:
  (1,1) both-pass  : base 在 support, DAPO 也在 support
  (0,0) both-fail  : 两者都 0/64
  (0,1) recovered  : base=0, DAPO≥1   ← DAPO 救活
  (1,0) newly_lost : base≥1, DAPO=0   ← DAPO 误锐化压死

为 (0,1) recovered 和 (1,0) newly_lost 各做 profile:
  - base IT 上的 P(correct) c/64 分布 (newly_lost 该多是 rare-tail)
  - difficulty bucket 分布 (Easy/Medium/Hard)
  - base IT mode_correct status (新丢的多是 base mode-correct 还是 mode-wrong?)
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
LABELS = ROOT / "v3/shared/data/gsm8k/test_difficulty_labels.jsonl"
TEST_PC = ROOT / "v3/shared/data/gsm8k/test_pc.jsonl"
OUT_FILE = ROOT / "v3/E5_grpo/outputs/k64_dapo_ck15/dapo_pass64_transition.png"


def normalize(s):
    if s is None: return None
    s = str(s).strip().replace(",", "").replace("$", "").replace(" ", "")
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v): return s
        return str(int(v)) if v == int(v) else str(v)
    except: return s


def main():
    golds = []
    for line in open(TEST_PC):
        ex = json.loads(line); txt = ex["completion"][0]["content"]
        if "\\boxed{" in txt:
            e = txt.rfind("}"); s = txt.rfind("\\boxed{") + len("\\boxed{")
            golds.append(normalize(txt[s:e].strip()))
        else:
            golds.append(None)
    labels = {}
    for line in open(LABELS):
        d = json.loads(line); labels[d["question_idx"]] = d["bucket"]

    bd = json.load(open(BASE_K64))
    base_ans = [[normalize(a) for a in s["any_preds"]] for s in bd["samples"]]
    dd = json.load(open(DAPO_K64))
    dapo_ans = [[normalize(a) for a in s["any_preds"]] for s in dd["samples"]]

    n_q = len(golds)
    rows = []
    for i in range(n_q):
        gold = golds[i]
        b_cnt = Counter(base_ans[i]); d_cnt = Counter(dapo_ans[i])
        b_c = b_cnt.get(gold, 0); d_c = d_cnt.get(gold, 0)
        b_mode = max(b_cnt.items(), key=lambda kv: kv[1])[0] if b_cnt else None
        d_mode = max(d_cnt.items(), key=lambda kv: kv[1])[0] if d_cnt else None
        rows.append(dict(
            qi=i, bucket=labels.get(i, "?"),
            base_c=b_c, dapo_c=d_c,
            base_mode_correct=(b_mode == gold), dapo_mode_correct=(d_mode == gold),
        ))

    def cat(r):
        bp = r["base_c"] >= 1; dp = r["dapo_c"] >= 1
        if bp and dp: return "11_both_pass"
        if not bp and not dp: return "00_both_fail"
        if not bp and dp: return "01_recovered"
        return "10_newly_lost"

    for r in rows:
        r["transition"] = cat(r)

    by_t = {k: [r for r in rows if r["transition"] == k] for k in ["11_both_pass", "00_both_fail", "01_recovered", "10_newly_lost"]}
    print("=== pass@64 transition counts ===")
    for k, lst in by_t.items():
        print(f"  {k:<18}: {len(lst):4d} ({len(lst)/n_q*100:.2f}%)")

    # Profile: difficulty bucket split per transition
    print("\n=== difficulty bucket split per transition ===")
    print(f"  {'transition':<18} {'Easy':>6} {'Medium':>7} {'Hard':>5}")
    for k in ["11_both_pass", "00_both_fail", "01_recovered", "10_newly_lost"]:
        lst = by_t[k]
        bks = Counter(r["bucket"] for r in lst)
        print(f"  {k:<18} {bks.get('Easy', 0):>6} {bks.get('Medium', 0):>7} {bks.get('Hard', 0):>5}")

    # Profile: base IT correct count distribution per transition
    print("\n=== base IT c (correct in 64) distribution per transition ===")
    print(f"  {'transition':<18} {'mean':>6} {'median':>7} {'min':>4} {'max':>4}")
    for k in ["11_both_pass", "00_both_fail", "01_recovered", "10_newly_lost"]:
        lst = by_t[k]
        if not lst: continue
        cs = np.array([r["base_c"] for r in lst])
        print(f"  {k:<18} {cs.mean():>6.2f} {np.median(cs):>7.1f} {cs.min():>4d} {cs.max():>4d}")

    # base IT mode_correct status per transition (for the 39 newly_lost especially)
    print("\n=== base IT mode_correct status per transition ===")
    print(f"  {'transition':<18} {'mode=correct':>13} {'mode=wrong':>11}")
    for k in ["11_both_pass", "00_both_fail", "01_recovered", "10_newly_lost"]:
        lst = by_t[k]
        if not lst: continue
        mc = sum(1 for r in lst if r["base_mode_correct"])
        mw = len(lst) - mc
        print(f"  {k:<18} {mc:>13d} {mw:>11d}")

    # ============== Plot ==============
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    # (A) Transition counts donut/bar
    ax = fig.add_subplot(gs[0, 0])
    cats = ["11_both_pass", "00_both_fail", "01_recovered", "10_newly_lost"]
    labels_full = ["both pass (1,1)\nDAPO 没改变 support", "both fail (0,0)\n两者都 0/64 (capability ceiling)",
                   "recovered (0,1)\nDAPO 救活", "newly_lost (1,0)\nDAPO 锐化压死"]
    counts = [len(by_t[c]) for c in cats]
    colors = ["#9ca3af", "#1f2937", "#16a34a", "#dc2626"]
    bars = ax.bar(range(4), counts, color=colors)
    for i, (b, c) in enumerate(zip(bars, counts)):
        ax.text(i, c + 15, f"{c}\n({c/n_q*100:.1f}%)", ha="center", fontsize=9, fontweight="semibold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["(1,1)", "(0,0)", "(0,1)\nrecover", "(1,0)\nlost"], fontsize=8)
    ax.set_ylabel("# questions")
    ax.set_title("L1 — pass@64 transition 计数 (base→DAPO)\n净支持变化 = recover - lost = 16 - 39 = -23",
                 loc="left", fontsize=10, fontweight="semibold")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    legend_txt = "\n".join([f"{c}: {l.split(chr(10))[0]}" for c, l in zip(["(1,1)", "(0,0)", "(0,1)", "(1,0)"], labels_full)])

    # (B) Difficulty bucket split (stacked bar per category)
    ax = fig.add_subplot(gs[0, 1])
    bks = ["Easy", "Medium", "Hard"]
    bk_colors = ["#16a34a", "#f59e0b", "#dc2626"]
    width = 0.6
    x = np.arange(4)
    bottoms = np.zeros(4)
    for bi, bk in enumerate(bks):
        vals = np.array([sum(1 for r in by_t[c] if r["bucket"] == bk) for c in cats])
        ax.bar(x, vals, width, bottom=bottoms, color=bk_colors[bi], label=bk, alpha=0.85)
        for i, v in enumerate(vals):
            if v > 5:
                ax.text(i, bottoms[i] + v / 2, str(int(v)), ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
        bottoms += vals
    ax.set_xticks(x); ax.set_xticklabels(["(1,1)", "(0,0)", "(0,1)\nrecover", "(1,0)\nlost"], fontsize=8)
    ax.set_ylabel("# questions")
    ax.set_title("L2 — difficulty bucket × transition\n(Easy=高 base pass, Hard=低 base pass)",
                 loc="left", fontsize=10, fontweight="semibold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    # (C) base IT mode_correct status per transition
    ax = fig.add_subplot(gs[0, 2])
    mc_data = []
    mw_data = []
    for c in cats:
        lst = by_t[c]
        mc = sum(1 for r in lst if r["base_mode_correct"])
        mw = len(lst) - mc
        mc_data.append(mc); mw_data.append(mw)
    ax.bar(x, mc_data, width, color="#16a34a", label="base mode=correct", alpha=0.85)
    ax.bar(x, mw_data, width, bottom=mc_data, color="#dc2626", label="base mode=wrong", alpha=0.85)
    for i, (mc, mw) in enumerate(zip(mc_data, mw_data)):
        if mc > 5:
            ax.text(i, mc / 2, str(int(mc)), ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        if mw > 5:
            ax.text(i, mc + mw / 2, str(int(mw)), ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["(1,1)", "(0,0)", "(0,1)\nrecover", "(1,0)\nlost"], fontsize=8)
    ax.set_ylabel("# questions")
    ax.set_title("L3 — base IT mode status × transition\n关注 (1,0): base mode 是对是错? → 看 newly_lost 类型",
                 loc="left", fontsize=10, fontweight="semibold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    # (D) base IT c (correct count) distribution for newly_lost + recovered
    ax = fig.add_subplot(gs[1, 0])
    nl_c = [r["base_c"] for r in by_t["10_newly_lost"]]
    rec_c = [r["base_c"] for r in by_t["01_recovered"]]
    bins = np.arange(0, 66, 2)
    ax.hist(nl_c, bins=bins, color="#dc2626", alpha=0.7, edgecolor="white",
            label=f"newly_lost (n={len(nl_c)}, mean={np.mean(nl_c):.1f})")
    if rec_c:
        ax.hist(rec_c, bins=bins, color="#16a34a", alpha=0.6, edgecolor="white",
                label=f"recovered (n={len(rec_c)}, mean={np.mean(rec_c):.1f})")
    ax.set_xlabel("base IT 上 c (64 sample 中正确数)")
    ax.set_ylabel("# questions")
    ax.set_title("L4 — newly_lost vs recovered 的 base IT 'c' 分布\nnewly_lost 应集中在 low-c (rare tail)",
                 loc="left", fontsize=10, fontweight="semibold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, linestyle=":")

    # (E) DAPO c distribution for recovered (how much DAPO recovered)
    ax = fig.add_subplot(gs[1, 1])
    rec_d_c = [r["dapo_c"] for r in by_t["01_recovered"]]
    ax.hist(rec_d_c, bins=bins, color="#16a34a", alpha=0.7, edgecolor="white",
            label=f"recovered: DAPO c (n={len(rec_d_c)})")
    ax.set_xlabel("DAPO 上 c (64 sample 中正确数)")
    ax.set_ylabel("# questions")
    ax.set_title("L5 — DAPO 救活 (recovered) 题的 DAPO c 分布\n看 DAPO 救活的稳定度 (高 c = 真稳定救活, 低 c = 边缘)",
                 loc="left", fontsize=10, fontweight="semibold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, linestyle=":")

    # (F) Summary stats text
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    txt = []
    txt.append("=== Pass@64 Transition Summary ===\n")
    for c, lab in zip(cats, ["(1,1) both_pass", "(0,0) both_fail", "(0,1) recovered", "(1,0) newly_lost"]):
        lst = by_t[c]
        bks_d = Counter(r["bucket"] for r in lst)
        mc = sum(1 for r in lst if r["base_mode_correct"])
        cs = [r["base_c"] for r in lst]
        cs_mean = np.mean(cs) if cs else 0
        txt.append(f"\n{lab}: {len(lst)} 题 ({len(lst)/n_q*100:.1f}%)")
        txt.append(f"  bucket: E={bks_d.get('Easy', 0)} M={bks_d.get('Medium', 0)} H={bks_d.get('Hard', 0)}")
        txt.append(f"  base mode-correct: {mc}/{len(lst)} ({mc/max(len(lst), 1)*100:.0f}%)")
        txt.append(f"  base IT mean c: {cs_mean:.1f}/64 ({cs_mean/64*100:.0f}%)")
    txt.append("\n关键 insight:")
    txt.append("• newly_lost 39 题, base 平均 c 应低")
    txt.append("  → 这些是 'rare-tail correct' 的题")
    txt.append("  → DAPO 锐化压死了 long-tail")
    txt.append("\n• recovered 16 题, base c=0 (定义)")
    txt.append("  → DAPO 在这些题上**生成了之前没有的对答案**")
    txt.append("  → 这是 'capability gain' (真正学到东西)")
    ax.text(0, 1, "\n".join(txt), fontsize=8.5, family="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f9fafb", edgecolor="gray"))

    fig.suptitle(
        "DAPO ck-15 vs base IT — pass@64 per-question transition 分析  (1319 题)\n"
        "definition: pass@64=1 iff c≥1 in 64 sample → 4 类 (1,1)/(0,0)/(0,1)/(1,0)",
        fontsize=12, fontweight="semibold", y=1.00,
    )
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nsaved: {OUT_FILE}")


if __name__ == "__main__":
    main()

"""DAPO ck-15 vs base IT — single combined figure (集合图, 5 layers).

Mirror of v3/E2_sft/tools/_plot_step130_combined.py for DAPO R15 ck-15.
DAPO JSON sample is samples[i] dict (question/gold/any_preds/responses), differs
from SFT JSON layout (top-level per_sample_answers/per_sample_responses lists);
we extract DAPO into the same in-memory shape, then everything downstream is same.

Color: DAPO uses red (was green for SFT) to distinguish in 3-method narrative.
K curves drawn line-only (no markers) per user 2026-05-11.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
_fm.fontManager.addfont("/mnt/c/Windows/Fonts/msyh.ttc")
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
DAPO_K64 = ROOT / "v3" / "E5_grpo" / "outputs" / "k64_dapo_ck15" / "r15_dapo_checkpoint-15_k64.json"
BASE_K64 = ROOT / "v3" / "E1_baseline" / "outputs" / "pass_at_k_20260427_222954" / "base_gemma-2-2b-it_k64.json"
# MATH-500-aug (500 q): base K=128 verbose subset + DAPO K=64 verbose (fresh)
BASE_MATH_K64 = ROOT / "v3" / "E5_grpo" / "outputs" / "k128_merged_math500_aug_slice" / "base_k128_math500_aug_verbose.json"
DAPO_MATH_K64 = ROOT / "v3" / "E5_grpo" / "outputs" / "pass_at_k_math500_aug_20260518_231346" / "r15_dapo_checkpoint-15_k128_merged.json"
LABELS = ROOT / "v3" / "shared" / "data" / "gsm8k" / "test_difficulty_labels.jsonl"
TEST_PC = ROOT / "v3" / "shared" / "data" / "gsm8k" / "test_pc.jsonl"
BASE_MODEL = ROOT / "models" / "gemma-2-2b-it"

OUT_FILE = ROOT / "v3" / "E5_grpo" / "outputs" / "k64_dapo_ck15" / "dapo_ck15_combined.png"
TAG = "DAPO ck-15"
STEP = 15  # Just for title display (DAPO ckpt name)
LR = "DAPO"  # Display name

K = 64
KS = [1, 2, 4, 8, 16, 32, 64]
COLOR_BASE = "black"
COLOR_SFT = "#dc2626"  # red for DAPO (was green for SFT)
STEP_RE = re.compile(r"\*\*\s*(?:Step\s+)?\d+\.", re.IGNORECASE)


def normalize(s):
    if s is None: return None
    s = str(s).strip().replace(",", "").replace("$", "").replace(" ", "")
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v): return s
        return str(int(v)) if v == int(v) else str(v)
    except (ValueError, TypeError, OverflowError):
        return s


def pass_at_k_unbiased(c, n, k):
    if n - c < k: return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def per_q_metrics(ans_list, golds, pre_c=None):
    cmass, mmass, wconc, mode_correct, c_arr = [], [], [], [], []
    for i, (ans, gold) in enumerate(zip(ans_list, golds)):
        cnt = Counter(ans)
        total = sum(cnt.values())
        # Use pre-computed c (any_correct_per_K) from eval script if given — matches L2 definition
        c = pre_c[i] if pre_c is not None else cnt.get(gold, 0)
        cmass.append(c / total)
        mmass.append(max(cnt.values()) / total)
        mode_ans = max(cnt.items(), key=lambda kv: kv[1])[0]
        mode_correct.append(1 if mode_ans == gold else 0)
        c_arr.append(c)
        wrong_total = total - c
        if wrong_total > 0:
            non_correct = {k: v for k, v in cnt.items() if k != gold}
            top_wrong = max(non_correct.values()) if non_correct else 0
            wconc.append(top_wrong / wrong_total)
        else:
            wconc.append(None)
    return {
        "cmass": np.array(cmass), "mmass": np.array(mmass),
        "wconc": wconc, "mode_correct": np.array(mode_correct),
        "c": np.array(c_arr),
    }


def passk_majk_curves(c_arr, ans_list, golds):
    """maj@k via chain-permutation expectation (DSmath/Yue style).
       Recompute c locally so pass@K and maj@K use the same correctness criterion."""
    import random as _r
    passk, majk = {}, {}
    n_q = len(ans_list)
    rng = _r.Random(42)
    T = 200
    # Recompute c via the same normalize() string-eq used by maj
    c_local = [sum(1 for a in row if a is not None and a == g) for row, g in zip(ans_list, golds)]
    for k in KS:
        passk[k] = sum(pass_at_k_unbiased(c, K, k) for c in c_local) / n_q * 100
        maj_sum = 0.0
        for row, g in zip(ans_list, golds):
            if not row: continue
            hits = 0
            for _ in range(T):
                shuf_k = rng.sample(row, min(k, len(row)))
                valid = [a for a in shuf_k if a is not None]
                if not valid: continue
                if Counter(valid).most_common(1)[0][0] == g:
                    hits += 1
            maj_sum += hits / T
        majk[k] = maj_sum / n_q * 100
    return passk, majk


def main():
    # Load
    print("[load] golds + labels...")
    golds, labels = [], {}
    for line in open(TEST_PC):
        ex = json.loads(line)
        txt = ex["completion"][0]["content"]
        if "\\boxed{" in txt:
            e = txt.rfind("}"); s = txt.rfind("\\boxed{") + len("\\boxed{")
            golds.append(normalize(txt[s:e].strip()))
        else:
            golds.append(None)
    for line in open(LABELS):
        d = json.loads(line)
        labels[d["question_idx"]] = d["bucket"]

    print(f"[load] base K=64 + DAPO ck-15 K=64...")
    bd = json.load(open(BASE_K64))
    base_ans = [[normalize(a) for a in s["any_preds"]] for s in bd["samples"]]
    base_resps = [s["responses"] for s in bd["samples"]]
    # DAPO JSON: samples[i] is dict {question, gold, any_preds, responses, ...}
    dd = json.load(open(DAPO_K64))
    sft_ans = [[normalize(a) for a in s["any_preds"]] for s in dd["samples"]]
    sft_resps = [s["responses"] for s in dd["samples"]]

    n_q = len(golds)
    # Use eval-script pre-computed any_correct_per_K (L2-aligned, math_equal strict compare)
    base_pre_c = [s["any_correct_per_K"] for s in bd["samples"]]
    dapo_pre_c = [s["any_correct_per_K"] for s in dd["samples"]]
    base = per_q_metrics(base_ans, golds, pre_c=base_pre_c)
    sft = per_q_metrics(sft_ans, golds, pre_c=dapo_pre_c)
    base_passk, base_majk = passk_majk_curves(base["c"], base_ans, golds)
    sft_passk, sft_majk = passk_majk_curves(sft["c"], sft_ans, golds)

    # ===== MATH K=64 OOD overlay — load full samples, compute pass@k + maj@k =====
    import glob
    MATH_TEST_NUMERIC = ROOT / "v3" / "shared" / "data" / "math" / "test_numeric.jsonl"
    def _load_math_passk_majk(json_path):
        """maj@k via chain-permutation expectation (DSmath/Yue style)."""
        if json_path is None or not Path(json_path).exists():
            return None, None
        import random as _r
        md = json.load(open(json_path))
        samples = md["samples"]
        K_actual = md.get("config", {}).get("K", K)
        m_golds = [normalize(s.get("gold")) for s in samples]
        m_ans = [[normalize(a) for a in s["any_preds"]] for s in samples]
        # Recompute c locally so pass uses same criterion as maj
        m_c = [sum(1 for a in row if a is not None and a == g) for row, g in zip(m_ans, m_golds)]
        n_q = len(samples)
        rng = _r.Random(42)
        T = 200
        passk, majk = {}, {}
        for k in KS:
            if k > K_actual: continue
            passk[k] = sum(pass_at_k_unbiased(c, K_actual, k) for c in m_c) / n_q * 100
            maj_sum = 0.0
            for row, g in zip(m_ans, m_golds):
                if not row: continue
                hits = 0
                for _ in range(T):
                    shuf_k = rng.sample(row, min(k, len(row)))
                    valid = [a for a in shuf_k if a is not None]
                    if not valid: continue
                    if Counter(valid).most_common(1)[0][0] == g:
                        hits += 1
                maj_sum += hits / T
            majk[k] = maj_sum / n_q * 100
        return passk, majk
    # MATH-500-aug: base K=128 subset + DAPO K=64 fresh on same 500
    math_base_passk, math_base_majk = _load_math_passk_majk(BASE_MATH_K64)
    dapo_math_path = DAPO_MATH_K64
    math_dapo_passk, math_dapo_majk = _load_math_passk_majk(dapo_math_path)
    print(f"[MATH overlay] base loaded: {math_base_passk is not None} | dapo loaded: {math_dapo_passk is not None}")

    # ===== MATH per-question ABC data for L1.2 MATH panel =====
    math_base_c = math_base_mc = math_base_mm = math_sft_c = math_sft_mc = math_sft_mm = None
    m_ans_base_aligned = m_ans_sft_aligned = m_golds_aligned = None
    try:
        if Path(BASE_MATH_K64).exists() and dapo_math_path and Path(dapo_math_path).exists():
            mb = json.load(open(BASE_MATH_K64))
            md = json.load(open(dapo_math_path))
            # Align by question text — base and DAPO files have DIFFERENT q orders.
            base_by_q = {s["question"]: s for s in mb["samples"]}
            math_base_c, math_base_mc, math_base_mm = [], [], []
            math_sft_c, math_sft_mc, math_sft_mm = [], [], []
            m_ans_base_aligned, m_ans_sft_aligned, m_golds_aligned = [], [], []
            for d_sam in md["samples"]:
                q = d_sam["question"]
                if q not in base_by_q: continue
                b_sam = base_by_q[q]
                g = normalize(b_sam.get("gold"))
                # base side (K=128)
                b_ans = [normalize(a) for a in b_sam["any_preds"]]
                math_base_c.append(b_sam["any_correct_per_K"])
                valid_b = [a for a in b_ans if a]
                if valid_b:
                    ma, mc = Counter(valid_b).most_common(1)[0]
                    math_base_mc.append(ma == g); math_base_mm.append(mc / len(b_ans))
                else:
                    math_base_mc.append(False); math_base_mm.append(0.0)
                # dapo side (K=64)
                d_ans = [normalize(a) for a in d_sam["any_preds"]]
                math_sft_c.append(d_sam["any_correct_per_K"])
                valid_d = [a for a in d_ans if a]
                if valid_d:
                    ma, mc = Counter(valid_d).most_common(1)[0]
                    math_sft_mc.append(ma == g); math_sft_mm.append(mc / len(d_ans))
                else:
                    math_sft_mc.append(False); math_sft_mm.append(0.0)
                m_ans_base_aligned.append(b_ans)
                m_ans_sft_aligned.append(d_ans)
                m_golds_aligned.append(g)
            math_base_c = np.array(math_base_c); math_base_mc = np.array(math_base_mc); math_base_mm = np.array(math_base_mm)
            math_sft_c = np.array(math_sft_c); math_sft_mc = np.array(math_sft_mc); math_sft_mm = np.array(math_sft_mm)
            print(f"[MATH ABC] aligned by question, N={len(math_base_c)}")
    except Exception as e:
        print(f"[MATH ABC] skipped: {e}")

    bucket_idx = {"Easy": [], "Medium": [], "Hard": []}
    for i in range(n_q):
        b = labels.get(i, "?")
        if b in bucket_idx: bucket_idx[b].append(i)
    bucket_sizes = {b: len(v) for b, v in bucket_idx.items()}

    def to_b(p):
        if p >= 0.9: return 0
        if p <= 0.1: return 2
        return 1
    base_self = np.array([to_b(p) for p in base["cmass"]])
    sft_self = np.array([to_b(p) for p in sft["cmass"]])

    # Tokenize for L5 (slowest)
    print("[L5] tokenizing all responses...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    def collect_lens_steps(resps):
        lens, steps = [], []
        for q_resps in resps:
            enc = tok(q_resps, add_special_tokens=False, return_attention_mask=False)
            for r, ids in zip(q_resps, enc["input_ids"]):
                lens.append(len(ids))
                steps.append(len(STEP_RE.findall(r)))
        return np.array(lens), np.array(steps)
    base_len, base_step = collect_lens_steps(base_resps)
    sft_len, sft_step = collect_lens_steps(sft_resps)

    print("[render] combined figure...")
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(16, 40))
    # Layout (rows): 0=L1.1, 1=L1.2a, 2=L1.2b, 3=L6.1, 4=L6.2, 5=L5, 6=L8, 7=L9, 8=L10
    gs = fig.add_gridspec(9, 2, height_ratios=[1, 1.1, 1.1, 1.1, 1.2, 1, 1, 1, 1], hspace=0.85, wspace=0.40)
    axes = np.empty((9, 2), dtype=object)
    SKIP_CELLS = {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (5, 0), (5, 1), (6, 0), (6, 1), (7, 0), (7, 1), (8, 0), (8, 1)}
    for i in range(9):
        for j in range(2):
            if (i, j) in SKIP_CELLS:
                continue
            axes[i, j] = fig.add_subplot(gs[i, j])
    # Row 0: L1.1 full split
    sub_gs_l11 = gs[0, :].subgridspec(1, 2, wspace=0.30)
    ax_l11_gsm = fig.add_subplot(sub_gs_l11[0, 0])
    ax_l11_math = fig.add_subplot(sub_gs_l11[0, 1])
    # Row 1: L1.2a GSM8K full row
    sub_gs_l12_gsm = gs[1, :].subgridspec(1, 2, width_ratios=[1.2, 1], wspace=0.30)
    ax_l12_bar = fig.add_subplot(sub_gs_l12_gsm[0, 0])
    ax_l12_mat = fig.add_subplot(sub_gs_l12_gsm[0, 1])
    # Row 2: L1.2b MATH full row
    sub_gs_l12_math = gs[2, :].subgridspec(1, 2, width_ratios=[1.2, 1], wspace=0.30)
    ax_l12_math_bar = fig.add_subplot(sub_gs_l12_math[0, 0])
    ax_l12_math_mat = fig.add_subplot(sub_gs_l12_math[0, 1])
    ax_l8 = fig.add_subplot(gs[6, :])
    sub_gs_l9 = gs[7, :].subgridspec(1, 2, wspace=0.30)
    ax_l9_gsm = fig.add_subplot(sub_gs_l9[0, 0])
    ax_l9_math = fig.add_subplot(sub_gs_l9[0, 1])
    sub_gs_l10 = gs[8, :].subgridspec(1, 2, wspace=0.30)
    ax_l10_gsm = fig.add_subplot(sub_gs_l10[0, 0])
    ax_l10_math = fig.add_subplot(sub_gs_l10[0, 1])
    # ax_l31_bucket removed — L3.1 merged into L1.2
    sub_gs_l5 = gs[5, :].subgridspec(1, 2, wspace=0.30)
    ax_l13_hist = fig.add_subplot(sub_gs_l5[0, 0])
    ax_l13_bar = fig.add_subplot(sub_gs_l5[0, 1])

    # ============ L1.1 — pass@K + maj@K curves (GSM8K | MATH split) ============
    # Left: GSM8K (in-domain), Right: MATH numeric (OOD)
    ax = ax_l11_gsm
    ax.plot(KS, [base_passk[k] for k in KS], "-", color=COLOR_BASE, linewidth=2,
            label="base pass@K")
    ax.plot(KS, [base_majk[k] for k in KS], "--", color=COLOR_BASE, linewidth=1.6,
            alpha=0.7, label="base maj@K")
    ax.plot(KS, [sft_passk[k] for k in KS], "-", color=COLOR_SFT, linewidth=2,
            label=f"{TAG} pass@K")
    ax.plot(KS, [sft_majk[k] for k in KS], "--", color=COLOR_SFT, linewidth=1.6,
            alpha=0.85, label=f"{TAG} maj@K")
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("K"); ax.set_ylabel("accuracy (%)")
    ax.set_title(f"L1.1a — GSM8K (in-domain, n=1319)",
                 loc="left", fontsize=9, fontweight="semibold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    _y_gsm = ([base_passk[k] for k in KS] + [base_majk[k] for k in KS]
              + [sft_passk[k] for k in KS] + [sft_majk[k] for k in KS])
    ax.set_ylim(math.floor(min(_y_gsm) / 5) * 5, math.ceil(max(_y_gsm) / 5) * 5)

    ax = ax_l11_math
    if math_base_passk is not None:
        ax.plot(KS, [math_base_passk[k] for k in KS], "-", color=COLOR_BASE, linewidth=2,
                label="base pass@K")
        ax.plot(KS, [math_base_majk[k] for k in KS], "--", color=COLOR_BASE, linewidth=1.6,
                alpha=0.7, label="base maj@K")
        title_m = (f"base p@1={math_base_passk[1]:.2f} p@64={math_base_passk[64]:.2f} "
                   f"maj@64={math_base_majk[64]:.2f}")
    else:
        title_m = "base 待跑"
    if math_dapo_passk is not None:
        ax.plot(KS, [math_dapo_passk[k] for k in KS], "-", color=COLOR_SFT, linewidth=2,
                label=f"{TAG} pass@K")
        ax.plot(KS, [math_dapo_majk[k] for k in KS], "--", color=COLOR_SFT, linewidth=1.6,
                alpha=0.85, label=f"{TAG} maj@K")
        title_m += (f"\n{TAG} p@1={math_dapo_passk[1]:.2f} p@64={math_dapo_passk[64]:.2f} "
                    f"maj@64={math_dapo_majk[64]:.2f}")
    else:
        title_m += f"\n{TAG} 跑中(等)"
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("K"); ax.set_ylabel("accuracy (%)")
    ax.set_title(f"L1.1b — MATH-500-aug (OOD, n=500)",
                 loc="left", fontsize=9, fontweight="semibold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    _y_math = []
    if math_base_passk is not None:
        _y_math += [math_base_passk[k] for k in KS] + [math_base_majk[k] for k in KS]
    if math_dapo_passk is not None:
        _y_math += [math_dapo_passk[k] for k in KS] + [math_dapo_majk[k] for k in KS]
    if _y_math:
        ax.set_ylim(math.floor(min(_y_math) / 5) * 5, math.ceil(max(_y_math) / 5) * 5)

    # ============ L1.2 — 5-bucket bar + 5×5 transition (merged from old L3.1) for GSM8K + MATH ============
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from _paper_style_panels import plot_abc5_bar, plot_abc5_transition, plot_transition_10x10
    def _bk_idx(c, mc):
        if c == 0: return 2
        if mc: return 0
        return 1
    plot_abc5_bar(ax_l12_bar, base["c"], base["mode_correct"], base["mmass"],
                  sft["c"], sft["mode_correct"], sft["mmass"],
                  ckpt_label=TAG, dataset_name="GSM8K", title_prefix="L1.2a")
    plot_abc5_transition(ax_l12_mat, base["c"], base["mode_correct"], base["mmass"],
                         sft["c"], sft["mode_correct"], sft["mmass"],
                         ckpt_label=TAG, dataset_name="GSM8K", title_prefix="L1.2a.right")
    if math_base_c is not None and math_sft_c is not None:
        plot_abc5_bar(ax_l12_math_bar, math_base_c, math_base_mc, math_base_mm,
                      math_sft_c, math_sft_mc, math_sft_mm,
                      ckpt_label=TAG, dataset_name="MATH", title_prefix="L1.2b")
        plot_abc5_transition(ax_l12_math_mat, math_base_c, math_base_mc, math_base_mm,
                             math_sft_c, math_sft_mc, math_sft_mm,
                             ckpt_label=TAG, dataset_name="MATH", title_prefix="L1.2b.right")
    else:
        ax_l12_math_bar.text(0.5, 0.5, "MATH per-Q data unavailable", ha="center", va="center",
                              transform=ax_l12_math_bar.transAxes, fontsize=10, color="grey")
        ax_l12_math_bar.axis("off"); ax_l12_math_mat.axis("off")
    # L2 panels REMOVED per user request

    # L3.1 MERGED into L1.2 (5-bucket bar + 5×5 transition above) per user request

    # L3.2 removed per user request

    # L4 panels REMOVED per user request

    # ============ L5 — pass@1 mechanism (per-Q Δc) ============
    delta_c = sft["c"] - base["c"]
    sum_dc = int(delta_c.sum())
    pass1_diff_pp = sum_dc / (64 * n_q) * 100
    # L5.1: per-Q Δc histogram
    ax = ax_l13_hist
    bins_dc = np.arange(-65, 66, 2)
    ax.hist(delta_c, bins=bins_dc, color=COLOR_SFT, alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.0, alpha=0.6)
    ax.axvline(delta_c.mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"mean Δc = {delta_c.mean():+.2f}")
    n_pos = int((delta_c > 0).sum()); n_neg = int((delta_c < 0).sum()); n_zero = int((delta_c == 0).sum())
    ax.set_xlabel("Δc (trained − base, K=64)", labelpad=2)
    ax.set_ylabel("# questions", labelpad=2)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_title(f"L5.1 — per-Q Δc histogram   Σ Δc={sum_dc:+d} ({pass1_diff_pp:+.2f}pp pass@1)",
                 loc="left", fontsize=9, fontweight="semibold", pad=8)
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3, linestyle=":")

    # L5.2: Σ Δc by A/B/C transition bar
    ax = ax_l13_bar
    base_bk = np.array([_bk_idx(int(c), int(mc)) for c, mc in zip(base["c"], base["mode_correct"])])
    sft_bk = np.array([_bk_idx(int(c), int(mc)) for c, mc in zip(sft["c"], sft["mode_correct"])])
    cells_order = [(0, 0), (1, 0), (2, 1), (1, 1), (0, 1), (1, 2), (2, 2), (0, 2), (2, 0)]
    cell_labels = ["A→A", "B→A", "C→B", "B→B", "A→B", "B→C", "C→C", "A→C", "C→A"]
    sum_per = []; n_per = []
    for (i, j) in cells_order:
        mask = (base_bk == i) & (sft_bk == j)
        n_per.append(int(mask.sum()))
        sum_per.append(int(delta_c[mask].sum()) if mask.any() else 0)
    cols = ["#16a34a" if v > 0 else "#dc2626" if v < 0 else "#9ca3af" for v in sum_per]
    x = np.arange(len(cell_labels))
    ax.bar(x, sum_per, color=cols, edgecolor="white", linewidth=1)
    _ymin = min(sum_per + [0]); _ymax = max(sum_per + [0])
    _yr = max(_ymax - _ymin, 1)
    _off = _yr * 0.05
    for i, (v, n) in enumerate(zip(sum_per, n_per)):
        if v > 0:
            ax.text(i, v - _off * 0.5, f"{v:+d}\nn={n}", ha="center", va="top",
                    fontsize=7, fontweight="semibold", color="white" if abs(v) > _yr * 0.15 else "black")
        elif v < 0:
            ax.text(i, v + _off * 0.5, f"{v:+d}\nn={n}", ha="center", va="bottom",
                    fontsize=7, fontweight="semibold", color="white" if abs(v) > _yr * 0.15 else "black")
        else:
            ax.text(i, _off * 2, f"0\nn={n}", ha="center", va="bottom", fontsize=7, color="#666")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(cell_labels, fontsize=9)
    ax.set_ylabel("Σ Δc per transition", labelpad=2)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_ylim(_ymin - _yr * 0.15, _ymax + _yr * 0.20)
    ax.set_title(f"L5.2 — Σ Δc by A/B/C transition",
                 loc="left", fontsize=9, fontweight="semibold", pad=10)
    ax.grid(axis="y", alpha=0.3, linestyle=":")


    # ============ L6 — paper-style panels (paired hist + 10x10 transition) ============
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from _paper_style_panels import plot_paired_hist, plot_transition_10x10

    rates_base_gsm = base["c"] / K
    rates_sft_gsm = sft["c"] / K

    rates_base_math = rates_sft_math = None
    K_m = K_base_math = None
    if BASE_MATH_K64.exists() and DAPO_MATH_K64.exists():
        md_b = json.load(open(BASE_MATH_K64)); md_r = json.load(open(DAPO_MATH_K64))
        K_base_math = md_b.get("config", {}).get("K", 128)
        K_m = md_r.get("config", {}).get("K", K)
        base_rate_by_q = {s["question"]: s["any_correct_per_K"] / K_base_math for s in md_b["samples"]}
        rb, rs = [], []
        for s in md_r["samples"]:
            q = s["question"]
            if q in base_rate_by_q:
                rb.append(base_rate_by_q[q])
                rs.append(s["any_correct_per_K"] / K_m)
        rates_base_math = np.array(rb); rates_sft_math = np.array(rs)

    ax = axes[3, 0]
    plot_paired_hist(ax, rates_base_gsm, rates_sft_gsm,
                     title=f"L6.1a — GSM8K per-q pass-rate hist (paper §4.1 style)\n"
                           f"base mean={rates_base_gsm.mean():.3f} | DAPO ck-15 mean={rates_sft_gsm.mean():.3f}",
                     labels=(f"base K={K}", f"DAPO ck-15 K={K}"))
    ax = axes[3, 1]
    if rates_base_math is not None and rates_sft_math is not None:
        plot_paired_hist(ax, rates_base_math, rates_sft_math,
                         title=f"L6.1b — MATH per-q pass-rate hist\n"
                               f"base mean={rates_base_math.mean():.3f} | DAPO ck-15 mean={rates_sft_math.mean():.3f}",
                         labels=(f"base K={K_base_math}", f"DAPO ck-15 K={K_m}"))

    ax = axes[4, 0]
    plot_transition_10x10(ax, rates_base_gsm, rates_sft_gsm,
                          title="L6.2a — GSM8K 10×10 transition (base→DAPO, log color)")
    ax = axes[4, 1]
    if rates_base_math is not None and rates_sft_math is not None:
        plot_transition_10x10(ax, rates_base_math, rates_sft_math,
                              title="L6.2b — MATH 10×10 transition (base→DAPO, log color)")

    # L7 removed per user request
    from _paper_style_panels import plot_l8_delta_mass_base_anchor
    plot_l8_delta_mass_base_anchor(ax_l8, base_ans, sft_ans, golds,
                                    ckpt_label=TAG, title_prefix="L8")
    from _paper_style_panels import plot_l9_base_mode_correct, plot_l10_base_mode_wrong
    plot_l9_base_mode_correct(ax_l9_gsm, base_ans, sft_ans, golds, ckpt_label=TAG, title_prefix="L9a [GSM8K]")
    plot_l10_base_mode_wrong(ax_l10_gsm, base_ans, sft_ans, golds, ckpt_label=TAG, title_prefix="L10a [GSM8K]")
    if m_ans_sft_aligned is not None:
        plot_l9_base_mode_correct(ax_l9_math, m_ans_base_aligned, m_ans_sft_aligned, m_golds_aligned,
                                   ckpt_label=TAG, title_prefix="L9b [MATH]")
        plot_l10_base_mode_wrong(ax_l10_math, m_ans_base_aligned, m_ans_sft_aligned, m_golds_aligned,
                                  ckpt_label=TAG, title_prefix="L10b [MATH]")
    else:
        ax_l9_math.text(0.5, 0.5, 'MATH data unavailable', ha='center', va='center', transform=ax_l9_math.transAxes)
        ax_l10_math.text(0.5, 0.5, 'MATH data unavailable', ha='center', va='center', transform=ax_l10_math.transAxes)

    fig.suptitle("DAPO ck-15 (R15 best, dev-select) vs base IT — L1-L10 combined (L3.2/L7 removed)",
                 fontsize=12, fontweight="semibold", y=1.0)
    plt.tight_layout()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"\nsaved: {OUT_FILE}")


if __name__ == "__main__":
    main()

"""lr=5e-4 step=130 vs base IT — single combined diagnostic figure.

Includes EVERY panel referenced by L1-L5 analysis for step 130 vs base.
- Panels NEW to this figure (not in any standalone plot): L1.2 ABC, L2.2 maj@K bucket, L2.2 migration heatmap.
- Other panels are step 130 / base snapshots from existing plots:
    L1.1 from passk_majk_curves / passk_majk_by_ckpt step 130
    L2.1 from sft_per_bucket_trajectory middle col
    L3.1 from lr5e-4_difficulty_grid step 130 + base distribution overlay
    L3.1 from lr5e-4_mode_correctness_scatter base + step 130 panels
    L3.2 from lr5e-4_wrong_concentration left panel (selected ckpts overlay)
    L4.1 from length_e1style/lr5e-4_step130_length_e1style A.1
    L4.2 from length_e1style/lr5e-4_step130_length_e1style A.5

5 rows × 2 cols = 10 panels.
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
_font_path = Path("/mnt/c/Windows/Fonts/msyh.ttc")
if _font_path.is_file():
    _fm.fontManager.addfont(_font_path)
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "v3" / "E2_sft" / "outputs" / "test_eval_k64"
BASE_K64 = ROOT / "v3" / "E1_baseline" / "outputs" / "pass_at_k_20260427_222954" / "base_gemma-2-2b-it_k64.json"
LABELS = ROOT / "v3" / "shared" / "data" / "gsm8k" / "test_difficulty_labels.jsonl"
TEST_PC = ROOT / "v3" / "shared" / "data" / "gsm8k" / "test_pc.jsonl"
BASE_MODEL = ROOT / "models" / "gemma-2-2b-it"

# CLI args (default to lr=5e-4 step 130 for backward compatibility)
_ap = argparse.ArgumentParser()
_ap.add_argument("--lr", default="5e-4")
_ap.add_argument("--step", type=int, default=130)
_args = _ap.parse_args()
LR = _args.lr
STEP = _args.step
OUT_FILE = ROOT / "v3" / "E2_sft" / "outputs" / f"lr{LR}_step{STEP}_combined.png"

K = 64
KS = [1, 2, 4, 8, 16, 32, 64]
COLOR_BASE = "black"
COLOR_SFT = "#16a34a"
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

    print("[load] base K=64 + SFT step 130...")
    bd = json.load(open(BASE_K64))
    base_ans = [[normalize(a) for a in s["any_preds"]] for s in bd["samples"]]
    base_resps = [s["responses"] for s in bd["samples"]]
    sd = json.load(open(EVAL_DIR / f"sft_lr{LR}_r64_checkpoint-{STEP}.json"))
    sft_ans = [[normalize(a) for a in row] for row in sd["per_sample_answers"]]
    sft_resps = sd["per_sample_responses"]

    n_q = len(golds)
    # Use eval-script pre-computed correct count (L2-aligned, math_equal strict compare)
    base_pre_c = [s["any_correct_per_K"] for s in bd["samples"]]
    sft_pre_c = sd["per_question_correct_count"]
    base = per_q_metrics(base_ans, golds, pre_c=base_pre_c)
    sft = per_q_metrics(sft_ans, golds, pre_c=sft_pre_c)
    base_passk, base_majk = passk_majk_curves(base["c"], base_ans, golds)
    sft_passk, sft_majk = passk_majk_curves(sft["c"], sft_ans, golds)

    # ====== MATH numeric K=64 OOD overlay — load + compute pass@k + maj@k ======
    import glob as _glob
    MATH_TEST_NUMERIC = ROOT / "v3" / "shared" / "data" / "math" / "test_numeric.jsonl"
    def _load_math_passk_majk(json_path):
        if not json_path or not Path(json_path).exists(): return None, None
        with open(json_path) as f: jd = json.load(f)
        math_golds = []
        with open(MATH_TEST_NUMERIC) as f:
            for line in f: math_golds.append(json.loads(line)["gold"])
        N = len(math_golds)
        if "samples" not in jd: return None, None
        # sample row → list of K responses; extract first-int answer
        import re as _re
        def _extract(text):
            m = _re.search(r"\\boxed\{([^}]+)\}", text)
            if m:
                t = m.group(1).strip()
                mm = _re.search(r"-?\d+\.?\d*", t)
                if mm: return mm.group(0)
            return None
        any_preds = []
        c_arr = []
        for i, s in enumerate(jd["samples"]):
            preds = [_extract(r) for r in s["responses"]]
            any_preds.append(preds)
            c_arr.append(sum(1 for p in preds if p is not None and p == str(math_golds[i])))
        passk = {k: 100.0 * sum(pass_at_k_unbiased(ci, len(any_preds[0]), k) for ci in c_arr) / len(c_arr) for k in KS}
        majk = {}
        for k in KS:
            n_correct = 0
            for i, row in enumerate(any_preds):
                g = str(math_golds[i])
                first_k = [a for a in row[:k] if a is not None]
                if not first_k: continue
                if Counter(first_k).most_common(1)[0][0] == g:
                    n_correct += 1
            majk[k] = n_correct / N * 100
        return passk, majk
    # MATH-500-aug (500 q): base K=128 subset + SFT K=128 subset (filtered from 2927)
    base_math_k128_path = ROOT / "v3" / "E5_grpo" / "outputs" / "k128_merged_math500_aug_slice" / "base_k128_math500_aug_verbose.json"
    base_math_cands = []
    base_math_path = str(base_math_k128_path) if base_math_k128_path.exists() else None
    # _load_math_passk_majk uses fixed K=64 in inner; need K-dynamic version for K=128
    def _load_math_passk_majk_dyn(json_path):
        if not json_path or not Path(json_path).exists(): return None, None, None
        with open(json_path) as f: jd = json.load(f)
        if "samples" not in jd: return None, None, None
        K_file = jd["config"]["K"]
        samples = jd["samples"]
        N = len(samples)
        m_golds = [normalize(s["gold"]) for s in samples]
        any_preds = [[normalize(a) for a in s["any_preds"]] for s in samples]
        c_arr = [s["any_correct_per_K"] for s in samples]
        KS_actual = [k for k in [1, 2, 4, 8, 16, 32, 64, 128] if k <= K_file]
        passk, majk = {}, {}
        for k in KS_actual:
            passk[k] = 100.0 * sum(pass_at_k_unbiased(ci, K_file, k) for ci in c_arr) / N
            n_correct = 0
            for row, g in zip(any_preds, m_golds):
                first_k = [a for a in row[:k] if a is not None]
                if not first_k: continue
                if Counter(first_k).most_common(1)[0][0] == g:
                    n_correct += 1
            majk[k] = n_correct / N * 100
        return passk, majk, KS_actual
    math_base_passk, math_base_majk, math_base_KS = _load_math_passk_majk_dyn(base_math_path)
    # SFT MATH K=128 merged (verbose orig + slim new with seed=12345)
    sft_k128_path = ROOT / "v3" / "E5_grpo" / "outputs" / "k128_merged_math500_aug_slice" / "sft_lr5e-4_ck130_k128_math500_aug.json"
    def _load_math_passk_majk_k128(json_path):
        """Load slim K=128 file: per_q[i] = {gold, samples: [{c, a, L}]}"""
        if not json_path or not Path(json_path).exists(): return None, None, None
        d = json.load(open(json_path))
        per_q = d["per_q"]
        K_actual = len(per_q[0]["samples"])
        KS_actual = [k for k in [1, 2, 4, 8, 16, 32, 64, 128] if k <= K_actual]
        passk, majk = {}, {}
        for k in KS_actual:
            passk[k] = 100.0 * sum(pass_at_k_unbiased(sum(s["c"] for s in q["samples"]), K_actual, k)
                                   for q in per_q) / len(per_q)
            n_correct = 0
            for q in per_q:
                preds = [s["a"] for s in q["samples"][:k] if s["a"]]
                if not preds: continue
                if Counter(preds).most_common(1)[0][0] == str(q["gold"]).strip():
                    n_correct += 1
            majk[k] = n_correct / len(per_q) * 100
        return passk, majk, KS_actual
    math_sft_passk, math_sft_majk, math_sft_KS = _load_math_passk_majk_k128(sft_k128_path)
    print(f"[MATH overlay] base loaded: {math_base_passk is not None} | sft K=128 loaded: {math_sft_passk is not None}")

    # ===== MATH per-question ABC data for L1.2 MATH panel =====
    math_base_c = math_base_mc = math_base_mm = math_sft_c = math_sft_mc = math_sft_mm = None
    m_ans_base_aligned = m_ans_sft_aligned = m_golds_aligned = None
    try:
        if base_math_path and Path(base_math_path).exists():
            mb = json.load(open(base_math_path))
            m_golds_base = [normalize(s.get("gold")) for s in mb["samples"]]
            m_ans_base = [[normalize(a) for a in s["any_preds"]] for s in mb["samples"]]
            math_base_c = np.array([s["any_correct_per_K"] for s in mb["samples"]])
            math_base_mc = []; math_base_mm = []
            for row, g in zip(m_ans_base, m_golds_base):
                valid = [a for a in row if a]
                if not valid: math_base_mc.append(False); math_base_mm.append(0.0); continue
                mode_a, mode_c = Counter(valid).most_common(1)[0]
                math_base_mc.append(mode_a == g); math_base_mm.append(mode_c / len(row))
            math_base_mc = np.array(math_base_mc); math_base_mm = np.array(math_base_mm)
        if sft_k128_path.exists():
            md = json.load(open(sft_k128_path))
            per_q = md["per_q"]
            math_sft_c = np.array([sum(s["c"] for s in q["samples"]) for q in per_q])
            math_sft_mc = []; math_sft_mm = []
            for q in per_q:
                preds = [normalize(s["a"]) for s in q["samples"] if s["a"]]
                if not preds: math_sft_mc.append(False); math_sft_mm.append(0.0); continue
                mode_a, mode_c = Counter(preds).most_common(1)[0]
                math_sft_mc.append(mode_a == normalize(q["gold"]))
                math_sft_mm.append(mode_c / len(q["samples"]))
            math_sft_mc = np.array(math_sft_mc); math_sft_mm = np.array(math_sft_mm)
            N = min(len(math_base_c), len(math_sft_c))
            math_base_c = math_base_c[:N]; math_base_mc = math_base_mc[:N]; math_base_mm = math_base_mm[:N]
            math_sft_c = math_sft_c[:N]; math_sft_mc = math_sft_mc[:N]; math_sft_mm = math_sft_mm[:N]
            m_ans_base_aligned = m_ans_base[:N]
            m_ans_sft_aligned = [[normalize(s["a"]) for s in q["samples"]] for q in per_q[:N]]
            m_golds_aligned = m_golds_base[:N]
            print(f"[MATH ABC] aligned N={N}")
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
    fig = plt.figure(figsize=(15, 40))
    # Layout: row 0=L1.1, 1=L1.2a, 2=L1.2b, 3=L6.1, 4=L6.2, 5=L5, 6=L8, 7=L9, 8=L10
    gs = fig.add_gridspec(9, 2, height_ratios=[1, 1.1, 1.1, 1.1, 1.2, 1, 1, 1, 1], hspace=0.85)
    axes = np.empty((9, 2), dtype=object)
    SKIP_CELLS = {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (5, 0), (5, 1), (6, 0), (6, 1), (7, 0), (7, 1), (8, 0), (8, 1)}
    for i in range(9):
        for j in range(2):
            if (i, j) in SKIP_CELLS:
                continue
            axes[i, j] = fig.add_subplot(gs[i, j])
    sub_gs_l11 = gs[0, :].subgridspec(1, 2, wspace=0.30)
    ax_l11_gsm = fig.add_subplot(sub_gs_l11[0, 0])
    ax_l11_math = fig.add_subplot(sub_gs_l11[0, 1])
    sub_gs_l12_gsm = gs[1, :].subgridspec(1, 2, width_ratios=[1.2, 1], wspace=0.30)
    ax_l12_bar = fig.add_subplot(sub_gs_l12_gsm[0, 0])
    ax_l12_mat = fig.add_subplot(sub_gs_l12_gsm[0, 1])
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
    # SFT GSM8K K=128 (merged: orig K=64 + new seed=12345 K=64) overrides original sft_passk/majk
    sft_gsm8k_k128_path = ROOT / "v3" / "E5_grpo" / "outputs" / "k128_merged" / "sft_lr5e-4_ck130_k128_gsm8k.json"
    sft_gsm_KS = list(KS)
    if sft_gsm8k_k128_path.exists():
        d128 = json.load(open(sft_gsm8k_k128_path))
        per_q128 = d128["per_q"]
        K128 = len(per_q128[0]["samples"])
        sft_gsm_KS = [k for k in [1, 2, 4, 8, 16, 32, 64, 128] if k <= K128]
        sft_passk = {}
        sft_majk = {}
        for k in sft_gsm_KS:
            sft_passk[k] = 100.0 * sum(pass_at_k_unbiased(sum(s["c"] for s in q["samples"]), K128, k)
                                       for q in per_q128) / len(per_q128)
            n_correct = 0
            for q in per_q128:
                preds = [str(s["a"]) for s in q["samples"][:k] if s["a"]]
                if not preds: continue
                if Counter(preds).most_common(1)[0][0] == str(q["gold"]).strip():
                    n_correct += 1
            sft_majk[k] = n_correct / len(per_q128) * 100
        print(f"[L1.1a] SFT K={K128} loaded from merged file")

    # Upgrade base GSM8K to K=128 if merged verbose file exists
    base_gsm8k_k128_path = ROOT / "v3" / "E5_grpo" / "outputs" / "k128_merged" / "base_k128_gsm8k_verbose.json"
    base_gsm_KS = list(KS)
    if base_gsm8k_k128_path.exists():
        bd128 = json.load(open(base_gsm8k_k128_path))
        K_b128 = bd128["config"]["K"]
        n_b128 = len(bd128["samples"])
        base_gsm_KS = [k for k in [1, 2, 4, 8, 16, 32, 64, 128] if k <= K_b128]
        base_b128_golds = [normalize(s.get("gold")) for s in bd128["samples"]]
        base_b128_ans = [[normalize(a) for a in s["any_preds"]] for s in bd128["samples"]]
        base_passk = {}
        base_majk = {}
        for k in base_gsm_KS:
            base_passk[k] = 100.0 * sum(pass_at_k_unbiased(s["any_correct_per_K"], K_b128, k)
                                        for s in bd128["samples"]) / n_b128
            n_correct = 0
            for row, g in zip(base_b128_ans, base_b128_golds):
                first_k = [a for a in row[:k] if a is not None]
                if not first_k: continue
                if Counter(first_k).most_common(1)[0][0] == g:
                    n_correct += 1
            base_majk[k] = n_correct / n_b128 * 100
        print(f"[L1.1a] base K={K_b128} loaded from K=128 verbose merged file")

    ax = ax_l11_gsm
    ax.plot(base_gsm_KS, [base_passk[k] for k in base_gsm_KS], "-", color=COLOR_BASE, linewidth=2,
            label=f"base pass@K (K={base_gsm_KS[-1]})")
    ax.plot(base_gsm_KS, [base_majk[k] for k in base_gsm_KS], "--", color=COLOR_BASE, linewidth=1.6,
            alpha=0.7, label="base maj@K")
    ax.plot(sft_gsm_KS, [sft_passk[k] for k in sft_gsm_KS], "-", color=COLOR_SFT, linewidth=2,
            label=f"SFT pass@K (K={sft_gsm_KS[-1]})")
    ax.plot(sft_gsm_KS, [sft_majk[k] for k in sft_gsm_KS], "--", color=COLOR_SFT, linewidth=1.6,
            alpha=0.85, label="SFT maj@K")
    ax.set_xscale("log", base=2)
    KS_union = sorted(set(base_gsm_KS) | set(sft_gsm_KS))
    ax.set_xticks(KS_union); ax.set_xticklabels([str(k) for k in KS_union])
    ax.set_xlabel("K"); ax.set_ylabel("accuracy (%)")
    k_top = sft_gsm_KS[-1]
    k_bb = base_gsm_KS[-1]
    ax.set_title(f"L1.1a — GSM8K (in-domain, n=1319)",
                 loc="left", fontsize=9, fontweight="semibold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    _y_gsm = ([base_passk[k] for k in base_gsm_KS] + [base_majk[k] for k in base_gsm_KS]
              + [sft_passk[k] for k in sft_gsm_KS] + [sft_majk[k] for k in sft_gsm_KS])
    import math as _math
    ax.set_ylim(_math.floor(min(_y_gsm) / 5) * 5, _math.ceil(max(_y_gsm) / 5) * 5)

    # L1.1b — MATH numeric (OOD)
    ax = ax_l11_math
    title_m = ""
    if math_base_passk is not None:
        kb = math_base_KS
        ax.plot(kb, [math_base_passk[k] for k in kb], "-", color=COLOR_BASE, linewidth=2,
                label=f"base pass@K (K={kb[-1]})")
        ax.plot(kb, [math_base_majk[k] for k in kb], "--", color=COLOR_BASE, linewidth=1.6,
                alpha=0.7, label="base maj@K")
        title_m = (f"base K={kb[-1]}: p@1={math_base_passk[1]:.2f} p@{kb[-1]}={math_base_passk[kb[-1]]:.2f} "
                   f"maj@{kb[-1]}={math_base_majk[kb[-1]]:.2f}")
    else:
        title_m = "base pending"
    if math_sft_passk is not None:
        ks_sft = math_sft_KS
        ax.plot(ks_sft, [math_sft_passk[k] for k in ks_sft], "-", color=COLOR_SFT, linewidth=2,
                label=f"SFT pass@K (K={ks_sft[-1]})")
        ax.plot(ks_sft, [math_sft_majk[k] for k in ks_sft], "--", color=COLOR_SFT, linewidth=1.6,
                alpha=0.85, label="SFT maj@K")
        k_top = ks_sft[-1]
        title_m += (f" | SFT K={k_top}: p@1={math_sft_passk[1]:.2f} p@{k_top}={math_sft_passk[k_top]:.2f} "
                    f"maj@{k_top}={math_sft_majk[k_top]:.2f}")
    else:
        title_m += " | SFT pending"
    ax.set_xscale("log", base=2)
    KS_union = sorted(set(KS) | set(math_sft_KS or []) | set(math_base_KS or []))
    ax.set_xticks(KS_union); ax.set_xticklabels([str(k) for k in KS_union])
    ax.set_xlabel("K"); ax.set_ylabel("accuracy (%)")
    ax.set_title(f"L1.1b — MATH-500-aug (OOD, n=500)",
                 loc="left", fontsize=9, fontweight="semibold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    _y_math = []
    if math_base_passk is not None:
        _y_math += [math_base_passk[k] for k in math_base_KS] + [math_base_majk[k] for k in math_base_KS]
    if math_sft_passk is not None:
        _y_math += [math_sft_passk[k] for k in math_sft_KS] + [math_sft_majk[k] for k in math_sft_KS]
    if _y_math:
        ax.set_ylim(_math.floor(min(_y_math) / 5) * 5, _math.ceil(max(_y_math) / 5) * 5)

    # ============ L1.2 — 5-bucket bar + 5×5 transition (merged from old L3.1) for GSM8K + MATH ============
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from _paper_style_panels import plot_abc5_bar, plot_abc5_transition, plot_transition_10x10
    SFT_TAG = f"SFT step {STEP}"
    def _bk_idx(c, mc):
        if c == 0: return 2
        if mc: return 0
        return 1
    plot_abc5_bar(ax_l12_bar, base["c"], base["mode_correct"], base["mmass"],
                  sft["c"], sft["mode_correct"], sft["mmass"],
                  ckpt_label=SFT_TAG, dataset_name="GSM8K", title_prefix="L1.2a")
    plot_abc5_transition(ax_l12_mat, base["c"], base["mode_correct"], base["mmass"],
                         sft["c"], sft["mode_correct"], sft["mmass"],
                         ckpt_label=SFT_TAG, dataset_name="GSM8K", title_prefix="L1.2a.right")
    if math_base_c is not None and math_sft_c is not None:
        plot_abc5_bar(ax_l12_math_bar, math_base_c, math_base_mc, math_base_mm,
                      math_sft_c, math_sft_mc, math_sft_mm,
                      ckpt_label=SFT_TAG, dataset_name="MATH", title_prefix="L1.2b")
        plot_abc5_transition(ax_l12_math_mat, math_base_c, math_base_mc, math_base_mm,
                             math_sft_c, math_sft_mc, math_sft_mm,
                             ckpt_label=SFT_TAG, dataset_name="MATH", title_prefix="L1.2b.right")
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

    # GSM8K: SFT K=128 if available, else K=64
    rates_base_gsm = base["c"] / K
    if sft_gsm8k_k128_path.exists():
        d128 = json.load(open(sft_gsm8k_k128_path))
        K_g_sft = len(d128["per_q"][0]["samples"])
        rates_sft_gsm = np.array([sum(s["c"] for s in q["samples"]) / K_g_sft for q in d128["per_q"]])
    else:
        K_g_sft = K
        rates_sft_gsm = sft["c"] / K

    # MATH — prefer K=128 verbose base
    rates_base_math = rates_sft_math = None
    K_m_sft = None
    K_base_math_actual = K
    if base_math_path and Path(base_math_path).exists():
        md_b = json.load(open(base_math_path))
        K_base_math_actual = md_b["config"]["K"]
        rates_base_math = np.array([s["any_correct_per_K"] / K_base_math_actual for s in md_b["samples"]])
    if sft_k128_path.exists():
        md_r = json.load(open(sft_k128_path))
        K_m_sft = len(md_r["per_q"][0]["samples"])
        rates_sft_math = np.array([sum(s["c"] for s in q["samples"]) / K_m_sft for q in md_r["per_q"]])

    ax = axes[3, 0]
    plot_paired_hist(ax, rates_base_gsm, rates_sft_gsm,
                     title=f"L6.1a — GSM8K per-q pass-rate hist (paper §4.1 style)\n"
                           f"base mean={rates_base_gsm.mean():.3f} | SFT step{STEP} mean={rates_sft_gsm.mean():.3f}",
                     labels=(f"base K={K}", f"SFT step{STEP} K={K_g_sft}"))
    ax = axes[3, 1]
    if rates_base_math is not None and rates_sft_math is not None:
        plot_paired_hist(ax, rates_base_math, rates_sft_math,
                         title=f"L6.1b — MATH per-q pass-rate hist\n"
                               f"base mean={rates_base_math.mean():.3f} | SFT step{STEP} mean={rates_sft_math.mean():.3f}",
                         labels=(f"base K={K_base_math_actual}", f"SFT step{STEP} K={K_m_sft}"))

    ax = axes[4, 0]
    plot_transition_10x10(ax, rates_base_gsm, rates_sft_gsm,
                          title=f"L6.2a — GSM8K 10×10 transition (base→SFT step{STEP}, log color)")
    ax = axes[4, 1]
    if rates_base_math is not None and rates_sft_math is not None:
        plot_transition_10x10(ax, rates_base_math, rates_sft_math,
                              title=f"L6.2b — MATH 10×10 transition (base→SFT step{STEP}, log color)")

    # L7 removed per user request
    from _paper_style_panels import plot_l8_delta_mass_base_anchor
    plot_l8_delta_mass_base_anchor(ax_l8, base_ans, sft_ans, golds,
                                    ckpt_label=f"SFT ck-{STEP}", title_prefix="L8")
    from _paper_style_panels import plot_l9_base_mode_correct, plot_l10_base_mode_wrong
    plot_l9_base_mode_correct(ax_l9_gsm, base_ans, sft_ans, golds, ckpt_label=f"SFT ck-{STEP}", title_prefix="L9a [GSM8K]")
    plot_l10_base_mode_wrong(ax_l10_gsm, base_ans, sft_ans, golds, ckpt_label=f"SFT ck-{STEP}", title_prefix="L10a [GSM8K]")
    if m_ans_sft_aligned is not None:
        plot_l9_base_mode_correct(ax_l9_math, m_ans_base_aligned, m_ans_sft_aligned, m_golds_aligned,
                                   ckpt_label=f"SFT ck-{STEP}", title_prefix="L9b [MATH]")
        plot_l10_base_mode_wrong(ax_l10_math, m_ans_base_aligned, m_ans_sft_aligned, m_golds_aligned,
                                  ckpt_label=f"SFT ck-{STEP}", title_prefix="L10b [MATH]")
    else:
        ax_l9_math.text(0.5, 0.5, 'MATH data unavailable', ha='center', va='center', transform=ax_l9_math.transAxes)
        ax_l10_math.text(0.5, 0.5, 'MATH data unavailable', ha='center', va='center', transform=ax_l10_math.transAxes)

    fig.suptitle(f"lr={LR} r=64 step={STEP} (best ckpt) vs base IT — L1-L10 combined (L3.2/L7 removed)",
                 fontsize=12, fontweight="semibold", y=1.0)
    plt.tight_layout()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"\nsaved: {OUT_FILE}")


if __name__ == "__main__":
    main()

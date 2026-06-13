"""Figure-2: pass@K + maj@K curves on D_test (K=64), 3 methods.

base IT       : v3/E1_baseline/outputs/pass_at_k_20260427_222954/base_gemma-2-2b-it_k64_curves.json
SFT collapsed : v3/E2_sft/outputs/test_eval_k64/sft_lr1e-4_r64_checkpoint-186.json (1 epoch end)
DAPO ck-15    : v3/E5_grpo/outputs/k64_dapo_ck15/r15_dapo_checkpoint-15_k64.json

Sampling      : T=0.7, top_p=0.95, n=64
"""
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
BASE_CURVES = ROOT / "v3" / "E1_baseline" / "outputs" / "pass_at_k_20260427_222954" / "base_gemma-2-2b-it_k64_curves.json"
SFT_K64 = ROOT / "v3" / "E2_sft" / "outputs" / "test_eval_k64" / "sft_lr1e-4_r64_checkpoint-186.json"
DAPO_K64 = ROOT / "v3" / "E5_grpo" / "outputs" / "k64_dapo_ck15" / "r15_dapo_checkpoint-15_k64.json"
SFT_TEST_FILE = ROOT / "v3" / "shared" / "data" / "gsm8k" / "test_pc.jsonl"
OUT_FILE = ROOT / "v3" / "E5_grpo" / "outputs" / "figure2_three_methods_passk_majk.png"

KS = [1, 2, 4, 8, 16, 32, 64]


def normalize(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("$", "").replace(" ", "")
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return s
        if v == int(v):
            return str(int(v))
        return str(v)
    except (ValueError, TypeError, OverflowError):
        return s


def gold_from_completion(completion):
    txt = completion[0]["content"] if isinstance(completion, list) else completion
    if "\\boxed{" in txt:
        end = txt.rfind("}")
        start = txt.rfind("\\boxed{") + len("\\boxed{")
        return txt[start:end].strip()
    return None


def pass_at_k_unbiased(c, n, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def compute_curves_from_sft(sft_jf, test_file):
    j = json.load(open(sft_jf))
    per_q_ans = j["per_sample_answers"]
    # Load gold answers
    golds_norm = []
    with open(test_file) as f:
        for line in f:
            ex = json.loads(line)
            golds_norm.append(normalize(gold_from_completion(ex["completion"])))
    assert len(per_q_ans) == len(golds_norm), f"{len(per_q_ans)} vs {len(golds_norm)}"
    per_q_norm = [[normalize(a) for a in row] for row in per_q_ans]
    per_q_correct = [sum(1 for a in row if a == g) for row, g in zip(per_q_norm, golds_norm)]
    K_max = len(per_q_norm[0])
    passk = {}
    majk = {}
    for k in KS:
        if k > K_max:
            continue
        passk[k] = sum(pass_at_k_unbiased(c, K_max, k) for c in per_q_correct) / len(per_q_correct) * 100
        n_correct = 0
        for row, g in zip(per_q_norm, golds_norm):
            first_k = [a for a in row[:k] if a is not None]
            if not first_k:
                continue
            top = Counter(first_k).most_common(1)[0][0]
            if top == g:
                n_correct += 1
        majk[k] = n_correct / len(per_q_norm) * 100
    return passk, majk


def main():
    # base IT (k=1..64 already computed)
    base = json.load(open(BASE_CURVES))
    base_passk = {int(k): v * 100 for k, v in base["pass_at_k_numeric"].items()}
    base_majk = {int(k): v * 100 for k, v in base["maj_at_k_numeric"].items()}

    # SFT collapsed (recompute from per_sample_answers)
    sft_passk, sft_majk = compute_curves_from_sft(SFT_K64, SFT_TEST_FILE)

    # DAPO ck-15 (pass@k from metrics; maj@K recompute for all K from samples.any_preds)
    dapo_j = json.load(open(DAPO_K64))
    m = dapo_j["metrics"]
    dapo_passk = {k: m[f"pass_at_{k}_numeric"] * 100 for k in KS}
    dapo_majk = {}
    n_samples = len(dapo_j["samples"])
    for k in KS:
        n_correct = 0
        for s in dapo_j["samples"]:
            preds = s["any_preds"][:k]
            preds_norm = [normalize(p) for p in preds if p is not None]
            if not preds_norm:
                continue
            top = Counter(preds_norm).most_common(1)[0][0]
            if top == normalize(s["gold"]):
                n_correct += 1
        dapo_majk[k] = n_correct / n_samples * 100

    # Plot
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6))

    # base IT (black)
    ks = sorted(base_passk.keys())
    ax.plot(ks, [base_passk[k] for k in ks], "-",
            color="black", linewidth=1.8, label="base IT  pass@K", zorder=3)
    ax.plot(ks, [base_majk[k] for k in ks], "--",
            color="black", linewidth=1.4, alpha=0.65, label="base IT  maj@K", zorder=3)

    # SFT collapsed (blue)
    ax.plot(KS, [sft_passk[k] for k in KS], "-",
            color="#2563eb", linewidth=1.8,
            label="SFT (1 epoch end, lr=1e-4)  pass@K", zorder=3)
    ax.plot(KS, [sft_majk[k] for k in KS], "--",
            color="#2563eb", linewidth=1.4, alpha=0.65, label="SFT  maj@K", zorder=3)

    # DAPO (red)
    ax.plot(KS, [dapo_passk[k] for k in KS], "-",
            color="#dc2626", linewidth=1.8, label="DAPO ck-15  pass@K", zorder=4)
    ax.plot(KS, [dapo_majk[k] for k in KS], "--",
            color="#dc2626", linewidth=1.4, alpha=0.65, label="DAPO  maj@K", zorder=4)

    ax.set_xscale("log", base=2)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("K (samples per question)")
    ax.set_ylabel("accuracy on D_test (%)")
    ax.set_title("D_test pass@K & maj@K — 3 post-training methods (T=0.7, top_p=0.95, n=64)",
                 fontsize=11, fontweight="semibold", loc="left")
    ax.grid(alpha=0.3, linestyle=":")
    ax.set_ylim(35, 95)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    # Annotation: deltas vs base
    txt = (f"vs base IT (Δ pass@1, pass@64, maj@64)\n"
           f"  SFT     : {sft_passk[1]-base_passk[1]:+.2f}, "
           f"{sft_passk[64]-base_passk[64]:+.2f}, "
           f"{sft_majk[64]-base_majk[64]:+.2f}\n"
           f"  DAPO    : {dapo_passk[1]-base_passk[1]:+.2f}, "
           f"{dapo_passk[64]-base_passk[64]:+.2f}, "
           f"{dapo_majk[64]-base_majk[64]:+.2f}")
    ax.text(1.05, 90, txt, fontsize=8.5, family="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9))

    plt.tight_layout()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"saved: {OUT_FILE}")

    print("\n=== summary ===")
    print(f"  {'method':<22s} {'p@1':>7s}  {'p@64':>7s}  {'maj@64':>7s}")
    print(f"  {'base IT':<22s} {base_passk[1]:6.2f}%  {base_passk[64]:6.2f}%  {base_majk[64]:6.2f}%")
    print(f"  {'SFT (lr=1e-4 ck-186)':<22s} {sft_passk[1]:6.2f}%  {sft_passk[64]:6.2f}%  {sft_majk[64]:6.2f}%")
    print(f"  {'DAPO (R15 ck-15)':<22s} {dapo_passk[1]:6.2f}%  {dapo_passk[64]:6.2f}%  {dapo_majk[64]:6.2f}%")
    print()
    print(f"  Δ SFT  vs base: p@1={sft_passk[1]-base_passk[1]:+.2f}  p@64={sft_passk[64]-base_passk[64]:+.2f}  maj@64={sft_majk[64]-base_majk[64]:+.2f}")
    print(f"  Δ DAPO vs base: p@1={dapo_passk[1]-base_passk[1]:+.2f}  p@64={dapo_passk[64]-base_passk[64]:+.2f}  maj@64={dapo_majk[64]-base_majk[64]:+.2f}")


if __name__ == "__main__":
    main()

"""K=64 D_test: pass@K + maj@K curves, with CKPT as primary dimension.

Layout: 2 rows (5 panels each) = 10 panels, one per ckpt step.
Each panel: X = K (log), lines = LR (3 colors).
  solid line = pass@K, dashed line = maj@K.
"""
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "v3" / "E2_sft" / "outputs" / "test_eval_k64"
BASE_CURVES = ROOT / "v3" / "E1_baseline" / "outputs" / "pass_at_k_20260427_222954" / "base_gemma-2-2b-it_k64_curves.json"
TEST_FILE = ROOT / "v3" / "shared" / "data" / "gsm8k" / "test_pc.jsonl"
OUT_FILE = ROOT / "v3" / "E2_sft" / "outputs" / "passk_majk_by_ckpt.png"

LRS = ["1e-4", "5e-4", "1e-3"]
LR_COLORS = {"1e-4": "#3b82f6", "5e-4": "#16a34a", "1e-3": "#dc2626"}
KS = [1, 2, 4, 8, 16, 32, 64]
STEPS = [10, 30, 50, 70, 90, 110, 130, 150, 170, 186]


def parse(jf):
    parts = jf.stem.split("_")
    return parts[1][2:], int(parts[3].split("-")[1])


def gold_from_completion(completion):
    txt = completion[0]["content"] if isinstance(completion, list) else completion
    if "\\boxed{" in txt:
        end = txt.rfind("}")
        start = txt.rfind("\\boxed{") + len("\\boxed{")
        return txt[start:end].strip()
    return None


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


def pass_at_k_unbiased(c, n, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def main():
    # Load gold answers
    golds_norm = []
    with open(TEST_FILE) as f:
        for line in f:
            ex = json.loads(line)
            golds_norm.append(normalize(gold_from_completion(ex["completion"])))

    # Group by (lr, step)
    by_key = {}
    for jf in EVAL_DIR.glob("sft_lr*_checkpoint-*.json"):
        try:
            lr, step = parse(jf)
        except Exception:
            continue
        if lr not in LRS or step not in STEPS:
            continue
        d = json.load(open(jf))
        per_q_ans = d["per_sample_answers"]
        per_q_norm = [[normalize(a) for a in row] for row in per_q_ans]
        per_q_correct = [sum(1 for a in row if a == g) for row, g in zip(per_q_norm, golds_norm)]
        K_max = len(per_q_norm[0]) if per_q_norm else 64
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
        by_key[(lr, step)] = {"passk": passk, "majk": majk}

    # Load base IT curves
    base = json.load(open(BASE_CURVES))
    base_passk = {int(k): v * 100 for k, v in base["pass_at_k_numeric"].items()}
    base_majk = {int(k): v * 100 for k, v in base["maj_at_k_numeric"].items()}

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharex=True, sharey=True)

    base_ks = sorted(base_passk.keys())
    for ax, step in zip(axes.flat, STEPS):
        # Base IT reference (black) — drawn first so LR colors sit on top
        ax.plot(base_ks, [base_passk[k] for k in base_ks], "-",
                color="black", markersize=4, linewidth=1.6, alpha=0.85,
                label="base IT pass@K", zorder=2)
        ax.plot(base_ks, [base_majk[k] for k in base_ks], "--",
                color="black", markersize=3, linewidth=1.2, alpha=0.7,
                label="base IT maj@K", zorder=2)
        for lr in LRS:
            entry = by_key.get((lr, step))
            if not entry:
                continue
            ks = sorted(entry["passk"].keys())
            ax.plot(ks, [entry["passk"][k] for k in ks], "-",
                    color=LR_COLORS[lr], markersize=4, linewidth=1.5,
                    label=f"lr={lr} pass@K", zorder=3)
            ax.plot(ks, [entry["majk"][k] for k in ks], "--",
                    color=LR_COLORS[lr], markersize=3, linewidth=1.2, alpha=0.75,
                    label=f"lr={lr} maj@K", zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS], fontsize=8)
        ax.set_title(f"step = {step}", loc="left", fontsize=10, fontweight="semibold")
        ax.grid(alpha=0.25, linestyle=":")
        ax.set_ylim(20, 100)

    for ax in axes[-1, :]:
        ax.set_xlabel("K")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (%)")
    # Legend on the first panel (smallest information loss; bottom-right empty there at low K)
    axes[0, 0].legend(loc="lower right", fontsize=7, ncol=1, framealpha=0.9)

    fig.suptitle("K=64 D_test — pass@K (solid) + maj@K (dashed), per ckpt step × 3 LRs (black = base IT reference)",
                 fontsize=12, fontweight="semibold")
    plt.tight_layout()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"saved: {OUT_FILE}")


if __name__ == "__main__":
    main()

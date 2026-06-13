"""K=64 D_test: pass@K and maj@K curves for 3 LRs × 10 ckpts each.

Layout: 2 rows (pass@K, maj@K) × 3 cols (lr=1e-3, 1e-4, 5e-4).
X-axis: K (1, 2, 4, 8, 16, 32, 64), log scale.
Y-axis: accuracy (%).
Lines: one per ckpt (10 per panel), color = step.
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "v3" / "E2_sft" / "outputs" / "test_eval_k64"
TEST_FILE = ROOT / "v3" / "shared" / "data" / "gsm8k" / "test_pc.jsonl"
OUT_FILE = ROOT / "v3" / "E2_sft" / "outputs" / "passk_majk_curves.png"

LRS = ["1e-4", "5e-4", "1e-3"]
KS = [1, 2, 4, 8, 16, 32, 64]


def parse(jf):
    name = jf.stem  # sft_lr5e-4_r64_checkpoint-90
    parts = name.split("_")
    lr = parts[1][2:]
    step = int(parts[3].split("-")[1])
    return lr, step


def gold_from_completion(completion):
    """Extract \\boxed{N} from completion."""
    txt = completion[0]["content"] if isinstance(completion, list) else completion
    if "\\boxed{" in txt:
        end = txt.rfind("}")
        start = txt.rfind("\\boxed{") + len("\\boxed{")
        return txt[start:end].strip()
    return None


def load_golds():
    """Return list of gold answer strings (length 1319)."""
    golds = []
    with open(TEST_FILE) as f:
        for line in f:
            ex = json.loads(line)
            golds.append(gold_from_completion(ex["completion"]))
    return golds


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
    """Codex 2021 unbiased pass@k: 1 - C(n-c, k)/C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def compute_pass_at_K(per_q_correct_count, K_max, ks):
    """For each k in ks, compute mean pass@k across questions."""
    out = {}
    Q = len(per_q_correct_count)
    for k in ks:
        if k > K_max:
            continue
        s = 0.0
        for c in per_q_correct_count:
            s += pass_at_k_unbiased(c, K_max, k)
        out[k] = s / Q * 100
    return out


def compute_maj_at_K(per_q_answers, golds, ks):
    """For each k, compute majority-vote accuracy using first k samples."""
    out = {}
    Q = len(per_q_answers)
    for k in ks:
        n_correct = 0
        for ans_list, gold in zip(per_q_answers, golds):
            if not ans_list:
                continue
            first_k = ans_list[:k]
            if not first_k:
                continue
            # Majority vote (normalize for grouping)
            normed = [normalize(a) for a in first_k if a is not None]
            if not normed:
                continue
            cnt = Counter(normed)
            top = cnt.most_common(1)[0][0]
            if top == normalize(gold):
                n_correct += 1
        out[k] = n_correct / Q * 100
    return out


def main():
    golds_norm = [normalize(g) for g in load_golds()]
    print(f"loaded {len(golds_norm)} golds")

    # Group ckpts by lr
    by_lr = {lr: [] for lr in LRS}
    for jf in sorted(EVAL_DIR.glob("sft_lr*_checkpoint-*.json")):
        lr, step = parse(jf)
        if lr not in by_lr:
            continue
        d = json.load(open(jf))
        per_q_ans = d["per_sample_answers"]
        per_q_norm = [[normalize(a) for a in row] for row in per_q_ans]
        # per-question correct count for pass@k
        per_q_correct = [
            sum(1 for a in row if a == g)
            for row, g in zip(per_q_norm, golds_norm)
        ]
        K_max = len(per_q_norm[0]) if per_q_norm else 64
        passk = compute_pass_at_K(per_q_correct, K_max, KS)
        majk = compute_maj_at_K(per_q_norm, golds_norm, KS)
        by_lr[lr].append({"step": step, "passk": passk, "majk": majk})
        print(f"  {lr} step={step}: pass@1={passk[1]:.1f} pass@64={passk[64]:.1f} maj@64={majk[64]:.1f}")
    for lr in by_lr:
        by_lr[lr].sort(key=lambda r: r["step"])

    # Load base IT curves (E1 baseline run)
    base_curves_path = Path("/mnt/d/fine-tuning/v3/E1_baseline/outputs/pass_at_k_20260427_222954/base_gemma-2-2b-it_k64_curves.json")
    base = json.load(open(base_curves_path))
    base_passk = {int(k): v * 100 for k, v in base["pass_at_k_numeric"].items()}
    base_majk = {int(k): v * 100 for k, v in base["maj_at_k_numeric"].items()}

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey="row")

    for col, lr in enumerate(LRS):
        ckpts = by_lr[lr]
        if not ckpts:
            continue
        steps = [c["step"] for c in ckpts]
        # Color by step (viridis dark→bright = early→late)
        colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(ckpts)))

        for row, key in enumerate(["passk", "majk"]):
            ax = axes[row, col]
            for c, color in zip(ckpts, colors):
                ks = sorted(c[key].keys())
                vals = [c[key][k] for k in ks]
                ax.plot(ks, vals, "-", color=color, label=f"step={c['step']}",
                        markersize=4, linewidth=1.4, alpha=0.9)
            # Base IT reference (black, thick, dashed for visibility)
            base_data = base_passk if key == "passk" else base_majk
            base_ks = sorted(base_data.keys())
            base_vals = [base_data[k] for k in base_ks]
            ax.plot(base_ks, base_vals, "k--", linewidth=2.0, alpha=0.85,
                    label="base IT", zorder=10)
            ax.set_xscale("log", base=2)
            ax.set_xticks(KS)
            ax.set_xticklabels([str(k) for k in KS])
            if row == 0:
                ax.set_title(f"lr={lr}", loc="left", fontsize=11, fontweight="semibold")
            if col == 0:
                ylabel = "pass@K (%)" if key == "passk" else "maj@K (%)"
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("K")
            ax.grid(alpha=0.25, linestyle=":")
            if col == 2:
                ax.legend(loc="best", fontsize=7, ncol=2)

    fig.suptitle("K=64 D_test — pass@K and maj@K curves (3 LRs × 10 ckpts each)",
                 fontsize=12, fontweight="semibold")
    plt.tight_layout()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"\nsaved: {OUT_FILE}")


if __name__ == "__main__":
    main()

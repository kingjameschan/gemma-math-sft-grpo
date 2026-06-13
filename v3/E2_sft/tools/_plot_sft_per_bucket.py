"""SFT K=64 D_test: per-difficulty-bucket pass rates trajectory.

3 LRs × 10 ckpts × 3 buckets (Easy/Med/Hard).
Output: 1 figure with 2 rows × 3 cols:
  Row 1 = pass@1 (greedy) per bucket, one panel per LR
  Row 2 = pass@K avg per bucket, one panel per LR
+ tabular dump per LR.
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "v3" / "E2_sft" / "outputs" / "test_eval_k64"
LABELS = ROOT / "v3" / "shared" / "data" / "gsm8k" / "test_difficulty_labels.jsonl"
TEST_PC = ROOT / "v3" / "shared" / "data" / "gsm8k" / "test_pc.jsonl"
BASE_K64 = ROOT / "v3" / "E1_baseline" / "outputs" / "pass_at_k_20260427_222954" / "base_gemma-2-2b-it_k64.json"
OUT_FILE = ROOT / "v3" / "E2_sft" / "outputs" / "sft_per_bucket_trajectory.png"

LRS = ["1e-4", "5e-4", "1e-3"]
LR_COLORS = {"1e-4": "#1f77b4", "5e-4": "#16a34a", "1e-3": "#dc2626"}
STEPS = [10, 30, 50, 70, 90, 110, 130, 150, 170, 186]
K = 64
BUCKETS = ["Easy", "Medium", "Hard"]
BUCKET_COLORS = {"Easy": "#16a34a", "Medium": "#f59e0b", "Hard": "#dc2626"}


def normalize(s):
    if s is None: return None
    s = str(s).strip().replace(",","").replace("$","").replace(" ","")
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v): return s
        return str(int(v)) if v == int(v) else str(v)
    except (ValueError, TypeError, OverflowError):
        return s


def load_buckets():
    by_bucket = {b: [] for b in BUCKETS}
    for line in open(LABELS):
        d = json.loads(line)
        if d["bucket"] in by_bucket:
            by_bucket[d["bucket"]].append(d["question_idx"])
    return by_bucket


def load_golds():
    golds = []
    for line in open(TEST_PC):
        ex = json.loads(line)
        txt = ex["completion"][0]["content"]
        if "\\boxed{" in txt:
            e = txt.rfind("}"); s = txt.rfind("\\boxed{") + len("\\boxed{")
            golds.append(normalize(txt[s:e].strip()))
        else:
            golds.append(None)
    return golds


def base_per_bucket():
    """Returns {bucket: avg per-q pass@K} for base IT."""
    bd = json.load(open(BASE_K64))
    samples = bd["samples"]
    by = load_buckets()
    out = {}
    for b in BUCKETS:
        out[b] = float(np.mean([samples[i]["any_correct_per_K"] / K for i in by[b]]) * 100)
    return out


def compute_lr(lr, by_bucket, golds):
    """Returns dict[bucket] = {step: (p1, pK)}."""
    res = {b: {} for b in BUCKETS}
    for step in STEPS:
        fp = EVAL_DIR / f"sft_lr{lr}_r64_checkpoint-{step}.json"
        d = json.load(open(fp))
        cc = d["per_question_correct_count"]
        greedy_norm = [normalize(a) for a in d["greedy_extracted"]]
        for b in BUCKETS:
            idxs = by_bucket[b]
            p1 = sum(1 for i in idxs if greedy_norm[i] == golds[i]) / len(idxs) * 100
            pK = sum(cc[i] for i in idxs) / len(idxs) / K * 100
            res[b][step] = (p1, pK)
    return res


def main():
    by_bucket = load_buckets()
    golds = load_golds()
    base = base_per_bucket()

    print(f"BASE per-q pass@K: Easy={base['Easy']:.1f}%  Med={base['Medium']:.1f}%  Hard={base['Hard']:.1f}%\n")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    for col, lr in enumerate(LRS):
        res = compute_lr(lr, by_bucket, golds)

        # Print table
        print(f"=== lr={lr} ===")
        print(f"{'step':>4} | " + " | ".join(f"{b}_p1/{b[0]}_pK".ljust(13) for b in BUCKETS))
        for step in STEPS:
            row = "  ".join(f"{res[b][step][0]:>5.1f}/{res[b][step][1]:>5.1f}" for b in BUCKETS)
            print(f"{step:>4} | {row}")
        print()

        for row_idx, metric_name in enumerate(["p1", "pK"]):
            ax = axes[row_idx, col]
            for b in BUCKETS:
                vals = [res[b][s][row_idx] for s in STEPS]
                ax.plot(STEPS, vals, "o-", color=BUCKET_COLORS[b],
                        markersize=4, linewidth=1.6, label=b, alpha=0.95)
                # Base reference
                ax.axhline(base[b], color=BUCKET_COLORS[b], linestyle="--",
                           linewidth=0.8, alpha=0.45)
            ax.set_xticks(STEPS)
            ax.set_xticklabels([str(s) for s in STEPS], rotation=45, fontsize=8)
            if row_idx == 0:
                ax.set_title(f"lr={lr}", loc="left", fontsize=11, fontweight="semibold")
            if col == 0:
                ax.set_ylabel(f"{'pass@1 greedy' if metric_name == 'p1' else 'avg per-q pass@K'} (%)")
            if row_idx == 1:
                ax.set_xlabel("ckpt step")
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.25, linestyle=":")
            if col == 2 and row_idx == 0:
                ax.legend(loc="center right", fontsize=8)

    fig.suptitle("SFT per-bucket pass rates trajectory (3 LRs × 3 difficulty buckets), dashed = base IT",
                 fontsize=12, fontweight="semibold")
    plt.tight_layout()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"saved: {OUT_FILE}")


if __name__ == "__main__":
    main()

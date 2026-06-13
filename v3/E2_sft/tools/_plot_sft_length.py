"""SFT K=64 D_test length analysis: 3 LRs × 10 ckpts.

For each LR, render a 2x5 grid (10 ckpts) of length distribution histograms
with correct/wrong split. Plus a combined trajectory comparison plot.

Length proxy: char count of per_sample_responses (avoids slow re-tokenization).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "v3" / "E2_sft" / "outputs" / "test_eval_k64"
LRS = ["1e-4", "5e-4", "1e-3"]
LR_COLORS = {"1e-4": "#1f77b4", "5e-4": "#16a34a", "1e-3": "#dc2626"}
STEPS = [10, 30, 50, 70, 90, 110, 130, 150, 170, 186]
K = 64

# X-axis cap = p99.5 × 1.05 across ckpts (same convention as E1 length plot,
# excludes the 1024-token whitespace-bug spike)


def normalize(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("$", "").replace(" ", "")
    try:
        v = float(s)
        if v == int(v):
            return str(int(v))
        return str(v)
    except (ValueError, TypeError, OverflowError):
        return s


def load_ckpt(lr, step):
    return json.load(open(EVAL_DIR / f"sft_lr{lr}_r64_checkpoint-{step}.json"))


def collect_lengths(d, golds_norm):
    """Return (correct_lens, wrong_lens) char-count arrays."""
    per_resps = d["per_sample_responses"]
    per_ans = d["per_sample_answers"]
    correct, wrong = [], []
    for resps, anses, gold in zip(per_resps, per_ans, golds_norm):
        for r, a in zip(resps, anses):
            n = len(r)
            if normalize(a) == gold:
                correct.append(n)
            else:
                wrong.append(n)
    return np.array(correct), np.array(wrong)


def render_one_lr(lr, golds_norm):
    out_file = ROOT / "v3" / "E2_sft" / "outputs" / f"lr{lr}_length_grid.png"

    # First pass: gather all lengths to compute global p99.5 cap
    all_data = {}
    for step in STEPS:
        d = load_ckpt(lr, step)
        correct, wrong = collect_lengths(d, golds_norm)
        all_data[step] = (correct, wrong)
    all_lens_global = np.concatenate([np.concatenate(v) for v in all_data.values()])
    xmax = float(np.percentile(all_lens_global, 99.5)) * 1.05
    bin_edges = np.linspace(0, xmax, 41)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)

    table = []
    for ax, step in zip(axes.flat, STEPS):
        correct, wrong = all_data[step]
        ax.hist([correct, wrong], bins=bin_edges, stacked=True,
                color=["#16a34a", "#dc2626"], alpha=0.85, label=["correct", "wrong"],
                edgecolor="white", linewidth=0.3)
        all_lens = np.concatenate([correct, wrong])
        c_mean = correct.mean() if len(correct) else 0
        w_mean = wrong.mean() if len(wrong) else 0
        ax.axvline(all_lens.mean(), color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(f"step={step}  mean={all_lens.mean():.0f}c  C={c_mean:.0f} W={w_mean:.0f}",
                     loc="left", fontsize=9, fontweight="semibold")
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlim(0, xmax)
        table.append((step, all_lens.mean(), c_mean, w_mean,
                      np.percentile(all_lens, 50), np.percentile(all_lens, 95)))

    for ax in axes[1, :]:
        ax.set_xlabel("response char count (~chars/4 ≈ tokens)")
    for ax in axes[:, 0]:
        ax.set_ylabel("# responses")
    axes[0, 0].legend(loc="upper right", fontsize=8)

    fig.suptitle(f"lr={lr} r=64 K=64 D_test — response length distribution per ckpt (correct vs wrong)",
                 fontsize=12, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(out_file, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"saved: {out_file}")

    print(f"\n=== lr={lr} length stats (chars) ===")
    print(f"{'step':>4} | {'all_mean':>9} {'C_mean':>7} {'W_mean':>7} | {'p50':>5} {'p95':>5}")
    print("-" * 55)
    for r in table:
        print(f"{r[0]:>4} | {r[1]:>8.0f}c {r[2]:>6.0f}c {r[3]:>6.0f}c | {r[4]:>5.0f} {r[5]:>5.0f}")
    print()
    plt.close()


def main():
    # Load gold answers
    test_pc = ROOT / "v3" / "shared" / "data" / "gsm8k" / "test_pc.jsonl"
    golds_norm = []
    with open(test_pc) as f:
        for line in f:
            ex = json.loads(line)
            txt = ex["completion"][0]["content"]
            if "\\boxed{" in txt:
                end = txt.rfind("}")
                start = txt.rfind("\\boxed{") + len("\\boxed{")
                gold = txt[start:end].strip()
            else:
                gold = None
            golds_norm.append(normalize(gold))

    for lr in LRS:
        render_one_lr(lr, golds_norm)


if __name__ == "__main__":
    main()

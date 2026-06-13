"""lr=5e-4 length analysis in E1 format (1×3 panels per ckpt).

Mirrors v3/E1_baseline/tools/_plot_length_analysis.py:
  A.1 length distribution histogram + accuracy line
  A.4 length distribution by difficulty bucket
  A.5 tokens vs step-count 2D heatmap

Outputs 10 PNG files (one per ckpt) under v3/E2_sft/outputs/length_e1style/.
"""
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "v3" / "E2_sft" / "outputs" / "test_eval_k64"
LABELS = ROOT / "v3" / "shared" / "data" / "gsm8k" / "test_difficulty_labels.jsonl"
TEST_PC = ROOT / "v3" / "shared" / "data" / "gsm8k" / "test_pc.jsonl"
BASE_MODEL = ROOT / "models" / "gemma-2-2b-it"
OUT_DIR = ROOT / "v3" / "E2_sft" / "outputs" / "length_e1style"

LR = "5e-4"
STEPS = [10, 30, 50, 70, 90, 110, 130, 150, 170, 186]
STEP_RE = re.compile(r"\*\*\s*(?:Step\s+)?\d+\.", re.IGNORECASE)
BUCKET_COLORS = {"Easy": "#16a34a", "Medium": "#f59e0b", "Hard": "#dc2626"}


def normalize(s):
    if s is None: return None
    s = str(s).strip().replace(",", "").replace("$", "").replace(" ", "")
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v): return s
        return str(int(v)) if v == int(v) else str(v)
    except (ValueError, TypeError, OverflowError):
        return s


def render_ckpt(step, tok, labels, golds_norm):
    fp = EVAL_DIR / f"sft_lr{LR}_r64_checkpoint-{step}.json"
    d = json.load(open(fp))
    per_resps = d["per_sample_responses"]   # 1319 × 64 strings
    per_ans = d["per_sample_answers"]       # 1319 × 64 normalized

    # Flatten per-response rows
    print(f"  [step {step}] tokenizing {sum(len(r) for r in per_resps)} responses...")
    tokens, steps_arr, correct, buckets = [], [], [], []
    for q_idx, (resps, anses) in enumerate(zip(per_resps, per_ans)):
        gold = golds_norm[q_idx]
        bucket = labels.get(q_idx, "?")
        # Batch tokenize this question's K responses
        enc = tok(resps, add_special_tokens=False, return_attention_mask=False)
        for r_idx, resp in enumerate(resps):
            tokens.append(len(enc["input_ids"][r_idx]))
            steps_arr.append(len(STEP_RE.findall(resp)))
            pred = normalize(anses[r_idx])
            correct.append(pred == gold)
            buckets.append(bucket)
    tokens = np.array(tokens)
    steps_arr = np.array(steps_arr)
    correct = np.array(correct)
    buckets = np.array(buckets)

    # === Plot ===
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    p99 = np.percentile(tokens, 99)
    p99_5 = np.percentile(tokens, 99.5)
    # Globally unified X-axis (covers base IT p99.5≈482 + max SFT p99.5≈640)
    xmax = 700
    YMAX_A1 = 15000

    # ---- A.1: hist + acc ----
    ax = axes[0]
    n_bins = 40
    edges = np.linspace(0, xmax, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    bin_idx = np.digitize(tokens, edges) - 1
    accs, counts = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            accs.append(np.nan); counts.append(0); continue
        accs.append(correct[mask].mean() * 100)
        counts.append(int(mask.sum()))
    ax.bar(centers, counts, width=width * 0.95, color="#2563eb", alpha=0.85,
           edgecolor="white", linewidth=0.4)
    for q in [50, 75, 90, 95, 99, 99.5]:
        v = np.percentile(tokens, q)
        ax.axvline(v, color="#999", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.text(v, ax.get_ylim()[1] * 0.95, f" p{q}={int(v)}",
                fontsize=8, color="#555", rotation=90, va="top")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, YMAX_A1)
    ax.set_xlabel("response tokens (binned)")
    ax.set_ylabel("# responses in bin", color="#2563eb")
    ax.set_title(f"(A.1) length dist + acc  (mean={tokens.mean():.0f}, p99={int(p99)}, p99.5={int(p99_5)})",
                 loc="left", fontsize=11)
    ax2 = ax.twinx()
    ax2.plot(centers, accs, "o-", color="black", markersize=5, linewidth=1.8, alpha=0.85,
             label="accuracy in bin")
    ax2.set_ylabel("accuracy (%) in bin", color="black", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(False)

    # ---- A.4: density by bucket ----
    ax = axes[1]
    bins_dense = np.linspace(0, xmax, 40)
    bin_centers_dense = (bins_dense[:-1] + bins_dense[1:]) / 2
    for b in ["Easy", "Medium", "Hard"]:
        mask = buckets == b
        if mask.sum() == 0:
            continue
        data = tokens[mask]
        density, _ = np.histogram(data, bins=bins_dense, density=True)
        ax.plot(bin_centers_dense, density, "-", color=BUCKET_COLORS[b], linewidth=2.2,
                label=f"{b} (n={mask.sum()}, mean={data.mean():.0f}, p99={int(np.percentile(data, 99))})")
        ax.fill_between(bin_centers_dense, density, alpha=0.10, color=BUCKET_COLORS[b])
        ax.axvline(data.mean(), color=BUCKET_COLORS[b], linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("response tokens")
    ax.set_ylabel("density")
    ax.set_title("(A.4) length distribution by difficulty", loc="left", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)

    # ---- A.5: tokens vs step-count heatmap ----
    ax = axes[2]
    a5_xmax = xmax
    a5_ymax = max(int(np.percentile(steps_arr, 99.5)) + 1, 8)
    x_bins = np.linspace(0, a5_xmax, 21)
    y_bins = np.arange(0, a5_ymax + 2)
    in_range = (tokens <= a5_xmax) & (steps_arr <= a5_ymax)
    H, xe, ye = np.histogram2d(tokens[in_range], steps_arr[in_range], bins=[x_bins, y_bins])
    H_masked = np.ma.masked_where(H == 0, H)
    pcm = ax.pcolormesh(xe, ye, H_masked.T, cmap="Purples",
                        edgecolors="white", linewidth=0.5)
    fig.colorbar(pcm, ax=ax, label="# responses", shrink=0.85, pad=0.02)
    edges_a5 = np.linspace(0, a5_xmax, 25)
    centers_a5 = (edges_a5[:-1] + edges_a5[1:]) / 2
    bin_idx_a5 = np.digitize(tokens, edges_a5) - 1
    mean_steps = []
    for b in range(len(centers_a5)):
        mask = bin_idx_a5 == b
        mean_steps.append(steps_arr[mask].mean() if mask.sum() > 5 else np.nan)
    ax.plot(centers_a5, mean_steps, "o-", color="black", markersize=4, linewidth=1.6,
            label="mean steps in bin", zorder=5)
    ax.set_xlim(0, a5_xmax)
    ax.set_ylim(0, a5_ymax)
    ax.set_xlabel("response tokens")
    ax.set_ylabel("step count (**N. markers)")
    ax.set_title("(A.5) tokens vs step-count heatmap", loc="left", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(f"lr={LR} r=64 D_test K=64 — step={step}  (n_resp={len(tokens)})",
                 fontsize=12, fontweight="semibold")
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"lr5e-4_step{step}_length_e1style.png"
    plt.savefig(out_file, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    saved: {out_file.name}  | mean_tok={tokens.mean():.0f}, n_resp={len(tokens)}")


def main():
    print("[load] tokenizer...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("[load] difficulty labels...")
    labels = {}
    for line in open(LABELS):
        d = json.loads(line)
        labels[d["question_idx"]] = d["bucket"]

    print("[load] golds from test_pc.jsonl...")
    golds_norm = []
    for line in open(TEST_PC):
        ex = json.loads(line)
        txt = ex["completion"][0]["content"]
        if "\\boxed{" in txt:
            e = txt.rfind("}"); s = txt.rfind("\\boxed{") + len("\\boxed{")
            golds_norm.append(normalize(txt[s:e].strip()))
        else:
            golds_norm.append(None)

    print(f"\n[render] {len(STEPS)} ckpts...")
    for step in STEPS:
        render_ckpt(step, tok, labels, golds_norm)

    print(f"\n[done] all 10 PNGs in {OUT_DIR}")


if __name__ == "__main__":
    main()

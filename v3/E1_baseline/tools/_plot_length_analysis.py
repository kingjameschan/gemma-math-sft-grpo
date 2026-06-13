"""Class A length-evolution analysis plots for v3 (Section 3.A in 实验方案).

Inputs:
  - pass_at_k JSON (has all K responses per question)
  - difficulty labels JSONL (Easy/Medium/Hard per question)

Outputs (1 figure, 2×2 panels):
  A.1 Length distribution histogram
  A.3 Length vs correctness (binned)
  A.4 Length distribution split by difficulty bucket
  A.5 Token-count vs step-count scatter

Step-count definition (heuristic for Gemma2-IT): regex on response text
counting "**N." or "**Step N:" markers (Gemma2's typical numbered-step format).

Usage:
  python3 v3/tools/_plot_length_analysis.py path/to/pass_at_k.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
BASE_MODEL = ROOT / "models" / "gemma-2-2b-it"
LABELS_FILE = ROOT / "v3" / "shared" / "data" / "gsm8k" / "test_difficulty_labels.jsonl"

STEP_RE = re.compile(r"\*\*\s*(?:Step\s+)?\d+\.", re.IGNORECASE)


def count_steps(text: str) -> int:
    """Count step markers (**N. or **Step N:) in response."""
    return len(STEP_RE.findall(text))


def load_labels() -> dict:
    """Map question_idx → bucket."""
    if not LABELS_FILE.exists():
        return {}
    out = {}
    with open(LABELS_FILE) as f:
        for line in f:
            d = json.loads(line)
            out[d["question_idx"]] = d["bucket"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pass_at_k_json", help="path to pass_at_k_<TS>/<tag>_k<K>.json")
    ap.add_argument("--out", default=None, help="output png path")
    args = ap.parse_args()

    fp = Path(args.pass_at_k_json)
    d = json.load(open(fp))
    samples = d["samples"]
    K = d["config"]["K"]
    tag = d["config"].get("tag", "?")
    n_q = len(samples)

    labels = load_labels()
    if not labels:
        raise SystemExit(f"missing difficulty labels: {LABELS_FILE}")

    print(f"[load] {fp.name} | tag={tag} | K={K} | n_q={n_q}")

    # Tokenize all K×N responses
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    print("[tokenize] computing per-response token+step counts...")
    rows = []  # one row per response: {tokens, steps, correct, bucket, gold, pred}
    for i, s in enumerate(samples):
        gold = s["gold"]
        bucket = labels.get(i, "?")
        for r_idx, resp in enumerate(s["responses"]):
            n_tok = len(tok.encode(resp, add_special_tokens=False))
            n_step = count_steps(resp)
            pred = s["any_preds"][r_idx]
            correct = (pred == gold)
            rows.append({
                "q_idx": i, "r_idx": r_idx,
                "tokens": n_tok, "steps": n_step,
                "correct": correct, "bucket": bucket,
            })

    n_resp = len(rows)
    tokens = np.array([r["tokens"] for r in rows])
    steps = np.array([r["steps"] for r in rows])
    correct = np.array([r["correct"] for r in rows])
    buckets = np.array([r["bucket"] for r in rows])

    print(f"[stats] n_resp={n_resp}, tokens mean={tokens.mean():.0f} median={np.median(tokens):.0f} p99={np.percentile(tokens,99):.0f}")
    print(f"[stats] correct rate (per-resp)={correct.mean()*100:.2f}%")

    # === Plot setup ===
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
    # Globally unified X-axis (matches E2 SFT length plots: covers base p99.5≈482 + max SFT p99.5≈640)
    xmax = 700
    YMAX_A1 = 15000

    # ---- A.1+A.3 merged: length dist (bars) + acc (line) + percentile markers ----
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
    ax.bar(centers, counts, width=width * 0.95, color="#2563eb", alpha=0.85, edgecolor="white", linewidth=0.4)
    # percentile vertical lines (include p99.5)
    for q in [50, 75, 90, 95, 99, 99.5]:
        v = np.percentile(tokens, q)
        ax.axvline(v, color="#999", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.text(v, ax.get_ylim()[1] * 0.95, f" p{q}={int(v)}",
                fontsize=8, color="#555", rotation=90, va="top")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, YMAX_A1)
    ax.set_xlabel("response tokens (binned)")
    ax.set_ylabel("# responses in bin", color="#2563eb")
    ax.set_title(f"(A.1) length distribution + accuracy  (mean={tokens.mean():.0f}, p99={int(p99)}, p99.5={int(p99_5)})",
                 loc="left", fontsize=11)
    # Accuracy line on right axis
    ax2 = ax.twinx()
    ax2.plot(centers, accs, "o-", color="black", markersize=5, linewidth=1.8, alpha=0.85,
             label="accuracy in bin")
    ax2.set_ylabel("accuracy (%) in bin", color="black", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(False)

    # ---- A.4: Length distribution by difficulty (line densities, no overlap mud) ----
    ax = axes[1]
    bucket_colors = {"Easy": "#16a34a", "Medium": "#f59e0b", "Hard": "#dc2626"}
    bucket_order = ["Easy", "Medium", "Hard"]
    bins_dense = np.linspace(0, xmax, 40)
    bin_centers_dense = (bins_dense[:-1] + bins_dense[1:]) / 2
    for b in bucket_order:
        mask = buckets == b
        if mask.sum() == 0:
            continue
        data = tokens[mask]
        density, _ = np.histogram(data, bins=bins_dense, density=True)
        # Smooth line plot + light fill below for visual emphasis
        ax.plot(bin_centers_dense, density, "-", color=bucket_colors[b],
                linewidth=2.2,
                label=f"{b} (n={mask.sum()}, mean={data.mean():.0f}, p99={int(np.percentile(data, 99))})")
        ax.fill_between(bin_centers_dense, density, alpha=0.10, color=bucket_colors[b])
        # Mean marker on x axis
        ax.axvline(data.mean(), color=bucket_colors[b], linestyle="--",
                   linewidth=0.9, alpha=0.7)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("response tokens")
    ax.set_ylabel("density")
    ax.set_title("(A.4) length distribution by difficulty", loc="left", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)

    # ---- A.5: Token count vs step count (gridded heatmap, integer step Y) ----
    ax = axes[2]
    a5_xmax = xmax
    a5_ymax = max(int(np.percentile(steps, 99.5)) + 1, 8)
    # X bins: 20 (coarser for grid look). Y bins: per integer step value.
    x_bins = np.linspace(0, a5_xmax, 21)
    y_bins = np.arange(0, a5_ymax + 2)  # 0, 1, 2, ..., y_max+1
    # Filter and compute 2D hist manually
    in_range = (tokens <= a5_xmax) & (steps <= a5_ymax)
    H, xe, ye = np.histogram2d(tokens[in_range], steps[in_range], bins=[x_bins, y_bins])
    # Mask zero cells (don't draw)
    H_masked = np.ma.masked_where(H == 0, H)
    pcm = ax.pcolormesh(xe, ye, H_masked.T, cmap="Purples",
                        edgecolors="white", linewidth=0.5)
    cbar = fig.colorbar(pcm, ax=ax, label="# responses", shrink=0.85, pad=0.02)
    # bin and overlay mean
    edges_a5 = np.linspace(0, a5_xmax, 25)
    centers_a5 = (edges_a5[:-1] + edges_a5[1:]) / 2
    bin_idx_a5 = np.digitize(tokens, edges_a5) - 1
    mean_steps = []
    for b in range(len(centers_a5)):
        mask = bin_idx_a5 == b
        mean_steps.append(steps[mask].mean() if mask.sum() > 5 else np.nan)
    ax.plot(centers_a5, mean_steps, "o-", color="black", markersize=4, linewidth=1.6,
            label="mean steps in bin", zorder=5)
    # annotate slope
    valid = [(c, s) for c, s in zip(centers_a5, mean_steps) if not np.isnan(s)]
    if len(valid) >= 2:
        x = np.array([v[0] for v in valid])
        y = np.array([v[1] for v in valid])
        slope, _ = np.polyfit(x, y, 1)
        ax.text(0.04, 0.95, f"slope ≈ {slope:.3f} steps/tok\n(≈ 1 step / {1/slope:.0f} tok)",
                transform=ax.transAxes, fontsize=9, color="#1f2937",
                va="top", bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#aaa"))
    ax.set_xlim(0, a5_xmax)
    ax.set_ylim(0, a5_ymax + 0.5)
    ax.set_xlabel("response tokens")
    ax.set_ylabel("# step markers (**N./Step N:)")
    ax.set_title("(A.5) tokens vs step count (grid heatmap)", loc="left", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(f"Class A length analysis — {tag} (K={K}, n_q={n_q}, total responses={n_resp})",
                 fontsize=12, fontweight="semibold", y=1.0)
    plt.tight_layout()

    out_path = Path(args.out) if args.out else fp.parent / f"{fp.stem}_length_classA.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

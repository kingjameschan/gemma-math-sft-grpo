"""lr=5e-4 mode-mass + wrong-concentration analysis.

Direction 1: scatter (correct_mass vs mode_mass) per ckpt + base
Direction 2: wrong_concentration distribution + trajectory

Outputs 2 PNG files.
"""
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "v3" / "E2_sft" / "outputs" / "test_eval_k64"
BASE_K64 = ROOT / "v3" / "E1_baseline" / "outputs" / "pass_at_k_20260427_222954" / "base_gemma-2-2b-it_k64.json"
TEST_PC = ROOT / "v3" / "shared" / "data" / "gsm8k" / "test_pc.jsonl"

OUT_SCATTER = ROOT / "v3" / "E2_sft" / "outputs" / "lr5e-4_mode_correctness_scatter.png"
OUT_WRONG = ROOT / "v3" / "E2_sft" / "outputs" / "lr5e-4_wrong_concentration.png"

LR = "5e-4"
STEPS = [10, 30, 50, 70, 90, 110, 130, 150, 170, 186]
K = 64


def normalize(s):
    if s is None: return None
    s = str(s).strip().replace(",", "").replace("$", "").replace(" ", "")
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v): return s
        return str(int(v)) if v == int(v) else str(v)
    except (ValueError, TypeError, OverflowError):
        return s


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


def compute_metrics(per_q_answers, golds):
    """Return per-question (correct_mass, mode_mass, wrong_concentration_or_None)."""
    out = []
    for ans_list, gold in zip(per_q_answers, golds):
        normed = [normalize(a) for a in ans_list]
        cnt = Counter(normed)
        total = sum(cnt.values())
        if total == 0:
            out.append((0.0, 0.0, None))
            continue
        correct_count = cnt.get(gold, 0)
        correct_mass = correct_count / total
        mode_count = max(cnt.values())
        mode_mass = mode_count / total
        # Wrong concentration only when mode != correct (or when there are wrong samples)
        wrong_count = total - correct_count
        if wrong_count > 0:
            non_correct = {k: v for k, v in cnt.items() if k != gold}
            top_wrong = max(non_correct.values()) if non_correct else 0
            wrong_concentration = top_wrong / wrong_count
        else:
            wrong_concentration = None  # all correct, undefined
        out.append((correct_mass, mode_mass, wrong_concentration))
    return out


def load_base_metrics(golds):
    bd = json.load(open(BASE_K64))
    samples = bd["samples"]
    K_base = bd["config"]["K"]
    out = []
    for s, gold in zip(samples, golds):
        # base format: samples[i]["responses_extracted"] or "answers" — check
        if "any_preds" in s:
            normed = [normalize(a) for a in s["any_preds"]]
        elif "answers" in s:
            normed = [normalize(a) for a in s["answers"]]
        else:
            return None
        cnt = Counter(normed)
        total = sum(cnt.values())
        if total == 0:
            out.append((0, 0, None))
            continue
        cc = cnt.get(gold, 0)
        cm = cc / total
        mm = max(cnt.values()) / total
        wc = (max((v for k, v in cnt.items() if k != gold), default=0) / (total - cc)) if total > cc else None
        out.append((cm, mm, wc))
    return out


def main():
    golds = load_golds()
    print(f"loaded {len(golds)} golds")

    # Try loading base
    bd_check = json.load(open(BASE_K64))
    print("base sample keys:", list(bd_check["samples"][0].keys())[:8])
    base_metrics = load_base_metrics(golds)
    have_base = base_metrics is not None
    if have_base:
        print(f"base metrics computed for {len(base_metrics)} questions")

    # SFT all ckpts
    by_step = {}
    for step in STEPS:
        d = json.load(open(EVAL_DIR / f"sft_lr{LR}_r64_checkpoint-{step}.json"))
        m = compute_metrics(d["per_sample_answers"], golds)
        by_step[step] = m

    # === Direction 1: scatter grid ===
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    n_panels = (1 if have_base else 0) + len(STEPS)
    n_rows, n_cols = 3, 4   # 12 slots, 11 used (base + 10)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 11), sharex=True, sharey=True)

    panel_data = []
    if have_base:
        panel_data.append(("base IT", base_metrics))
    for step in STEPS:
        panel_data.append((f"step={step}", by_step[step]))

    for ax, (name, metrics) in zip(axes.flat, panel_data):
        cm = np.array([m[0] for m in metrics])
        mm = np.array([m[1] for m in metrics])
        # Compute % mode=correct (within tolerance)
        pct_match = np.mean(np.abs(cm - mm) < 1e-6) * 100
        # Diagonal y=x (lower bound: mode >= correct)
        ax.plot([0, 1], [0, 1], "--", color="#666", linewidth=0.8, alpha=0.5)
        ax.scatter(cm, mm, s=4, alpha=0.25, color="#1f77b4", edgecolors="none")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{name}  mode=correct: {pct_match:.0f}%",
                     loc="left", fontsize=9, fontweight="semibold")
        ax.grid(alpha=0.2, linestyle=":")
    # Hide unused
    for ax in axes.flat[n_panels:]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("correct_mass = P(correct | x)")
    for ax in axes[:, 0]:
        ax.set_ylabel("mode_mass = max P(y | x)")
    fig.suptitle(f"lr={LR} — Mode Strength vs Correctness scatter (1319 questions × 11 ckpts; y=x = mode is correct)",
                 fontsize=12, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(OUT_SCATTER, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"saved: {OUT_SCATTER}")
    plt.close()

    # === Direction 2: wrong concentration ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram for selected ckpts
    selected = []
    if have_base: selected.append(("base IT", "black", base_metrics))
    selected.append(("step 10", "#888", by_step[10]))
    selected.append(("step 130 (best)", "#16a34a", by_step[130]))
    selected.append(("step 186 (final)", "#dc2626", by_step[186]))
    bins = np.linspace(0, 1.0001, 26)  # 4% bins
    ax = axes[0]
    for name, color, metrics in selected:
        wcs = [m[2] for m in metrics if m[2] is not None]
        ax.hist(wcs, bins=bins, alpha=0.45, label=f"{name} (n={len(wcs)})", color=color,
                edgecolor=color, linewidth=0.8, histtype="step", linewidth_=None) if False else \
        ax.hist(wcs, bins=bins, alpha=0.40, label=f"{name} (n={len(wcs)})", color=color,
                edgecolor="white", linewidth=0.4)
    ax.set_xlabel("wrong_concentration = top wrong freq / total wrong freq")
    ax.set_ylabel("# questions")
    ax.set_title("(a) wrong-answer concentration distribution (only Q with ≥1 wrong sample)",
                 loc="left", fontsize=10)
    ax.set_xlim(0, 1.02)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25, linestyle=":")

    # Right: trajectory of mean wrong_concentration across ckpts
    ax = axes[1]
    means = []
    medians = []
    for step in STEPS:
        wcs = [m[2] for m in by_step[step] if m[2] is not None]
        means.append(np.mean(wcs) * 100)
        medians.append(np.median(wcs) * 100)
    ax.plot(STEPS, means, "o-", color="#1f77b4", label="mean", linewidth=1.6, markersize=5)
    ax.plot(STEPS, medians, "s-", color="#dc2626", label="median", linewidth=1.4, markersize=4)
    if have_base:
        b_wcs = [m[2] for m in base_metrics if m[2] is not None]
        b_mean = np.mean(b_wcs) * 100
        ax.axhline(b_mean, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
                   label=f"base IT mean = {b_mean:.1f}%")
    ax.set_xticks(STEPS)
    ax.set_xticklabels([str(s) for s in STEPS], rotation=45, fontsize=8)
    ax.set_xlabel("ckpt step")
    ax.set_ylabel("wrong_concentration (%)")
    ax.set_title("(b) trajectory: mean/median wrong concentration",
                 loc="left", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25, linestyle=":")

    fig.suptitle(f"lr={LR} — wrong-answer concentration analysis",
                 fontsize=12, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(OUT_WRONG, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"saved: {OUT_WRONG}")
    plt.close()

    # Print summary stats
    print("\n=== Summary stats ===")
    print(f"{'ckpt':<14} {'mode=correct%':>13} {'mean mode_mass':>15} {'mean correct_mass':>17} {'mean wrong_conc':>16}")
    if have_base:
        cm = np.array([m[0] for m in base_metrics])
        mm = np.array([m[1] for m in base_metrics])
        wcs = [m[2] for m in base_metrics if m[2] is not None]
        pct_match = np.mean(np.abs(cm - mm) < 1e-6) * 100
        print(f"{'base IT':<14} {pct_match:>12.1f}% {mm.mean():>14.3f} {cm.mean():>16.3f} {np.mean(wcs):>15.3f}")
    for step in STEPS:
        cm = np.array([m[0] for m in by_step[step]])
        mm = np.array([m[1] for m in by_step[step]])
        wcs = [m[2] for m in by_step[step] if m[2] is not None]
        pct_match = np.mean(np.abs(cm - mm) < 1e-6) * 100
        print(f"step={step:<8} {pct_match:>12.1f}% {mm.mean():>14.3f} {cm.mean():>16.3f} {np.mean(wcs):>15.3f}")


if __name__ == "__main__":
    main()

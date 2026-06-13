"""Post-hoc compute pass@k and maj@k curves from a saved pass@K eval JSON.

Reads `v3/outputs/pass_at_k_<TS>/<tag>_k<K>.json` (which has all K responses
per question + per-sample extracted preds), then computes:

  - pass@k for k in [1..K] using unbiased estimator (Chen et al. HumanEval):
        pass@k = 1 - C(n-c, k) / C(n, k)
    where n=K, c=#correct in K samples per question
  - maj@k for k in [1..K] using bootstrap:
        for each q, sample k preds without replacement, vote, check correct.
        repeat n_bootstrap times, average across questions and bootstraps.

Both numeric (5-layer extraction) and boxed (only \\boxed{}) variants.

Usage:
  python3 v3/tools/_compute_pass_maj_curves.py path/to/pass_at_k.json
  python3 v3/tools/_compute_pass_maj_curves.py path/to/pass_at_k.json --plot
"""
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k. n=total samples, c=correct, k=desired."""
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.exp(sum(math.log(n - c - i) - math.log(n - i) for i in range(k)))


def majority_vote(preds: list[str]) -> str:
    """Plurality vote over non-empty preds. Empty preds excluded."""
    valid = [p for p in preds if p != ""]
    if not valid:
        return ""
    cnt = Counter(valid)
    return cnt.most_common(1)[0][0]


def is_correct(pred: str, gold: str) -> bool:
    """Loose match copy from main eval — keep simple here, just str compare
    after numeric normalization. Re-uses the 'is_correct' status from the
    saved JSON if available."""
    if pred == "" or gold == "":
        return False
    # both should already be strip_string'd in saved data
    if pred == gold:
        return True
    # numeric fallback
    try:
        return math.isclose(float(pred.replace(",", "")), float(gold.replace(",", "")), abs_tol=1e-3)
    except Exception:
        return False


def compute_pass_curve(samples: list[dict], K: int, ks: list[int], pred_field: str) -> dict:
    """pass@k curve. Uses precomputed 'any_correct_per_K' / 'boxed_correct_per_K'."""
    correct_field = "any_correct_per_K" if pred_field == "any_preds" else "boxed_correct_per_K"
    n_q = len(samples)
    out = {}
    for k in ks:
        if k > K:
            continue
        total = 0.0
        for s in samples:
            c = s[correct_field]
            total += pass_at_k(K, c, k)
        out[k] = total / n_q
    return out


def compute_maj_curve(samples: list[dict], K: int, ks: list[int], pred_field: str,
                      n_bootstrap: int = 200, seed: int = 42) -> dict:
    """maj@k curve via bootstrap. For each q, sample k preds without replacement
    `n_bootstrap` times, vote, count correct, average."""
    rng = random.Random(seed)
    n_q = len(samples)
    out = {}
    for k in ks:
        if k > K:
            continue
        total_correct_frac = 0.0
        for s in samples:
            preds = s[pred_field]
            gold = s["gold"]
            if k == K:
                # No bootstrap needed, just vote on all
                pred = majority_vote(preds)
                total_correct_frac += 1.0 if is_correct(pred, gold) else 0.0
            else:
                # Bootstrap: sample k without replacement, vote, check
                hits = 0
                for _ in range(n_bootstrap):
                    sub = rng.sample(preds, k)
                    pred = majority_vote(sub)
                    if is_correct(pred, gold):
                        hits += 1
                total_correct_frac += hits / n_bootstrap
        out[k] = total_correct_frac / n_q
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="path to pass_at_k_<TS>/<tag>_k<K>.json")
    ap.add_argument("--ks", default="1,2,4,8,16,32,64",
                    help="comma-separated k values for curve points")
    ap.add_argument("--n_bootstrap", type=int, default=200,
                    help="bootstrap samples for maj@k (default 200)")
    ap.add_argument("--plot", action="store_true", help="save plot alongside json")
    args = ap.parse_args()

    fp = Path(args.json_path)
    d = json.load(open(fp))
    config = d.get("config", {})
    samples = d.get("samples", [])
    K = config.get("K", 0)
    if not samples:
        raise SystemExit(f"No samples found in {fp}")
    if not K:
        # infer
        K = len(samples[0].get("any_preds", []))
    print(f"[load] {fp}\n       K={K}, n_q={len(samples)}, tag={config.get('tag','?')}")

    ks = [int(k) for k in args.ks.split(",") if int(k) <= K]
    print(f"[compute] k values: {ks}, bootstrap={args.n_bootstrap}")

    print("\nPass@k (unbiased estimator):")
    pass_n = compute_pass_curve(samples, K, ks, "any_preds")
    pass_b = compute_pass_curve(samples, K, ks, "boxed_preds")
    print(f"{'k':>3}  {'pass@k numeric':>14}  {'pass@k boxed':>13}")
    for k in ks:
        print(f"  {k:>2}  {pass_n[k]*100:>13.2f}%  {pass_b[k]*100:>12.2f}%")

    print("\nMaj@k (bootstrap):")
    maj_n = compute_maj_curve(samples, K, ks, "any_preds", args.n_bootstrap)
    maj_b = compute_maj_curve(samples, K, ks, "boxed_preds", args.n_bootstrap)
    print(f"{'k':>3}  {'maj@k numeric':>14}  {'maj@k boxed':>13}")
    for k in ks:
        print(f"  {k:>2}  {maj_n[k]*100:>13.2f}%  {maj_b[k]*100:>12.2f}%")

    # Save curves to json
    out_path = fp.parent / f"{fp.stem}_curves.json"
    with open(out_path, "w") as f:
        json.dump({
            "source": str(fp.relative_to(ROOT)) if fp.is_relative_to(ROOT) else str(fp),
            "K": K,
            "ks": ks,
            "n_bootstrap": args.n_bootstrap,
            "pass_at_k_numeric": pass_n,
            "pass_at_k_boxed": pass_b,
            "maj_at_k_numeric": maj_n,
            "maj_at_k_boxed": maj_b,
        }, f, indent=2)
    print(f"\nsaved curves: {out_path}")

    # Compute answer entropy per question (numeric and boxed pred distributions)
    print("\nComputing answer entropy (per-question)...")
    entropy_any = []
    entropy_boxed = []
    correct_any = []  # majority-vote-correct flag per question
    correct_boxed = []
    for s in samples:
        # numeric pred entropy
        cnt = Counter(s["any_preds"])
        n = sum(cnt.values())
        H = -sum((c/n) * math.log(c/n) for c in cnt.values() if c > 0) if n > 0 else 0.0
        entropy_any.append(H)
        # boxed pred entropy (treat empty string as one category)
        cnt = Counter(s["boxed_preds"])
        n = sum(cnt.values())
        H = -sum((c/n) * math.log(c/n) for c in cnt.values() if c > 0) if n > 0 else 0.0
        entropy_boxed.append(H)
        # correctness via majority vote
        gold = s["gold"]
        correct_any.append(is_correct(majority_vote(s["any_preds"]), gold))
        correct_boxed.append(is_correct(majority_vote(s["boxed_preds"]), gold))

    import statistics
    print(f"  H_any  : mean={statistics.mean(entropy_any):.3f} median={statistics.median(entropy_any):.3f} max={max(entropy_any):.3f}")
    print(f"  H_boxed: mean={statistics.mean(entropy_boxed):.3f} median={statistics.median(entropy_boxed):.3f} max={max(entropy_boxed):.3f}")

    # Save entropy data alongside curves
    out_data = json.load(open(out_path))
    out_data["entropy"] = {
        "per_question_H_any": entropy_any,
        "per_question_H_boxed": entropy_boxed,
        "per_question_correct_any": [bool(x) for x in correct_any],
        "per_question_correct_boxed": [bool(x) for x in correct_boxed],
        "mean_H_any": statistics.mean(entropy_any),
        "mean_H_boxed": statistics.mean(entropy_boxed),
        "median_H_any": statistics.median(entropy_any),
        "max_H_theoretical": math.log(K),
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # ---------- Plot 1: pass@k + maj@k curves (numeric only) ----------
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(ks, [pass_n[k]*100 for k in ks], "-", label="pass@k", color="#2563eb", linewidth=2)
            ax.plot(ks, [maj_n[k]*100 for k in ks], "-", label="maj@k", color="#dc2626", linewidth=2)
            # annotate gap at last k
            k_last = ks[-1]
            ax.annotate(f"gap = {(pass_n[k_last]-maj_n[k_last])*100:.1f} pp",
                        xy=(k_last, (pass_n[k_last]+maj_n[k_last])*50),
                        xytext=(k_last*0.6, (pass_n[k_last]+maj_n[k_last])*50),
                        fontsize=10, ha="right",
                        arrowprops=dict(arrowstyle="<->", color="#666", lw=0.8))
            ax.set_xscale("log", base=2)
            ax.set_xticks(ks)
            ax.set_xticklabels([str(k) for k in ks])  # integer labels, not 2^x or scientific
            ax.set_xlabel("k"); ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"pass@k vs maj@k — {config.get('tag','?')} (K={K}, any-pred / 5-layer)")
            ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=11)
            plot_path = fp.parent / f"{fp.stem}_curves.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=180, bbox_inches="tight")
            print(f"saved plot: {plot_path}")
            plt.close()

            # ---------- Plot 2: entropy analysis, 1-panel (any-pred only) ----------
            fig, ax = plt.subplots(figsize=(10, 5))
            H_list = entropy_any
            correct_list = correct_any
            Hmax = math.log(K) + 0.05
            bins = np.linspace(0, Hmax, 26)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_width = bins[1] - bins[0]

            H_correct = [h for h, c in zip(H_list, correct_list) if c]
            H_wrong = [h for h, c in zip(H_list, correct_list) if not c]
            cnt_correct, _ = np.histogram(H_correct, bins=bins)
            cnt_wrong, _ = np.histogram(H_wrong, bins=bins)
            ax.bar(bin_centers, cnt_correct, width=bin_width*0.95,
                   color="#16a34a", alpha=0.85, label="correct (maj@K)", edgecolor="white", linewidth=0.4)
            ax.bar(bin_centers, cnt_wrong, width=bin_width*0.95,
                   bottom=cnt_correct, color="#dc2626", alpha=0.85, label="wrong (maj@K)",
                   edgecolor="white", linewidth=0.4)
            # Reference H lines
            for x, lbl in [(0, "H=0"), (math.log(2), "ln 2"),
                            (math.log(8), "ln 8"), (math.log(K), f"ln {K}")]:
                ax.axvline(x, color="#aaa", linestyle="--", linewidth=0.7, alpha=0.5)
                ax.text(x, ax.get_ylim()[1]*0.96, lbl, fontsize=8, color="#555",
                        ha="left" if x < Hmax/2 else "right", va="top")
            ax.set_xlabel("answer entropy H (nats)")
            ax.set_ylabel("# questions (stacked: green=correct, red=wrong)", fontsize=10)
            ax.set_title(f"answer entropy analysis — {config.get('tag','?')} (K={K}, n={len(samples)}, any-pred)",
                         fontsize=11, loc="left")
            ax.set_xlim(0, Hmax); ax.grid(axis="y", alpha=0.3)
            ax.legend(loc="upper left", fontsize=9)

            # Right axis: per-bin avg pass@1
            bin_indices = np.digitize(H_list, bins) - 1
            accs = []
            for b in range(len(bin_centers)):
                mask = bin_indices == b
                if not mask.any():
                    accs.append(np.nan); continue
                bin_correct_per_K = [s["any_correct_per_K"]/K for i, s in enumerate(samples) if mask[i]]
                accs.append(np.mean(bin_correct_per_K) * 100)
            ax2 = ax.twinx()
            ax2.plot(bin_centers, accs, "o-", color="black", markersize=5,
                     linewidth=1.8, alpha=0.85, label="avg pass@1 in bin")
            ax2.set_ylabel("avg pass@1 in bin (%)", color="black", fontsize=10)
            ax2.set_ylim(0, 105)
            ax2.legend(loc="upper right", fontsize=9)
            plot_path = fp.parent / f"{fp.stem}_entropy.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=180, bbox_inches="tight", facecolor="white")
            print(f"saved entropy plot: {plot_path}")
            plt.close()

        except ImportError as e:
            print(f"(matplotlib/numpy not available, skipped plots: {e})")


if __name__ == "__main__":
    main()

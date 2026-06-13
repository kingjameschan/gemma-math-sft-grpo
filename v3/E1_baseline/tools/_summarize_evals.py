"""Summarize all v3 evals from eval_log.jsonl into a clean markdown table.

Groups by engine type, sorts by timestamp descending. Marks the LATEST per
(engine, tag, ckpt, K) combo as "current" — so when querying "what's greedy
baseline?", just take the row marked CURRENT.

Usage:
  python3 v3/tools/_summarize_evals.py                  # full table
  python3 v3/tools/_summarize_evals.py --filter base    # only base ckpt
  python3 v3/tools/_summarize_evals.py --engine ds-cot  # only greedy CoT
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVAL_LOG = ROOT / "v3" / "shared" / "eval_log.jsonl"


def load_rows():
    if not EVAL_LOG.exists():
        return []
    with open(EVAL_LOG) as f:
        return [json.loads(line) for line in f]


def signature(d):
    """Unique signature for an experiment: same sig = same experiment, just re-run."""
    return (
        d.get("engine", ""),
        d.get("tag", d.get("model", "")),
        d.get("ckpt", "base"),
        d.get("K", 1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=None, help="filter by engine (e.g. ds-cot, ds-tir, pass_at_k)")
    ap.add_argument("--filter", default=None, help="substring match on tag")
    ap.add_argument("--all", action="store_true", help="show all rows (default: only CURRENT per signature)")
    args = ap.parse_args()

    rows = load_rows()
    if not rows:
        print("(no eval_log entries)")
        return

    # Group by signature; mark latest as CURRENT
    by_sig = defaultdict(list)
    for d in rows:
        by_sig[signature(d)].append(d)
    for sig, group in by_sig.items():
        group.sort(key=lambda d: d["timestamp"], reverse=True)
        for i, d in enumerate(group):
            d["_status"] = "CURRENT" if i == 0 else "older"

    # Filter
    filtered = []
    for d in rows:
        if args.engine and args.engine not in d.get("engine", ""):
            continue
        if args.filter and args.filter not in d.get("tag", ""):
            continue
        if not args.all and d.get("_status") != "CURRENT":
            continue
        filtered.append(d)

    # Group output by engine
    by_engine = defaultdict(list)
    for d in filtered:
        by_engine[d.get("engine", "?")].append(d)

    print(f"# v3 eval_log summary ({len(filtered)}/{len(rows)} rows shown)\n")

    for engine in sorted(by_engine):
        group = sorted(by_engine[engine], key=lambda d: d["timestamp"], reverse=True)
        print(f"## {engine}\n")
        if engine == "vllm-ds-cot":
            print("| status | timestamp | tag | ckpt | numeric | boxed | output |")
            print("|---|---|---|---|---:|---:|---|")
            for d in group:
                print(f"| {d.get('_status','?')} | {d['timestamp']} | {d.get('tag','?')} | "
                      f"{d.get('ckpt','base')} | {d.get('numeric_accuracy',0)*100:.2f}% | "
                      f"{d.get('boxed_accuracy',0)*100:.2f}% | `{d.get('output','?')}` |")
        elif engine == "vllm-ds-tir":
            print("| status | timestamp | tag | ckpt | numeric | boxed | exec_use | output |")
            print("|---|---|---|---|---:|---:|---:|---|")
            for d in group:
                exec_rate = d.get("exec_usage_rate", 0)
                print(f"| {d.get('_status','?')} | {d['timestamp']} | {d.get('tag','?')} | "
                      f"{d.get('ckpt','base')} | {d.get('numeric_accuracy',0)*100:.2f}% | "
                      f"{d.get('boxed_accuracy',0)*100:.2f}% | "
                      f"{exec_rate*100:.1f}% | `{d.get('output','?')}` |")
        elif "pass_at_k" in engine:
            print("| status | timestamp | tag | ckpt | K | pass@1 | pass@K | maj@K_n | maj@K_b | output |")
            print("|---|---|---|---|---:|---:|---:|---:|---:|---|")
            for d in group:
                K = d.get("K", "?")
                p1 = d.get(f"pass@1", 0)
                pK = d.get(f"pass@{K}", 0)
                maj_n = d.get(f"maj@{K}_numeric", 0)
                maj_b = d.get(f"maj@{K}_boxed", 0)
                print(f"| {d.get('_status','?')} | {d['timestamp']} | {d.get('tag','?')} | "
                      f"{d.get('ckpt','base')} | {K} | {p1*100:.2f}% | {pK*100:.2f}% | "
                      f"{maj_n*100:.2f}% | {maj_b*100:.2f}% | `{d.get('output','?')}` |")
        else:
            for d in group:
                print(f"- {d.get('_status','?')} {d['timestamp']} | {d.get('tag','?')} | {d}")
        print()


if __name__ == "__main__":
    main()

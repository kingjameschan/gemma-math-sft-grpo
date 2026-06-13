"""Freeze E1 baseline summary: length stats + truncation rates by bucket.

Reads:
  - pass_at_k JSON (K=64 raw)
  - difficulty_labels JSONL

Writes:
  - v3/outputs/e1_baseline_summary.json
    {
      "config": {...},
      "length_stats": {overall + per-bucket avg/p50/p75/p99/p99.5 etc.},
      "truncation_rates": {overall + per-bucket},
      "metrics": {greedy + sampling pass/maj/entropy},
    }

Usage:
  python3 v3/tools/_make_e1_summary.py path/to/pass_at_k.json
"""
import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
BASE_MODEL = ROOT / "models" / "gemma-2-2b-it"
LABELS_FILE = ROOT / "v3" / "shared" / "data" / "gsm8k" / "test_difficulty_labels.jsonl"
OUT_FILE = ROOT / "v3" / "E1_baseline" / "outputs" / "e1_baseline_summary.json"
EVAL_LOG = ROOT / "v3" / "shared" / "eval_log.jsonl"


def percentiles(arr):
    arr = np.asarray(arr)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "p50": int(np.percentile(arr, 50)),
        "p75": int(np.percentile(arr, 75)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "p99_5": int(np.percentile(arr, 99.5)),
        "p99_9": int(np.percentile(arr, 99.9)),
        "max": int(arr.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pass_at_k_json")
    args = ap.parse_args()

    fp = Path(args.pass_at_k_json)
    d = json.load(open(fp))
    samples = d["samples"]
    K = d["config"]["K"]
    tag = d["config"]["tag"]
    n_q = len(samples)

    # Load difficulty labels
    labels = {}
    with open(LABELS_FILE) as f:
        for line in f:
            r = json.loads(line)
            labels[r["question_idx"]] = r["bucket"]

    # Tokenize all responses, compute lengths
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    print("[tokenize] computing lengths...")
    lens_overall = []
    lens_by_bucket = defaultdict(list)
    n_truncated_at_1024_overall = 0
    n_truncated_at_1024_by_bucket = defaultdict(int)
    n_resp_by_bucket = defaultdict(int)
    questions_with_any_truncated = set()
    questions_with_any_truncated_by_bucket = defaultdict(set)

    TRUNCATE_THRESHOLD = 1020  # ≥ 1020 tok = effectively at 1024 ceiling

    for i, s in enumerate(samples):
        bucket = labels.get(i, "?")
        for resp in s["responses"]:
            n_tok = len(tok.encode(resp, add_special_tokens=False))
            lens_overall.append(n_tok)
            lens_by_bucket[bucket].append(n_tok)
            n_resp_by_bucket[bucket] += 1
            if n_tok >= TRUNCATE_THRESHOLD:
                n_truncated_at_1024_overall += 1
                n_truncated_at_1024_by_bucket[bucket] += 1
                questions_with_any_truncated.add(i)
                questions_with_any_truncated_by_bucket[bucket].add(i)

    n_resp_overall = len(lens_overall)

    # Length stats
    length_stats = {
        "overall": percentiles(lens_overall),
        "by_bucket": {b: percentiles(lens_by_bucket[b]) for b in ["Easy", "Medium", "Hard"]},
    }

    # Truncation rates (response-level + question-level)
    bucket_q_counts = Counter(labels.values())
    truncation_rates = {
        "overall": {
            "n_responses": n_resp_overall,
            "n_truncated_responses": n_truncated_at_1024_overall,
            "response_truncation_rate": round(n_truncated_at_1024_overall / n_resp_overall, 5),
            "n_questions": n_q,
            "n_questions_with_any_truncated": len(questions_with_any_truncated),
            "question_truncation_rate": round(len(questions_with_any_truncated) / n_q, 5),
        },
        "by_bucket": {},
    }
    for b in ["Easy", "Medium", "Hard"]:
        n_resp = n_resp_by_bucket[b]
        n_trunc = n_truncated_at_1024_by_bucket[b]
        n_q_b = bucket_q_counts[b]
        n_q_trunc = len(questions_with_any_truncated_by_bucket[b])
        truncation_rates["by_bucket"][b] = {
            "n_responses": n_resp,
            "n_truncated_responses": n_trunc,
            "response_truncation_rate": round(n_trunc / max(1, n_resp), 5),
            "n_questions": n_q_b,
            "n_questions_with_any_truncated": n_q_trunc,
            "question_truncation_rate": round(n_q_trunc / max(1, n_q_b), 5),
        }

    # Pull greedy + sampling metrics from eval_log
    metrics = {"sampling": {}, "greedy": {}}
    if EVAL_LOG.exists():
        with open(EVAL_LOG) as f:
            rows = [json.loads(l) for l in f]
        latest_greedy = max(
            (r for r in rows if r.get("engine") == "vllm-ds-cot" and r.get("ckpt") == "base"),
            key=lambda r: r["timestamp"], default=None,
        )
        latest_pass = max(
            (r for r in rows if r.get("engine") == "vllm-pass_at_k" and r.get("ckpt") == "base"
             and r.get("K") == K),
            key=lambda r: r["timestamp"], default=None,
        )
        if latest_greedy:
            metrics["greedy"] = {
                "boxed_rate": latest_greedy.get("boxed_rate"),
                "boxed_accuracy": latest_greedy.get("boxed_accuracy"),
                "numeric_accuracy": latest_greedy.get("numeric_accuracy"),
                "source": latest_greedy["output"],
            }
        if latest_pass:
            metrics["sampling"] = {k: v for k, v in latest_pass.items()
                                    if k.startswith("pass@") or k.startswith("maj@")}
            metrics["sampling"]["source"] = latest_pass["output"]

    # Compose summary
    summary = {
        "stage": "E1_baseline",
        "tag": tag,
        "K": K,
        "n_questions": n_q,
        "n_responses_total": n_resp_overall,
        "source_eval": str(fp.relative_to(ROOT)),
        "labels_file": str(LABELS_FILE.relative_to(ROOT)),
        "metrics": metrics,
        "length_stats": length_stats,
        "truncation_rates": truncation_rates,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved: {OUT_FILE}")

    # Pretty print key numbers
    print()
    print("=== E1 baseline summary ===")
    print()
    print("Length stats (tokens):")
    print(f"  {'bucket':>8}  {'n':>7}  {'mean':>5}  {'p50':>4}  {'p99':>4}  {'p99.5':>5}  {'max':>5}")
    for label, st in [("overall", length_stats["overall"])] + [
        (b, length_stats["by_bucket"][b]) for b in ["Easy", "Medium", "Hard"]
    ]:
        print(f"  {label:>8}  {st['n']:>7}  {st['mean']:>5.0f}  {st['p50']:>4}  {st['p99']:>4}  {st['p99_5']:>5}  {st['max']:>5}")

    print()
    print("Truncation rates (≥1020 tok of max_new_tokens=1024):")
    for label in ["overall"] + ["Easy", "Medium", "Hard"]:
        if label == "overall":
            tr = truncation_rates["overall"]
        else:
            tr = truncation_rates["by_bucket"][label]
        print(f"  {label:>8}  resp_trunc: {tr['n_truncated_responses']:>3}/{tr['n_responses']:>5}"
              f" ({tr['response_truncation_rate']*100:.3f}%)  "
              f"q_trunc: {tr['n_questions_with_any_truncated']:>3}/{tr['n_questions']:>4}"
              f" ({tr['question_truncation_rate']*100:.3f}%)")


if __name__ == "__main__":
    main()

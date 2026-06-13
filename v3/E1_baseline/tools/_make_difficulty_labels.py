"""Generate v3 difficulty labels from base IT model's pass@K eval.

Bucketing scheme (Option A — extreme thresholds for clean narrative):
  - Easy   : pass@1 >= 0.9   (model already robust)
  - Medium : 0.1 < pass@1 < 0.9   (model uncertain; sharpening target)
  - Hard   : pass@1 <= 0.1   (model robust wrong; capability-extension target)

Output: v3/data/gsm8k/test_difficulty_labels.jsonl
  one row per GSM8K test question with: question_idx, question, gold,
  pass1_base, num_correct_per_K, K, bucket.

This file is the FROZEN reference for all v3 experiments — every method's
behavior analysis on Easy/Medium/Hard joins against it.

Usage:
  python3 v3/tools/_make_difficulty_labels.py path/to/pass_at_k_<TS>/<tag>_k64.json
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "v3" / "shared" / "data" / "gsm8k"
OUT_FILE = OUT_DIR / "test_difficulty_labels.jsonl"


def assign_bucket(pass1: float) -> str:
    if pass1 >= 0.9:
        return "Easy"
    if pass1 <= 0.1:
        return "Hard"
    return "Medium"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pass_at_k_json", help="path to pass_at_k_<TS>/<tag>_k<K>.json")
    args = ap.parse_args()

    fp = Path(args.pass_at_k_json)
    d = json.load(open(fp))
    samples = d["samples"]
    K = d["config"]["K"]
    n = len(samples)

    rows = []
    bucket_count = {"Easy": 0, "Medium": 0, "Hard": 0}
    for i, s in enumerate(samples):
        c = s["any_correct_per_K"]
        pass1 = c / K
        bucket = assign_bucket(pass1)
        bucket_count[bucket] += 1
        rows.append({
            "question_idx": i,
            "question": s["question"],
            "gold": s["gold"],
            "pass1_base": round(pass1, 4),
            "num_correct_per_K": c,
            "K": K,
            "bucket": bucket,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"=== Difficulty labels (Option A: extreme thresholds) ===")
    print(f"  Source: {fp}")
    print(f"  Base K = {K}")
    print(f"  N = {n}")
    print()
    print(f"  Easy   (pass@1 >= 0.9): {bucket_count['Easy']:>4}  ({bucket_count['Easy']/n*100:.1f}%)")
    print(f"  Medium (0.1 < p < 0.9): {bucket_count['Medium']:>4}  ({bucket_count['Medium']/n*100:.1f}%)")
    print(f"  Hard   (pass@1 <= 0.1): {bucket_count['Hard']:>4}  ({bucket_count['Hard']/n*100:.1f}%)")
    print()
    print(f"  saved: {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

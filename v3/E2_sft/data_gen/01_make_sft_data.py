"""Convert GSM8K train.jsonl → v3 SFT data in HF prompt+completion format.

Source:  data/gsm8k/train.jsonl  (7473 samples, original GSM8K format)
Outputs: v3/shared/data/sft/train.jsonl (D_rl, 6973)
         v3/shared/data/sft/dev.jsonl   (D_dev, 500)

Format conversion per sample:
  question → prompt user message + DS-CoT suffix
  answer:
    - keep <<x=y>> inline computation annotations (GSM8K original style)
    - replace "#### N" → "\\boxed{N}"   (DS-CoT eval protocol)

Output schema (TRL prompt+completion conversational, native chat template):
  {
    "prompt": [
      {"role": "user", "content": "{question}\\nPlease reason step by step..."}
    ],
    "completion": [
      {"role": "assistant", "content": "{answer with <<>> kept, #### → \\\\boxed{}}"}
    ]
  }

TRL applies the tokenizer's native chat_template to prompt and completion
separately, automatically masking prompt tokens from the loss — no
template patching needed.

Train/dev split is deterministic (seed=42) over indices, so D_rl ∩ D_dev = ∅
and the same seed always produces the same split.

Usage:
  python3 v3/data_gen/01_make_sft_data.py
  python3 v3/data_gen/01_make_sft_data.py --n_dev 500 --seed 42
"""
import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC_FILE = ROOT / "data" / "gsm8k" / "train.jsonl"
OUT_DIR = ROOT / "v3" / "shared" / "data" / "sft"

# DSMath README L196 — exact text matching v3/eval/01_eval_ds_cot.py
USER_INSTRUCTION_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)

# `#### N` (with trailing whitespace) → match the canonical GSM8K final-answer marker
HASH_ANSWER_RE = re.compile(r"####\s*(\S+)\s*$")


def convert_answer(raw_answer: str) -> str:
    """Convert GSM8K gold answer to v3 SFT gold:
       - Keep <<x=y>> annotations as-is
       - Replace '#### N' (anywhere, typically end) with '\\boxed{N}'
    """
    m = HASH_ANSWER_RE.search(raw_answer)
    if m is None:
        raise ValueError(f"no '#### N' marker in answer: {raw_answer!r}")
    final_num = m.group(1)
    body = raw_answer[: m.start()].rstrip()  # everything before #### marker
    return f"{body}\n\\boxed{{{final_num}}}"


def make_prompt_completion(question: str, answer: str) -> dict:
    user_content = question + USER_INSTRUCTION_SUFFIX
    assistant_content = convert_answer(answer)
    return {
        "prompt": [
            {"role": "user", "content": user_content},
        ],
        "completion": [
            {"role": "assistant", "content": assistant_content},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC_FILE))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--n_dev", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read source
    raw = []
    with open(src) as f:
        for line in f:
            raw.append(json.loads(line))
    print(f"[load] {len(raw)} samples from {src}")

    # 2. Split: shuffle indices with seed → first n_dev = dev, rest = train
    rng = random.Random(args.seed)
    indices = list(range(len(raw)))
    rng.shuffle(indices)
    dev_idx = set(indices[: args.n_dev])
    train_idx = [i for i in indices[args.n_dev :]]
    dev_idx_list = indices[: args.n_dev]
    print(
        f"[split] seed={args.seed} → train={len(train_idx)}, dev={len(dev_idx_list)}"
    )

    # 3. Convert + write
    n_train_ok = n_train_skip = 0
    n_dev_ok = n_dev_skip = 0
    train_path = out_dir / "train.jsonl"
    dev_path = out_dir / "dev.jsonl"

    with open(train_path, "w") as ftr, open(dev_path, "w") as fdv:
        # Iterate in original order; route by index membership
        for i, ex in enumerate(raw):
            try:
                row = make_prompt_completion(ex["question"], ex["answer"])
            except ValueError as e:
                # malformed answer (no '#### N') → skip
                if i in dev_idx:
                    n_dev_skip += 1
                else:
                    n_train_skip += 1
                continue
            line = json.dumps(row, ensure_ascii=False)
            if i in dev_idx:
                fdv.write(line + "\n")
                n_dev_ok += 1
            else:
                ftr.write(line + "\n")
                n_train_ok += 1

    print(f"[write] {train_path}: {n_train_ok} ok, {n_train_skip} skipped")
    print(f"[write] {dev_path}: {n_dev_ok} ok, {n_dev_skip} skipped")

    # 4. Sanity: print first sample of each
    print("\n=== train sample[0] ===")
    with open(train_path) as f:
        print(f.readline().rstrip())
    print("\n=== dev sample[0] ===")
    with open(dev_path) as f:
        print(f.readline().rstrip())


if __name__ == "__main__":
    main()

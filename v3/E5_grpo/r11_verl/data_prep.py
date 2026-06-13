"""Convert v3 GSM8K jsonl (TRL chat format) → verl parquet format.

verl schema (per row):
    {
        "data_source": "gsm8k",
        "prompt":      [{"role": "user", "content": "..."}],
        "ability":     "math",
        "reward_model": {"style": "rule", "ground_truth": "<number>"},
        "extra_info":  {"question": "...", "split": "train"|"dev"|"test", "index": int}
    }

Run:  python data_prep.py
Output: /mnt/d/fine-tuning/v3/E5_grpo/r11_verl/data/{train,dev,test}.parquet
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/d/fine-tuning/v3/shared/data/sft")
OUT  = Path("/mnt/d/fine-tuning/v3/E5_grpo/r11_verl/data")
OUT.mkdir(parents=True, exist_ok=True)

_BOXED = re.compile(r"\\boxed\{([^}]+)\}")


def gold_from_completion(completion_msgs):
    """Extract gold number from assistant turn's \\boxed{N} (GSM8K SFT format)."""
    text = completion_msgs[0]["content"] if completion_msgs else ""
    m = list(_BOXED.finditer(text))
    if m:
        s = m[-1].group(1).strip().replace(",", "").replace("$", "")
        return s
    # Fallback: last number
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else ""


def question_from_prompt(prompt_msgs):
    """Strip the DS-CoT suffix to recover bare question."""
    txt = prompt_msgs[0]["content"]
    return txt.split("\nPlease reason step by step")[0].strip()


def convert(in_path, split):
    rows = []
    with open(in_path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            gold = gold_from_completion(d["completion"])
            question = question_from_prompt(d["prompt"])
            rows.append({
                "data_source": "gsm8k",
                "prompt":      d["prompt"],  # keep as-is, includes DS-CoT suffix
                "ability":     "math",
                "reward_model": {"style": "rule", "ground_truth": gold},
                "extra_info":  {"question": question, "split": split, "index": i},
            })
    df = pd.DataFrame(rows)
    out = OUT / f"{split}.parquet"
    df.to_parquet(out, index=False)
    print(f"  {split}: {len(df)} rows → {out}")


if __name__ == "__main__":
    print("Converting GSM8K jsonl → verl parquet...")
    convert(ROOT / "train.jsonl", "train")
    convert(ROOT / "dev.jsonl",   "dev")
    convert(ROOT / "test.jsonl",  "test")
    print("Done.")

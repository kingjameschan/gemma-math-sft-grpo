"""Convert TRL chat-format GSM8K JSONL into verl parquet rows.

Default input:
    v3/shared/data/sft/{train,dev}.jsonl

Default output:
    v3/E5_grpo/r11_verl/data/{train,dev}.parquet

Run from any working directory:
    python v3/E5_grpo/r11_verl/data_prep.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = REPO_ROOT / "v3" / "shared" / "data" / "sft"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "v3" / "E5_grpo" / "r11_verl" / "data"
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def gold_from_completion(completion_messages: list[dict]) -> str:
    """Extract the final boxed value from the assistant completion."""
    text = completion_messages[0]["content"] if completion_messages else ""
    matches = list(BOXED_RE.finditer(text))
    if matches:
        return (
            matches[-1]
            .group(1)
            .strip()
            .replace(",", "")
            .replace("$", "")
        )

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else ""


def question_from_prompt(prompt_messages: list[dict]) -> str:
    """Recover the bare question by removing the fixed CoT suffix."""
    text = prompt_messages[0]["content"]
    return text.split("\nPlease reason step by step")[0].strip()


def convert(in_path: Path, out_path: Path, split: str) -> int:
    rows = []
    with in_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            item = json.loads(line)
            rows.append(
                {
                    "data_source": "gsm8k",
                    "prompt": item["prompt"],
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": gold_from_completion(item["completion"]),
                    },
                    "extra_info": {
                        "question": question_from_prompt(item["prompt"]),
                        "split": split,
                        "index": index,
                    },
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in args.splits:
        in_path = args.input_dir / f"{split}.jsonl"
        out_path = args.output_dir / f"{split}.parquet"
        if not in_path.is_file():
            raise FileNotFoundError(f"Missing input split: {in_path}")
        count = convert(in_path, out_path, split)
        print(f"{split}: {count} rows -> {out_path}")


if __name__ == "__main__":
    main()

"""下载 GSM8K 数据集到 data/gsm8k/"""
import json, os
from datasets import load_dataset

os.makedirs("data/gsm8k", exist_ok=True)

for split in ["train", "test"]:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    path = f"data/gsm8k/{split}.jsonl"
    with open(path, "w") as f:
        for item in ds:
            f.write(json.dumps({"question": item["question"], "answer": item["answer"]}) + "\n")
    print(f"Saved {len(ds)} samples → {path}")

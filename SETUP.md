# Setup — data & model acquisition

Weights and datasets are not committed. Recreate the expected layout below, then use the
Docker images or conda envs in the README.

```
fine-tuning/
├── models/
│   └── gemma-2-2b-it/          # base model (gated)
└── data/
    ├── gsm8k/                  # train.jsonl (7473) + test.jsonl (1319)
    └── math500_aug/            # 500-question numeric MATH slice
```

## Base model (gated)
`gemma-2-2b-it` requires accepting Google's license.

```bash
# Hugging Face (after accepting the license on the model page):
huggingface-cli download google/gemma-2-2b-it --local-dir models/gemma-2-2b-it
# or ModelScope (no gating):
modelscope download --model LLM-Research/gemma-2-2b-it --local_dir models/gemma-2-2b-it
```

## Datasets
- **GSM8K** — `openai/gsm8k` (main). Convert to the DS-CoT chat format used here
  (`{question}\nPlease reason step by step, and put your final answer within \boxed{}.`).
  The SFT training set is built by `v3/E2_sft/data_gen/01_make_sft_data.py`.
- **MATH500 (numeric slice)** — the 500-question numeric subset used for the secondary
  eval (`data/math500_aug/`).

Datasets are re-downloadable; nothing here is private. The verl RL data (parquet) is
produced from the GSM8K jsonl by `v3/E5_grpo/r11_verl/data_prep.py`.

## Environments
See the repo `README.md`:
- **train / eval** — `docker/Dockerfile.train`, `docker/Dockerfile.eval`
  (or `requirements-train.txt` / `requirements-eval.txt`).
- **RL (verl)** — `docker/Dockerfile.grpo` / `requirements-grpo.txt`.

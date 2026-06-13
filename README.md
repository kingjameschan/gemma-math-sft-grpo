# gemma-math-sft-grpo

A hands-on study of classic **post-training algorithms** on a small instruct model,
holding **model + data + eval constant** and varying only the algorithm. The aim is
source-level understanding of the SFT → DPO → GRPO/DAPO → distillation pipeline
(TRL / PEFT / vLLM internals) — a learning / interview-prep project.

> This is the **v3** redo (clean, from scratch). Model weights, datasets, and raw
> per-sample eval dumps are intentionally **not** committed — they're large and fully
> reproducible from the code here.

## Setup (held fixed across methods)
- **Base model:** `gemma-2-2b-it` (+ `gemma-2-2b` pretrain base for the distillation contrast)
- **Tasks:** GSM8K + a 500-question numeric MATH slice (`math500_aug`)
- **Eval:** DeepSeek-style 5-layer answer extraction + `math_equal`; pass@k / maj@k via vLLM (K up to 128)
- **LoRA:** r=64, α=32, all-linear, dropout=0
- **Hardware:** single RTX 5080 16 GB (local) + some cloud runs

## Experiments
| Dir | What |
|---|---|
| `E1_baseline/` | base-model DS-CoT baseline + eval harness |
| `E2_sft/` | SFT on GSM8K gold; LR / checkpoint sweep |
| `E5_grpo/` | GRPO / DAPO RL runs (verl) + ablations |
| `E6_distill/` | off-policy distillation (OpenMathInstruct-2, Llama-3.1-405B CoT) into IT **and** pretrain base; before/after pass@k |
| `shared/` | shared answer-extraction / eval utilities |

Each experiment dir holds its code (`train/`, `eval/`, `tools/`), result figures
(`outputs/*.png`), and notes (`FINDINGS.md` / `README.md`). Headline metrics live in
`outputs/eval_log.jsonl`.

## Not in the repo (regenerate from code)
`*.safetensors` adapters, datasets (`data/`), and verbose per-sample eval JSON.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QLoRA fine-tuning pipeline (SFT + DPO) for enhancing Qwen2.5-7B math reasoning on GSM8K. Target: ByteDance LLM algorithm internship portfolio project.

- **Base model**: `models/Qwen2.5-7B` (local, ~15GB)
- **Hardware**: RTX 5080 Laptop 16GB VRAM
- **Conda env**: `siren`
- **Dataset**: GSM8K — `data/gsm8k/train.jsonl` (7473) + `test.jsonl` (1319)

## Key Commands

```bash
# Activate environment
conda activate siren

# Step 1: Convert raw GSM8K → ChatML format (required before SFT v2)
python scripts/prepare_sft_data.py

# Step 2: Train SFT (v2 is current; v1 in src/ is deprecated)
python scripts/train_sft_v2.py

# Step 3: Evaluate base model baseline
python scripts/baseline_eval.py

# Step 4: Batch-evaluate all saved checkpoints
python scripts/evaluate_all_checkpoints.py

# Step 5: Multi-metric analysis report from eval jsonl files
python scripts/analyze_results.py
```

All scripts use `os.path.abspath(__file__)` for path resolution — run them from anywhere.

## Architecture

### Data flow

```
data/gsm8k/{train,test}.jsonl          # raw {"question", "answer"}
    ↓ scripts/prepare_sft_data.py
data/sft_formatted/{train,test}_sft.jsonl   # {"messages": [system/user/assistant]}
    ↓ scripts/train_sft_v2.py
checkpoints/qwen2.5-7b-sft-v2/             # LoRA adapters per step
    ↓ scripts/evaluate_all_checkpoints.py
outputs/eval_details_{step}.jsonl           # per-sample {question, gold_val, pred_val, is_correct, full_response}
    ↓ scripts/analyze_results.py
stdout report (accuracy, format rate, avg steps)
```

### Training (v2 — use this, not src/train_sft.py)

`scripts/train_sft_v2.py` is the canonical trainer. Key differences from v1:
- Uses `SFTConfig` + `assistant_only_loss=True` (loss masking via TRL ≥0.29)
- Reads data with `messages` field; TRL calls `apply_chat_template` internally
- Chat template is **patched at runtime** to inject `{% generation %}` markers around assistant content — required for loss masking to work with Qwen2.5's default template

**LoRA config**: r=16, alpha=32, dropout=0.05, targets: `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`
**Training**: batch=1, grad_accum=16, lr=1e-4, cosine schedule, 3 epochs, bf16, `paged_adamw_32bit`
**Checkpoints**: saved every 50 steps to `checkpoints/qwen2.5-7b-sft-v2/`

### Evaluation

All eval scripts share the same answer extraction logic:
```python
re.search(r'####\s*(-?\d[\d,]*\.?\d*)', text)  # fallback: last number in text
```

System prompt (English, consistent across all scripts):
> "You are a mathematical reasoning assistant. Please solve the following math problem step by step and provide the final answer at the end preceded by ####."

`evaluate_all_checkpoints.py` is idempotent — it skips steps with existing `outputs/eval_details_{step}.jsonl` and reuses cached results.

### Checkpoint layout

- `checkpoints/qwen2.5-7b-sft-gsm8k/` — v1 run (steps 200–800, final adapter)
- `checkpoints/qwen2.5-7b-sft-v2/` — v2 run (steps 50–450)
- Each checkpoint contains only the LoRA adapter (`adapter_config.json` + `adapter_model.safetensors`); base model is loaded separately from `models/Qwen2.5-7B`

## DPO Phase — Current State (2026-03-27)

### Experiment Results (full 1319-sample test set)

| Model | Accuracy | Notes |
|-------|----------|-------|
| SFT-800 | 75.4% | baseline |
| dpo_100 (v1, buggy) | 74.9% | ref model bug |
| dpo_400 (v1, buggy) | 74.7% | ref model bug |
| **dpo_400_v2 (fixed)** | **~85% (eval in progress)** | ref model fixed |

### Critical Bug Fixed in train_dpo.py

Original code used `ref_model=None` with SFT LoRA as the trainable adapter.
In PEFT mode, TRL computes ref logprobs by **disabling LoRA** — so reference was the raw base model, NOT the SFT model. This caused DPO to partially undo SFT learning.

**Fix (already applied)**: merge SFT adapter into base weights first, then add a fresh LoRA for DPO training. Now disabling LoRA → reference = SFT merged model (correct).

```python
# train_dpo.py — correct approach
model = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=False)
model = model.merge_and_unload()   # base weights now = SFT
model = get_peft_model(model, lora_config)  # fresh LoRA for DPO
ref_model = None  # disabling LoRA → SFT merged ✓
```

Training metrics after fix: loss 0.694→0.010, rewards/margins 0.036→10.12 (vs near-zero before).

### DPO Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train_dpo.py` | DPO training (v2, ref model fixed, 3 epochs, grad_accum=4) |
| `scripts/evaluate_dpo_full.py` | HF evaluation on full 1319 test set (batch_size=4, ~3h) |
| `scripts/merge_dpo_adapter.py` | Merge DPO adapter → full model (for vLLM/GGUF) |
| `scripts/error_analysis.py` | McNemar's test + error classification |
| `scripts/generate_report_figures.py` | Generate 5 figures for report |

### DPO Data

- `data/dpo/dpo_train_full.jsonl` — 839 pairs from first 4000 GSM8K train questions
- Generated by llama.cpp (Q8_0 SFT-merged model), adaptive 2-shot strategy, temperature=0.7
- ~21% yield rate (only questions where SFT gets some right and some wrong)

### Pending Tasks

1. **Wait for dpo_400_v2 full eval to finish** (running: `outputs/eval_dpo_400_v2_full.jsonl`)
   ```bash
   # Check progress
   python -c "import json; r=[json.loads(l) for l in open('outputs/eval_dpo_400_v2_full.jsonl')]; c=sum(1 for x in r if x['is_correct']); print(f'{len(r)}/1319, {c/len(r):.2%}')"
   ```

2. **Run McNemar's test** after eval completes:
   ```bash
   python scripts/error_analysis.py \
     --sft outputs/eval_sft800_full.jsonl \
     --dpo outputs/eval_dpo_400_v2_full.jsonl \
     --dpo_name dpo_400_v2
   ```

3. **Set up vLLM in WSL2** for faster future evaluations (~5-10x speedup):
   ```bash
   conda create -n vllm python=3.11 -y && conda activate vllm
   pip install vllm
   # Then merge adapter and run evaluate_vllm.py (see docs/wsl_handoff.md)
   ```

4. **Write experiment report** `docs/experiment_report.md`

### Checkpoint Locations

- `checkpoints/qwen2.5-7b-sft-v2/checkpoint-800` — SFT adapter (best checkpoint)
- `checkpoints/qwen2.5-7b-dpo/dpo_400_v2/final_adapter` — DPO v2 adapter (use this)
- `checkpoints/qwen2.5-7b-dpo/dpo_*/` — older buggy DPO variants (for comparison only)

### Evaluation Output Files

- `outputs/eval_sft800_full.jsonl` — SFT-800 full results (1319 samples)
- `outputs/eval_dpo_100_full.jsonl` — dpo_100 v1 full results
- `outputs/eval_dpo_400_full.jsonl` — dpo_400 v1 full results
- `outputs/eval_dpo_400_v2_full.jsonl` — dpo_400_v2 full results (in progress)

### HW Constraints

- RTX 5080 16GB VRAM. Training (~12-14GB) and evaluation (~8-10GB) **cannot run simultaneously** — always serial.
- Windows: use `siren` conda env for training/HF eval
- WSL2: use `vllm` conda env for vLLM inference (faster eval)

## Important Notes

- `scripts/_check_*.py` files are one-off debugging scripts; not part of the main pipeline
- `outputs/baseline_errors.jsonl` — baseline error log (raw base model, 100 samples)
- `outputs/sft_eval_results.jsonl` — v1 final eval results (different format from `eval_details_*.jsonl`)
- The v2 trainer splits the first 200 shuffled samples as eval set; remaining ~7273 are train
- **Inference consistency**: HF and llama.cpp produce different results due to merge precision. All final evaluations use HF generate for consistency.

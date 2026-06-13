# v3 — Comparative study of classic post-training algorithms

## Goal

**Hold model + data + eval constant. Vary only the post-training algorithm.** Quantify
which post-training paradigm contributes most to GSM8K performance on a small instruct
model under tight compute (16GB VRAM).

```
        SFT  ──→  RFT  ──→  online RFT  ──→  DPO  ──→  GRPO
       gold     1× sample    iter sample    pairs    full RL
        ↓         ↓             ↓            ↓         ↓
                    All on Gemma2-2B-IT, GSM8K, DS-CoT eval
```

This frames the project as **algorithmic ablation**, with each method as an
instance of a generalized policy gradient (DSMath §5.2.2 unified paradigm).

## Five algorithms

| # | Method | Data source | Objective | Key insight |
|---|---|---|---|---|
| 1 | **SFT** | GSM8K gold (7473) | NLL on (q, a*) | Imitate human reasoning |
| 2 | **RFT** | self-sample × k, keep correct (1 round) | NLL on (q, a_self) | Self-distillation, dedupes diverse correct paths |
| 3 | **online RFT** | self-sample × k from CURRENT policy (iter) | NLL on (q, a_self) per round | RFT + on-policy data refresh |
| 4 | **DPO** | (chosen, rejected) pairs | β·log[πθ(c)/πref(c)] − β·log[πθ(r)/πref(r)] | Contrastive preference, no reward model |
| 5 | **GRPO** | group-of-G samples + rule reward | PPO-style with group-mean baseline | Full RL, no critic, group advantage |

DSMath unified view: all are special cases of `∇θ log π(o|q) · GC(q,o,t,π_target)`
where `GC` (gradient coefficient) and target distribution differ by method.

## Constants (held fixed across all 5)

- **Base model**: `models/gemma-2-2b-it`
- **Dataset**: GSM8K (`data/gsm8k/train.jsonl` 7473, `test.jsonl` 1319)
- **Format**: DS-CoT — user instruction `{q}\nPlease reason step by step, and put your final answer within \boxed{}.`, no system prompt, Gemma2 native chat template
- **Eval protocol**: `v3/E1_baseline/eval/01_eval_ds_cot.py` (DS 5-layer extract, math_equal numerical compare)
- **vLLM eval params** (data-driven from baseline run):
  - `max_new_tokens = 1024`  (DS literature standard; empirical p99.5=445 → 2× buffer)
  - `max_model_len = 1280`   (= max_prompt(238) + max_new_tokens(1024) + 18 round-up; 1.6× concurrency vs DS 2048 default)
  - `gpu_memory_utilization = 0.85`
  - `stop_strings = [" \n \n \n"]`  (whitespace-tail bug guard for Gemma2-IT)
  - **Note**: `max_model_len` is the direct concurrency knob; `max_new_tokens` just bounds output length but vLLM scheduler reserves `max_model_len` per request.
- **LoRA** (locked 2026-04-29 at S1 planning):
  - `r = 64` — Schulman 2025 *LoRA Without Regret* Figure 2: r=64 first row safely in NLL saturation plateau (r=32 still on edge); ~80× capacity buffer over GSM8K SFT info content (~1.75M bits).
  - `alpha = 32` — Schulman verbatim. Gives α/r=0.5; with α fixed, optimal LR is rank-invariant per Schulman scaling fit (LoRA_pow=0).
  - `target_modules = all linear` — q/k/v/o/gate_proj/up_proj/down_proj. Schulman: "LoRA performs better when applied to all weight matrices, especially MLP".
  - `dropout = 0` — Hu 2021 LoRA paper default; Schulman uses default PEFT init.
  - Rationale: r doesn't affect optimal LR within capacity-sufficient regime (1/r prefactor normalizes update magnitude); fixed across all 5 methods so cross-method comparison varies only the algorithm.
- **Hardware**: RTX 5080 16GB
- **Training env**: `siren` (TRL 0.29) — switch to `train-env` if padding_free needed for DPO
- **Eval env**: `vllm-env`
- **Primary metric**: `numeric_accuracy` (DS literature standard, 5-layer chain)
- **Secondary**: `boxed_accuracy` (strict boxed format)

## Established v3 Baseline (2026-04-27)

Gemma2-2B-IT base under DS-CoT protocol:

| Eval | boxed_rate | boxed_accuracy | numeric_accuracy | program_accuracy |
|---|---:|---:|---:|---:|
| CoT | 45.56% | 30.10% | **61.33%** | — |
| TIR | 46.17% | 31.08% | 61.33% | 0.00% |

**Key finding**: TIR ≡ CoT for Gemma2-IT. `exec_usage_rate = 0%` — Gemma2-IT NEVER
spontaneously uses ` ```python``` ` blocks. DSMath-Instruct's TIR ability comes from
SFT-on-tool-use-traces, which Gemma2-IT lacks. Therefore all 5 algorithms
optimize CoT-only performance; tool-use is out of scope for v3 comparison.

## Stage Plan (incremental — to be confirmed stage by stage)

| Stage | Method | Data prep | Train | Eval | Status |
|---|---|---|---|---|---|
| S0 | baseline | — | — | DS-CoT base 61.33% | DONE |
| S1 | SFT | convert v2 train.jsonl gold → DS format | 1 SFT run, sweep ckpts | sweep eval | next |
| S2 | RFT | 1× sample IT base with DS prompt, keep correct, dedup | 1 SFT-on-correct run | sweep eval | |
| S3 | online RFT | resample from S2 model → train → resample → train (~3 rounds) | iterative SFT-on-fresh-correct | sweep eval per round | |
| S4 | DPO | sample IT base, pair (boxed-correct, boxed-wrong) | DPO single run | sweep eval | |
| S5 | GRPO | rule reward (boxed_accuracy), group_size=8 | GRPO single run | sweep eval | |
| S6 | comparison plot | aggregate all results, single chart | — | — | |

## Layout

```
v3/
├── eval/        — DS-CoT eval scripts (single + sweep + maj@k)
│   ├── 01_eval_ds_cot.py  — single-ckpt CoT eval (done)
│   └── 02_eval_ds_tir.py  — TIR eval (done; TIR≡CoT for Gemma2-IT)
├── train/       — SFT/DPO/GRPO with DS-style prompt
├── data_gen/    — sampling scripts (no system prompt + DS suffix)
├── data/        — DS-format training data
├── checkpoints/ — LoRA adapters
├── outputs/     — eval results, figures, eval_log
├── tools/       — plotting / analysis
└── refs/        — DeepSeekMath (symlink)
```

## Reuse from v2

- Models, raw datasets, and refs are shared dirs.
- v2 ckpts are NOT reusable (PE+`####N` format mismatch with v3 DS-CoT).
- v2 sampling-pair data (`v2/data/dpo/`) is NOT reusable (PE-format prompts).
- v2 narrative figures stay; v3 will generate its own algorithm-comparison plots.

## Versions (inherited from v2)

torch 2.10+cu128, transformers 4.57.6, peft 0.18.1, trl 0.29.0, datasets 4.7.0

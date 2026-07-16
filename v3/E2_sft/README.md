# E2 — Supervised Fine-Tuning

E2 studies how answer-conditioned SFT changes finite-sampling coverage and probability concentration, using the same Gemma-2-2B-IT Base and evaluation family as E1.

## Selected run

- LoRA: rank `64`, alpha `32`, all-linear, dropout `0`
- Analyzed checkpoint: learning rate `5e-4`, checkpoint `130`
- Primary comparison: Base versus SFT on GSM8K (1,319) and MATH-500-aug (500)
- Reported sampling budget: up to `K=128`

At the analyzed checkpoint, high-K `pass@K` rises slightly while `pass@1` and `maj@K` fall substantially. The dashboard diagnostics are consistent with non-selective distribution dilution: SFT weakens existing answer modes, including correct modes on questions that Base already solves reliably.

## Retained evidence chain

| Path | Purpose |
|---|---|
| `data_gen/01_make_sft_data.py` | Builds the supervised GSM8K chat-format data |
| `train/01_sft.py` | TRL/PEFT LoRA training entry point |
| `tools/_plot_step130_combined.py` | Selected-checkpoint L1–L10 diagnostic dashboard |
| `outputs/lr5e-4_step130_combined.png` | Primary rendered evidence |

The original LR sweep, checkpoint-by-checkpoint plots, and large per-sample dumps remain local but are intentionally absent from the recruiter-facing branch. The result is a selected-run behavioral analysis, not a claim that this SFT recipe is globally optimal.

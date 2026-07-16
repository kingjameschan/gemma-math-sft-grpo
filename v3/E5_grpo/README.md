# E5 — DAPO / GRPO Behavior Diagnostics

E5 tests whether short-horizon RL with verifiable rewards primarily expands observed solution coverage or redistributes probability over answer modes already reachable by Gemma-2-2B-IT.

## Selected runs

| Method | Run | Selected checkpoint | Role in the study |
|---|---|---:|---|
| DAPO | R15 | checkpoint 15 | Main RL diagnostic |
| GRPO | clean R16 | step 42 | Cross-algorithm robustness check |

The two runs differ in training length, seed, and hyperparameters, so they are not used to rank DAPO against GRPO. Both are compared with Base through complete curves at common `k≤64`.

## Main observations

- `pass@1` and `maj@K` improve, while maximum-K `pass@K` remains comparable to or slightly below Base within the observed range
- Correct Base-dominant answer modes gain probability mass; incorrect Base-dominant modes lose mass
- Medium-difficulty questions contribute most of the gain; Easy is close to saturation and Hard changes less
- Same-chain PPL places RL-generated chains in a high-likelihood region under Base, while Claude Opus 4.7 / Gemini 3.1 Pro chains score substantially higher; this supports distributional compatibility rather than proving identical sampling ability

## Retained evidence chain

| Path | Purpose |
|---|---|
| `r11_verl/data_prep.py` | Converts chat-format GSM8K data to verl parquet rows |
| `r15_dapo/run_dapo_r15.sh` / `reward_judge_r15.py` | DAPO launch and verifiable reward |
| `r16_grpo_clean/run_grpo_r16_clean.sh` / `reward_judge.py` | Clean GRPO launch and verifiable reward |
| `eval/01_grpo_dev_eval.py` | Development checkpoint evaluation |
| `tools/_plot_dapo_ck15_combined.py` | DAPO L1–L10 dashboard logic |
| `tools/_plot_r16_step42_deep.py` | GRPO L1–L10 dashboard logic |
| `outputs/k64_dapo_ck15/dapo_ck15_combined.png` | Primary DAPO figure |
| `outputs/k64_r16_step42/r16_step42_combined.png` | Primary GRPO figure |
| `tools/yue_ppl_8panel_selfppl.py` | Forward-pass same-chain PPL computation |
| `tools/yue_ppl_8panel_selfppl_plot.py` | Deterministic plot regeneration from the compact result |
| `outputs/yue_ppl_analysis/ppl_8panel_selfppl_results.json` | Compact per-chain PPL values |
| `outputs/yue_ppl_analysis/yue_8panel_selfppl.png` | Same-chain PPL figure |

RL training used verl + Ray + vLLM on cloud L40S/L20 GPUs. The root Dockerfile is a reconstructed environment recipe; exact reruns also require model weights, adapters, generated chain pools, and site-specific storage paths that are intentionally not committed.

## Scope boundary

This is a single-model, selected-checkpoint empirical diagnosis on GSM8K and a numerically verifiable MATH subset. Finite-K coverage is not an absolute capability boundary, and the four-problem PPL study is a mechanism probe rather than a population estimate.

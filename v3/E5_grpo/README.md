# E5 — RL post-training (DAPO + GRPO) via verl

On-policy RL on `gemma-2-2b-it` with the [verl](https://github.com/volcengine/verl)
framework, LoRA (r=64), rule/judge reward. Two headline runs are kept here:

| Run | Method | Notes |
|---|---|---|
| **R15** (`r15_dapo/`) | **DAPO** | best DAPO config — Clip-Higher + Dynamic Sampling + Overlong Buffer + token-level loss; best checkpoint **ck-15** |
| **R16** (`r16_grpo/`, `r16_grpo_clean/`) | **GRPO** | group-relative PPO, group_size=8, no critic |

> The intermediate/failed DAPO iterations (R11 verl bring-up, R12–R14) and the
> incomplete long rerun (R17) are kept locally but excluded from this repo.

## Layout
| Path | What |
|---|---|
| `train/01_grpo.py`, `train/02_grpo_judge.py` | GRPO training entry points |
| `r15_dapo/`, `r16_grpo*/` | per-run launch scripts + reward judge + ckpt hooks |
| `eval/01_grpo_dev_eval.py` | dev-set selection eval |
| `audit/judge_audit.py` | reward-judge auditing |
| `tools/` | pass@k / maj@k curves, mode-mass migration, ppl-evolution, paper-style panels |
| `outputs/` | result figures + `eval_log.jsonl` |

## Key figures (`outputs/`)
- `figure2_three_methods_passk_majk.png` — base vs SFT vs DAPO, pass@k + maj@k (the headline comparison)
- `k64_dapo_ck15/` — DAPO ck-15 deep dive (pass@1 mechanism, transition matrices, entropy)
- `k64_r16_step42/` — GRPO R16 step-42 combined panel
- `mode_mass_delta/` — base-anchored Δ-mass migration for SFT / DAPO / GRPO
- `gold_ppl/`, `*_ppl_*_evolution*` — gold-answer perplexity evolution over RL steps

## Results
DAPO (R15, ck-15) lifts MATH500 pass@1 (K=64) over base (≈29.8% → 32.7%) and recovers
single-shot accuracy that plain SFT sacrifices. Full curves: `figure2_*` + the `outputs/`
figures; per-run metrics in `eval_log.jsonl`.

## Environment
RL ran in a verl + ray + vLLM Docker image on cloud GPUs (L40S / L20). The exact pinned
stack is captured in `docker/Dockerfile.grpo` / `requirements-grpo.txt` (repo root); see
also `r11_verl/HANDOVER.md` for the bring-up history and known issues.

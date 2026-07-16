# RLVR Training Behavior and Reasoning-Distribution Diagnostics

> **Gemma-2-2B-IT · GSM8K · MATH-500-aug · SFT / DAPO / GRPO**
>
> An empirical replication and diagnostic study of *why RL works* in verifiable-reward reasoning.

Inspired by the divergence between `maj@K` and `pass@K` after RL reported in [DeepSeekMath](https://arxiv.org/abs/2402.03300), this project asks:

> **Does RLVR expand the set of correct reasoning paths a model can reach, or does it primarily redistribute probability mass over paths already present in the base policy?**

Using a shared base model and evaluation pipeline, I trained SFT, DAPO, and GRPO policies and analyzed more than final-answer accuracy: `pass@K / maj@K`, question-level state transitions, difficulty buckets, answer-mode probability mass, and same-chain perplexity.

## TL;DR

- **SFT sharply changes the relationship between coverage and likelihood mass.** On GSM8K / MATH, `pass@1` drops by **18.6 / 10.1 pp**, while `pass@128` rises by **1.5 / 2.6 pp** and `maj@128` falls by **6.1 / 7.4 pp**.
- **DAPO and GRPO recover and improve single-sample accuracy and majority voting, without raising the observed maximum-K coverage.** At matched K, RL `pass@K` is comparable to or slightly below Base.
- **The evidence is more consistent with probability redistribution.** RL shifts mass toward existing correct answer modes and away from incorrect modes; the PPL probe also places RL chains in a high-likelihood region of the Base model.
- **Scope matters.** This is an empirical result for `Gemma-2-2B-IT + GSM8K + MATH-500-aug + the selected checkpoints/seeds`, not a universal proof that RL can never expand capability.

## Headline Results

Each cell reports **`pass@1 / pass@K / maj@K`**. `K` is the largest value actually reported for that run; sampling budgets are not artificially presented as identical.

| Method | GSM8K full test set (n=1,319) | MATH-500-aug (n=500) |
|---|---:|---:|
| **Base** | K=128: **61.3 / 94.8 / 69.7%** | K=128: **28.4 / 79.4 / 38.0%** |
| **SFT** `lr=5e-4, ck130` | K=128: **42.8 / 96.4 / 63.6%** | K=128: **18.3 / 82.0 / 30.6%** |
| **DAPO** `R15, ck15` | K=64: **65.2 / 91.6 / 71.6%** | K=64: **31.0 / 73.7 / 40.0%** |
| **GRPO** `R16, step42` | K=64: **66.6 / 92.3 / 73.4%** | K=128: **33.3 / 78.4 / 43.1%** |

> **Protocol notes**
> - MATH-500-aug is a fixed 500-problem, numerically verifiable set: the legacy 293-problem `math500_numeric` set plus 207 additional MATH problems, with no duplicate problem text. Selection is based on whether the final answer can be checked by a deterministic rule, not on whether any model in this study answers the problem correctly. The canonical local JSONL has SHA-256 `c323818f84a46810de4f3afd40180b12786fe5bd6a7e9aac0b62e520c20db02d`.
> - The DAPO MATH pool combines two K=64 sampling pools, while the combined dashboard reports only through `k=64`.
> - Cross-method comparisons should use the complete curves at common `k≤64`, rather than compare endpoints with different K. DAPO and GRPO also differ in steps, seeds, and hyperparameters, so this project does not rank one algorithm above the other.
> - Headline values follow the frozen dashboard pipeline, which recomputes curve correctness from normalized extracted answers. The canonical evaluator uses `math_equal`; evaluator headers and dashboard endpoints should therefore not be mixed as if they were one metric stream. A paper-grade rerun should regenerate every panel from evaluator-owned correctness flags.
> - Values are point estimates; paired bootstrap significance tests are not yet included. Under a worst-case question-level variance approximation, the upper bound on the 95% CI half-width is roughly 4.4 pp for n=500 and 2.7 pp for n=1,319. Small differences are therefore treated as descriptive trends.

## Evidence 1: `pass@K`–`maj@K` Divergence

`pass@K` measures whether at least one of K samples is correct and acts as a finite-sampling coverage proxy. `maj@K` measures whether the majority answer is correct and is more sensitive to where probability mass concentrates across answer modes.

- GSM8K at `k=64`: Base / DAPO / GRPO `pass@64 = 93.15 / 91.58 / 92.34%`
- MATH: Base / DAPO `pass@64 = 74.79 / 73.73%`; Base / GRPO `pass@128 = 79.40 / 78.40%`

The main RL gain therefore appears in `pass@1` and `maj@K`, rather than in expansion of problem coverage within the observed K range.

### Combined Diagnostic Dashboards

Each dashboard connects macro K-curves, difficulty and state transitions, per-question pass rates, and Base-anchored answer-mode mass. The source figures are approximately 3k×8k, so they are collapsible to keep the landing page readable; click an image to open it at full resolution.

<details>
<summary><strong>SFT — slightly higher high-K coverage, but lower pass@1 and maj@K under non-selective distribution dilution</strong></summary>
<p align="center">
  <a href="v3/E2_sft/outputs/lr5e-4_step130_combined.png"><img src="v3/E2_sft/outputs/lr5e-4_step130_combined.png" width="800" alt="SFT L1-L10 combined dashboard on GSM8K and MATH-500-aug"></a>
</p>
<p align="center"><sub>Base vs SFT lr=5e-4 ck130; GSM8K n=1,319 and MATH-500-aug n=500</sub></p>
</details>

<details open>
<summary><strong>DAPO — +3.2 pp on correct dominant modes and -7.8 pp on incorrect dominant modes, consistent with probability redistribution</strong></summary>
<p align="center">
  <a href="v3/E5_grpo/outputs/k64_dapo_ck15/dapo_ck15_combined.png"><img src="v3/E5_grpo/outputs/k64_dapo_ck15/dapo_ck15_combined.png" width="800" alt="DAPO R15 checkpoint 15 L1-L10 combined dashboard on GSM8K and MATH-500-aug"></a>
</p>
<p align="center"><sub>Base vs DAPO R15 ck15; the combined dashboard reports through K=64</sub></p>
</details>

<details>
<summary><strong>GRPO — +3.1 pp on correct dominant modes and -16.1 pp on incorrect dominant modes, providing a cross-algorithm robustness check</strong></summary>
<p align="center">
  <a href="v3/E5_grpo/outputs/k64_r16_step42/r16_step42_combined.png"><img src="v3/E5_grpo/outputs/k64_r16_step42/r16_step42_combined.png" width="800" alt="Clean GRPO R16 step 42 L1-L10 combined dashboard on GSM8K and MATH-500-aug"></a>
</p>
<p align="center"><sub>Base vs clean GRPO R16 step42; used as a robustness check, not as an algorithm ranking against DAPO</sub></p>
</details>

## Evidence 2: Question-Level Migration and Answer Modes

Questions are grouped by Base state, difficulty, and dominant answer mode. The analysis then tracks per-question pass-rate changes and Base-anchored mode-mass migration.

- SFT does not simply create a stable wrong attractor. The evidence instead supports **distribution dilution**: existing answer modes are weakened non-selectively, with the largest damage on Easy problems.
- DAPO / GRPO increase average probability mass on the correct Base dominant mode by approximately **3.2 / 3.1 pp**, while reducing mass on the incorrect Base dominant mode by approximately **7.8 / 16.1 pp**.
- Most improvement comes from the Medium bucket. Easy is near saturation, while gains on Hard problems remain limited.

The compact [SFT note](v3/E2_sft/README.md) and [RL note](v3/E5_grpo/README.md) record the retained checkpoints, primary figures, and interpretation boundaries.

## Evidence 3: Same-Chain PPL Probe

<p align="center">
  <a href="v3/E5_grpo/outputs/yue_ppl_analysis/yue_8panel_selfppl.png"><img src="v3/E5_grpo/outputs/yue_ppl_analysis/yue_8panel_selfppl.png" width="760" alt="Same-chain perplexity comparison across Base, SFT, DAPO, GRPO, and external reasoning chains"></a>
</p>
<p align="center"><sub>A same-chain PPL mechanism probe over four problems, split into correct and incorrect chains; click for the full-resolution figure</sub></p>

For four selected problems evaluated under correct/incorrect chain groupings:

- `PPL_base(Y_DAPO)=1.182` vs `PPL_DAPO(Y_DAPO)=1.132`
- `PPL_base(Y_GRPO)=1.168` vs `PPL_GRPO(Y_GRPO)=1.103`
- Base assigns substantially higher PPL to external Claude Opus 4.7 / Gemini 3.1 Pro reasoning chains, with aggregate medians of `2.819 / 2.540`

RL-generated chains are therefore not strongly out-of-distribution for Base, which provides supporting evidence that RL amplifies behavior already assigned meaningful likelihood by the base policy. Similar PPL only establishes distributional compatibility; **it does not prove that Base can reliably sample the same correct chains**. With only four selected problems, this result is a mechanism probe rather than a population-level statistical conclusion.

## Experimental Setup

| Item | Setting |
|---|---|
| Base | `google/gemma-2-2b-it` |
| Tasks | GSM8K test, 1,319 problems (ID); MATH-500-aug, 500 numerically verifiable problems (OOD) |
| Methods | SFT with TRL/PEFT; DAPO and clean GRPO with verl |
| LoRA | `r=64, alpha=32, all-linear, dropout=0` |
| Evaluation | vLLM sampling; DeepSeek-style five-layer answer extraction + `math_equal` |
| Local hardware | RTX 5080 16 GB for SFT and evaluation |
| RL hardware | Cloud L40S / L20 |

## Evidence and Implementation Index

| Method | Data / training entry point | Evaluation entry point | Analysis script | Primary evidence |
|---|---|---|---|---|
| Base | [`SETUP.md`](SETUP.md) | [`03_eval_pass_at_k.py`](v3/E1_baseline/eval/03_eval_pass_at_k.py) | [`_compute_pass_maj_curves.py`](v3/E1_baseline/tools/_compute_pass_maj_curves.py) | [E1 protocol](v3/E1_baseline/README.md) |
| SFT | [`01_make_sft_data.py`](v3/E2_sft/data_gen/01_make_sft_data.py) · [`01_sft.py`](v3/E2_sft/train/01_sft.py) | Reuse the E1 evaluator with LoRA rank set to 64 | [`_plot_step130_combined.py`](v3/E2_sft/tools/_plot_step130_combined.py) | [combined dashboard](v3/E2_sft/outputs/lr5e-4_step130_combined.png) · [run note](v3/E2_sft/README.md) |
| DAPO | [`data_prep.py`](v3/E5_grpo/r11_verl/data_prep.py) · [`run_dapo_r15.sh`](v3/E5_grpo/r15_dapo/run_dapo_r15.sh) · [`reward_judge_r15.py`](v3/E5_grpo/r15_dapo/reward_judge_r15.py) | [Dev checkpoint selection](v3/E5_grpo/eval/01_grpo_dev_eval.py) · E1 full evaluator | [`_plot_dapo_ck15_combined.py`](v3/E5_grpo/tools/_plot_dapo_ck15_combined.py) | [combined dashboard](v3/E5_grpo/outputs/k64_dapo_ck15/dapo_ck15_combined.png) |
| GRPO | [`data_prep.py`](v3/E5_grpo/r11_verl/data_prep.py) · [`run_grpo_r16_clean.sh`](v3/E5_grpo/r16_grpo_clean/run_grpo_r16_clean.sh) · [`reward_judge.py`](v3/E5_grpo/r16_grpo_clean/reward_judge.py) | [Dev checkpoint selection](v3/E5_grpo/eval/01_grpo_dev_eval.py) · E1 full evaluator | [`_plot_r16_step42_deep.py`](v3/E5_grpo/tools/_plot_r16_step42_deep.py) | [combined dashboard](v3/E5_grpo/outputs/k64_r16_step42/r16_step42_combined.png) · [run note](v3/E5_grpo/README.md) |
| Same-chain PPL | Local chain pools + selected adapters | [`yue_ppl_8panel_selfppl.py`](v3/E5_grpo/tools/yue_ppl_8panel_selfppl.py) | [`yue_ppl_8panel_selfppl_plot.py`](v3/E5_grpo/tools/yue_ppl_8panel_selfppl_plot.py) | [compact result](v3/E5_grpo/outputs/yue_ppl_analysis/ppl_8panel_selfppl_results.json) · [figure](v3/E5_grpo/outputs/yue_ppl_analysis/yue_8panel_selfppl.png) |

## Repository Map

| Path | Contents |
|---|---|
| [`v3/E1_baseline/`](v3/E1_baseline/) | Base protocol and resumable pass@K / maj@K evaluator |
| [`v3/E2_sft/`](v3/E2_sft/) | SFT data/training entry points and the selected-checkpoint dashboard |
| [`v3/E5_grpo/`](v3/E5_grpo/) | verl DAPO / GRPO entry points, reward code, dashboards, and the PPL probe |
| [`v3/shared/`](v3/shared/) | Shared answer extraction and equivalence utilities |
| [`SETUP.md`](SETUP.md) | Model, data, and local/container environment preparation |

## Reproduction Scope and Environments

Training, evaluation, and verl RL use separate environments to avoid dependency conflicts across TRL, vLLM, and verl.

```bash
# SFT / training
docker build -f docker/Dockerfile.train -t gemma-math:train .

# vLLM evaluation
docker build -f docker/Dockerfile.eval -t gemma-math:eval .

# verl RL
docker build -f docker/Dockerfile.grpo -t gemma-math:grpo .
```

The pinned stack is Python 3.11 / CUDA 12.8 / torch 2.10. See [`SETUP.md`](SETUP.md) for model, data, and environment preparation.

### Clean-Clone Reproducibility

| Layer | Current status |
|---|---|
| Environment build | Dockerfiles and separated requirement sets are committed |
| SFT data and training | Main entry points are committed; download Base and GSM8K, then replace historical local paths with paths in the new environment |
| Shared evaluation | The chunk/resume evaluator is committed; datasets and Base/LoRA weights are required |
| Exact DAPO / GRPO rerun | Launch and reward code are committed; cloud-specific absolute paths, RL parquet data, and storage configuration must be rebuilt in the target environment |
| Combined-dashboard regeneration | Rendered figures and their analysis code are committed; large per-sample JSON files are intentionally excluded, so a clean clone cannot regenerate them byte-for-byte without recreating those dumps |
| Same-chain PPL figure | The compact numeric result and plotting script are committed and reproduce the visualization; recomputing forward-pass PPL requires the excluded chain pools, Base weights, and selected adapters |

This distinction separates auditable code and configuration from a claim of one-command reproduction of the original runs.

## Claim Boundaries

This project is an **empirical replication and behavioral diagnosis**, not a proposal of a new RL algorithm. It also does not equate finite-K `pass@K` with an absolute model capability boundary. Current limitations include one Base model, single-seed and short-horizon RL runs, unmatched training budgets across algorithms, a numerically verifiable MATH subset, and possible failure to observe very low-probability correct paths at finite K.

Model weights, checkpoints, datasets, and large per-sample dumps are not committed. The default branch retains the canonical training/evaluation entry points, compact PPL result, and four primary figures; local experiment archaeology is intentionally excluded from the public tree.

## License

[MIT](LICENSE)

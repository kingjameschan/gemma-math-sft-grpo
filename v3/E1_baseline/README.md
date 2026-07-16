# E1 — Base Protocol and Evaluation Harness

E1 fixes the inference and scoring protocol reused by SFT, DAPO, and GRPO.

## Fixed protocol

- **Prompt:** DeepSeek-style CoT — `{question}\nPlease reason step by step, and put your final answer within \\boxed{}.`
- **Template:** Gemma-2 native chat template, no system prompt
- **Sampling:** vLLM, temperature `0.7`, top-p `0.95`
- **Extraction:** five-layer final-answer extraction followed by numerical `math_equal`
- **Metrics:** unbiased `pass@k` plus answer-mode `maj@k`
- **Execution:** chunked output with checkpoint/resume support

## Retained files

| Path | Purpose |
|---|---|
| `eval/03_eval_pass_at_k.py` | Canonical shared sampling and scoring harness |
| `tools/_compute_pass_maj_curves.py` | Curve aggregation from completed evaluation dumps |
| `tools/_make_difficulty_labels.py` | Deterministic Base-derived difficulty buckets used by the dashboards |

The public branch intentionally omits model weights, datasets, large per-sample dumps, and exploratory plots. Headline Base values and the retained cross-method dashboards are summarized in the repository root README.

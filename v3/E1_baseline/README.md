# E1 — Baseline & eval harness

Establishes the **fixed eval protocol** used by every later experiment, and measures the
untuned `gemma-2-2b-it` baseline.

## Eval protocol (held constant everywhere)
- **Prompt:** DS-CoT — `{question}\nPlease reason step by step, and put your final answer within \boxed{}.` (no system prompt, Gemma2 native chat template).
- **Extraction:** DeepSeek-style 5-layer answer extraction (`v3/shared/answer_extraction.py`) → `math_equal` numeric compare.
- **Metrics:** `pass@k` (unbiased estimator) and `maj@k` (majority vote), via vLLM batched sampling (T=0.7, top_p=0.95).
- **Engine:** vLLM; eval driver `eval/03_eval_pass_at_k.py` (chunked, resumable).

## Files
| Path | What |
|---|---|
| `eval/01_eval_ds_cot.py` | single-ckpt DS-CoT eval |
| `eval/02_eval_ds_tir.py` | tool-integrated-reasoning eval (finding: TIR ≡ CoT for Gemma2-IT, 0% spontaneous code use) |
| `eval/03_eval_pass_at_k.py` | the shared pass@k / maj@k harness (chunked) |
| `eval/_gold_ppl*.py` | gold-answer perplexity probes |
| `tools/` | difficulty bucketing, length/truncation analysis, summary plotters |
| `outputs/` | baseline figures + `eval_log.jsonl` |

## Baseline (recorded)
`gemma-2-2b-it`, DS-CoT, GSM8K: **numeric_accuracy = 61.33%** (greedy). MATH500 pass@1
(K=64) ≈ 29.8%. See `outputs/eval_log.jsonl` and the baseline figures.

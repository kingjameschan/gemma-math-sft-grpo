# R12 — Full DAPO 3-piece on TRL 0.29 (no verl)

## Why R12

R10 already has the *easy* DAPO bits (loss_type=dapo token-level batch norm,
mask_truncated_completions, asymmetric ε_low=0.2 / ε_high=0.28). R10 was
missing the *hard* bits:

| DAPO piece | R10 | R12 |
|---|---|---|
| Token-level batch normalization (`loss_type=dapo`) | yes | yes (unchanged) |
| Asymmetric clip (ε_low ≠ ε_high) **declared** | yes | yes (unchanged) |
| Asymmetric clip **actually fires** | NO (μ=1 ⇒ ratio=1) | **yes (μ=2)** |
| Mask truncated completions | yes | yes (unchanged) |
| **Soft Overlong Punishment** (linear penalty in tail of context) | NO | **yes (buffer=64)** |
| **Dynamic Sampling** (drop zero-std groups, over-sample) | NO | **yes (subclass)** |

R11 attempted the "use verl" route to get all of these for free, but verl +
TRL/vLLM 0.10.1.dev didn't compose cleanly in our docker (12 smoke iterations
all blocked on dep mismatch — see `../r11_verl/HANDOVER.md`).

R12 fallback: just **subclass `GRPOTrainer`** in TRL 0.29 itself. No new
deps, runs in the same `vllm-trl:v3-judge-fa2` image as R10.

## What's in this directory

```
r12_dapo/
├── README.md                                  ← this file
└── train/
    └── 02_grpo_judge_dapo.py                  ← train script (R12 subclass)
```

Launchers / watchers / excel live alongside the R10 ones in `scripts_local/`:

```
scripts_local/
├── run_baseit_dapo_r12.sh                     ← cloud launcher
├── local_watcher_dapo_r12.sh                  ← local sync + eval watcher
└── update_excel_dapo_r12.py                   ← excel summary builder
```

## Code-level diff vs R10

### 1. `DynamicSamplingGRPOTrainer(GRPOTrainer)` subclass

- **Override**: `_generate_and_score_completions(self, inputs)`.
- **Algo**: call `super()` to get the parent's full output dict (rewards
  scored, advantages computed). Reshape `advantages` to `(N_groups, G)`.
  Group is "informative" iff at least one `|advantage| > 1e-6`. Drop
  zero-std groups. If kept rows < target_size, pull next batch from a
  private dataloader iterator (lazy, recreated on `StopIteration`) and
  loop. Cap at `--max_num_gen_batches` rounds (default 10).
- **Concat**: every output dict tensor is row-concatenated. Width-padded
  per key (left-pad for `prompt_*`, right-pad for `completion_*`,
  pad value=`pad_token_id` for ids, 0 for masks/logps, 1.0 for IS ratio).
- **Truncate**: final dict trimmed to exactly `target_size` rows.
- **Recompute** `num_items_in_batch` from the kept `completion_mask`
  so DAPO loss normalizer (line 2081 in TRL grpo_trainer.py) sees the
  correct denominator.

### 2. Soft Overlong Punishment in `judge_reward`

```python
def soft_overlong_penalty(n_tokens, max_len=384, buffer=64):
    start = max_len - buffer       # = 320
    if n_tokens <= start:  return 1.0
    if n_tokens >= max_len: return 0.0
    return 1.0 - (n_tokens - start) / buffer
```

Applied multiplicatively on top of the judge score:
`final_reward = judge_score * soft_overlong_penalty(n_tokens)`.

When `--enable_soft_overlong` is set, the R10 `length_factor()` is auto-
disabled (they'd compete). Token count comes from `completion_ids` that
TRL passes to reward functions (line 1118 in grpo_trainer.py), with a
tokenizer fallback if the kwarg isn't passed.

### 3. μ=2 (Clip-Higher actually firing)

Pure config diff — no code change beyond the launcher.

When `num_iterations=1`, the policy used to compute `old_per_token_logps`
is the same one used to compute `per_token_logps` → `coef = exp(0) = 1`,
neither low (1<1−ε) nor high (1>1+ε) clip branch triggers.

When `num_iterations=2`, the second inner loop sees the post-step policy,
so `coef = π_new / π_old ≠ 1` for tokens where the policy moved → the
asymmetric clip ε_low=0.2 / ε_high=0.28 finally has a job.

## Known limitations / TODO

- **Single-GPU only.** `_calculate_rewards` already gathers across processes
  (line 1158 in grpo_trainer.py), but our local-only filtering would
  diverge across procs in multi-GPU. We assert `num_processes=1` implicitly
  by checking `advantages.numel() % G == 0`.
- **Per-step metric inflation.** Each call to `super()._generate_and_score_completions`
  appends to `self._metrics` and `self._logs`. So when oversampling triggers
  K rounds in one real step, those metrics get K entries. The avg-at-log
  step (line 2145) absorbs this, but `step_time` will look ~K× larger.
- **IS ratio re-pad.** When concatenating output dicts from K rounds, each
  may have different `(B, T)` width. We right-pad with neutral values
  (1.0 for IS ratio, 0.0 for logps). The completion_mask zeros these out
  in the loss anyway, so this is safe in principle — but UNTESTED on a
  case where the mask doesn't perfectly cover the padded positions.
- **prompt_ids left-pad.** Parent left-pads prompt_ids with pad_token_id
  (line 1571). When concatenating dicts of different prompt widths, we
  re-left-pad. Verified: `prompt_mask` correctly tracks the left-pad zone.
- **Reward call counter.** The reward function maintains `_step_judge_stats["calls"]`
  globally — under K-round oversampling, this counts each oversampled batch
  separately. JUDGE_STATS log shows the true call count.

## How to run

Same workflow as R10. From local WSL:

```bash
# 1. SSH to AWS instance and run the launcher (or use SSM):
#    ./scripts_local/run_baseit_dapo_r12.sh
# 2. Locally start the watcher to sync + eval as ckpts appear:
nohup bash /mnt/d/fine-tuning/scripts_local/local_watcher_dapo_r12.sh \
  > /mnt/d/fine-tuning/scripts_local/local_watcher_dapo_r12_stdout.log 2>&1 &
disown
# 3. Open the live excel:
#    /mnt/d/fine-tuning/v3/E5_grpo/outputs/R12_baseit_r12_dapo_full_15ep_summary.xlsx
```

## Smoke test (recommended before full run)

Before launching the 240-step / ~18h run, recommend a 5-step smoke test:

```bash
GOOGLE_API_KEY=... python3 /mnt/d/fine-tuning/v3/E5_grpo/r12_dapo/train/02_grpo_judge_dapo.py \
  --lr 1e-5 --beta 0.0 --num_iterations 2 \
  --loss_type dapo --epsilon 0.2 --epsilon_high 0.28 \
  --mask_truncated_completions \
  --max_num_gen_batches 10 \
  --enable_soft_overlong --soft_overlong_buffer 64 \
  --max_new_tokens 384 \
  --group_size 8 --batch_size 16 --accum 24 \
  --max_steps 5 --save_steps 5 \
  --use_vllm --vllm_mode colocate --vllm_gpu_memory_utilization 0.25 \
  --init_adapter "" \
  --output_dir /tmp/r12_smoke
```

Smoke pass criteria:
- `[dapo_ds]` lines in stdout show keep ratio < 1.0 on at least one step.
- `dapo_ds_stats.jsonl` has 5 lines.
- No `IndexError` / `RuntimeError` from the concat/pad path.
- GPU peak < 14 GB (we have 16 GB).
- Final ckpt-5 directory exists.

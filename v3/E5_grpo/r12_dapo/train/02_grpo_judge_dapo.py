"""v3 E5 R12: GRPO with **Dynamic Sampling** (DAPO 3-piece complete) on TRL 0.29.

R10 already shipped: loss_type=dapo (token-level batch norm), mask_truncated_completions,
asymmetric clip ε_low=0.2 / ε_high=0.28.  But R10 left:
  1. **No Dynamic Sampling** — zero-std groups (all-correct or all-wrong) still passed
     gradient (advantage=0 for the group, but they ate slots in the mini-batch and
     diluted the informative groups' contribution under DAPO token-level normalize).
  2. **No Soft Overlong Punishment** — only a binary mask_truncated_completions wipe
     when len == max; no smooth ramp for completions approaching the cap.
  3. **μ=1** — asymmetric clip ε_low/ε_high never fires because num_iterations=1
     means coef = exp(0) = 1, neither low nor high branch active.

R12 adds all three:
  1. `DynamicSamplingGRPOTrainer` — over-sample prompts, drop groups with std=0,
     keep batch full of informative groups. Cap at `--max_num_gen_batches` (default 10).
  2. Soft Overlong reward shaping in `judge_reward` — linear penalty over the last
     `--soft_overlong_buffer` tokens (default 64) of `--soft_overlong_max_len` (384).
  3. Caller responsibility: pass `--num_iterations 2` in the launcher.

Runs in EXISTING R10 docker image (vllm-trl:v3-judge-fa2).  No new deps.

Usage (see ../../scripts_local/run_baseit_dapo_r12.sh):
  GOOGLE_API_KEY=... python3 02_grpo_judge_dapo.py \
    --lr 1e-5 --beta 0.0 --num_iterations 2 \
    --loss_type dapo --epsilon 0.2 --epsilon_high 0.28 \
    --mask_truncated_completions \
    --max_num_gen_batches 10 \
    --enable_soft_overlong --soft_overlong_buffer 64 \
    --max_new_tokens 384 \
    [other args same as R10]
"""
import argparse
import datetime
import gc
import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).resolve().parents[4]
TRAIN_FILE = ROOT / "v3" / "shared" / "data" / "sft" / "train.jsonl"
DEFAULT_MODEL = ROOT / "models" / "gemma-2-2b-it"
DEFAULT_INIT_ADAPTER = ROOT / "v3" / "E2_sft" / "checkpoints" / "sft_lr5e-4_r64" / "checkpoint-130"
CKPT_BASE = ROOT / "v3" / "E5_grpo" / "checkpoints" / "fastgrid" / "stage1_v3_r12"
TRAIN_LOG = ROOT / "v3" / "E5_grpo" / "outputs" / "fastgrid" / "stage1_v3_r12_train_log.jsonl"
JUDGE_STATS = ROOT / "v3" / "E5_grpo" / "outputs" / "fastgrid" / "stage1_v3_r12_judge_stats.jsonl"
JUDGE_FAIL_SAMPLES = ROOT / "v3" / "E5_grpo" / "outputs" / "fastgrid" / "stage1_v3_r12_judge_fail_samples.jsonl"
JUDGE_RETRY_LOG = ROOT / "v3" / "E5_grpo" / "outputs" / "fastgrid" / "stage1_v3_r12_judge_retry_log.jsonl"
SAMPLE_LOG = ROOT / "v3" / "E5_grpo" / "outputs" / "fastgrid" / "stage1_v3_r12_samples.jsonl"
DAPO_DS_LOG = ROOT / "v3" / "E5_grpo" / "outputs" / "fastgrid" / "stage1_v3_r12_dapo_ds_stats.jsonl"
SAMPLE_EVERY = 5  # log 3 samples every N reward calls

sys.path.insert(0, str(ROOT / "v3" / "shared"))
from answer_extraction import extract_answer, math_equal_numerical, gold_from_completion


# ============================================================
# LLM Judge (Gemini 2.5 Flash-Lite) — verbatim from R10
# ============================================================
from google import genai
from google.genai import types

_client = None
def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _client

JUDGE_MODEL = "gemini-3.1-flash-lite-preview"

JUDGE_PROMPT = """You grade a student's solution to a GSM8K math problem.

Question: {question}
Gold answer (numeric): {gold}
Student response: {response}

PROCEDURE — follow these steps in order:

STEP A — RECOMPUTE EACH ARITHMETIC LINE
   Before scoring, recompute every "A op B = C" in the student's solution.
   Note where student's number first differs from your computation.
   Example: if student writes "14 + 12 + 24 = 40", you compute 50 and note
   the slip at this exact line. Do NOT back-solve from the wrong final answer.

STEP B — DETERMINE is_correct
   Look at the student's stated FINAL answer (any natural form: \\boxed{{N}},
   "**Answer:** N", "**Therefore... N**", bolded number, "X units of Y").
   Strip units/commas/$/% before comparing to gold (tolerance ±0.001).
   is_correct = True if and only if final number matches gold.

STEP C — ASSIGN score (band-based, conditional on is_correct)
   You MUST observe these caps strictly:

   IF is_correct = TRUE → score MUST be in {{5,6,7,8,9,10}}:
     9-10: Clean step-by-step. Every operation justified by a noun phrase.
           No dead-end intermediates. Conclusion explicit.
     7-8:  Correct but minimal/sparse OR has 1 weird intermediate step.
     5-6:  LUCKY-CORRECT — right number reached via mislabeled / confused /
           contradictory reasoning. Numerical match by chance, not logic.

   IF is_correct = FALSE → score MUST be in {{0,1,2,3,4}}:
     4:    Wrong but on-task — right values, ONE wrong operation OR ONE
           arithmetic slip in an otherwise correct chain.
     3:    Wrong with multiple compounding errors but right general approach.
     2:    Misread question (used wrong values throughout) OR dropped an
           entire required component OR final stated number contradicts the
           response's own work (e.g., work derives 80, boxed 40).
     1:    Internally contradictory throughout / declares problem unsolvable.
     0:    Empty / refused / pure restatement / pure formatting.

   ABSOLUTE RULES (no exceptions):
   - is_correct=False → score CANNOT be 5 or above, regardless of how good
     the reasoning looks. "Many steps shown" does not rescue a wrong answer.
   - is_correct=True → score CANNOT be 4 or below.

STEP D — WRITE reason
   The reason MUST point to the FIRST place where student's work diverges
   from yours (from STEP A). Do not invent a plausible error narrative by
   back-solving from the wrong final answer. If no specific divergence,
   say "lucky-correct path" or "internally contradictory" or "fully clean".

Format style (markdown, plain prose, with or without \\boxed{{}}, with or
without <<X=Y>> markers) is irrelevant — judge ONLY by content.

Output ONLY this JSON (no markdown fence, no commentary):
{{"is_correct": <true|false>, "score": <integer 0-10>, "reason": "<one sentence pointing to specific step or band condition>"}}"""


_TRANSIENT_TAGS = {"rate_limit", "server_503", "timeout", "network", "json_parse"}


def _classify_err(e):
    msg = str(e)[:200]
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg: return "rate_limit", msg
    if "503" in msg or "UNAVAILABLE" in msg: return "server_503", msg
    if "timeout" in msg.lower() or "deadline" in msg.lower(): return "timeout", msg
    if "connect" in msg.lower() or "ssl" in msg.lower(): return "network", msg
    return type(e).__name__, msg


def _try_extract_json(text):
    """Tolerant JSON extraction from Gemini output (markdown-wrapped, trailing garbage)."""
    if not text: return None
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s[:-3].strip()
        elif "```" in s:
            s = s.rsplit("```", 1)[0].strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    dec = json.JSONDecoder()
    i = s.find("{")
    while 0 <= i < len(s):
        try:
            obj, _ = dec.raw_decode(s[i:])
            return obj
        except Exception:
            i = s.find("{", i + 1)
    return None


PER_CALL_TIMEOUT_S = 60
EARLY_TERM_PCT = 0.875


def judge_one(question, gold, response, max_retries=15):
    """Verbatim from R10 — see 02_grpo_judge.py for design notes."""
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, response=response[:1500])
    last_tag, last_msg = "unknown", ""
    attempts_log = []
    for attempt in range(1, max_retries + 1):
        temp = 0.0 if attempt == 1 or last_tag != "json_parse" else min(0.3 + 0.1 * (attempt - 1), 1.0)
        cfg = types.GenerateContentConfig(
            temperature=temp,
            response_mime_type="application/json",
            http_options=types.HttpOptions(timeout=PER_CALL_TIMEOUT_S * 1000),
        )
        t0 = time.time()
        try:
            r = _get_client().models.generate_content(
                model=JUDGE_MODEL, contents=prompt, config=cfg,
            )
        except Exception as e:
            dt = time.time() - t0
            last_tag, last_msg = _classify_err(e)
            attempts_log.append((attempt, last_tag, round(dt, 1), last_msg[:150]))
            if last_tag in _TRANSIENT_TAGS and attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1), 30) + random.random())
                continue
            return None, None, last_msg, last_tag, attempt, attempts_log
        dt = time.time() - t0
        text = r.text or ""
        try:
            data = json.loads(text)
        except Exception:
            data = _try_extract_json(text)
        if data is None:
            last_tag, last_msg = "json_parse", text[:300]
            attempts_log.append((attempt, last_tag, round(dt, 1), last_msg[:150]))
            if attempt < max_retries:
                time.sleep(min(0.5 * attempt, 5) + random.random() * 0.5)
                continue
            return None, None, last_msg, last_tag, attempt, attempts_log
        try:
            score = float(data.get("score", -1))
        except Exception:
            return None, None, str(data)[:200], "score_type", attempt, attempts_log
        is_correct = data.get("is_correct", None)
        if is_correct not in (True, False):
            is_correct = None
        if 0 <= score <= 10:
            return score / 10.0, is_correct, text, "ok", attempt, attempts_log
        return None, is_correct, str(data)[:200], "score_range", attempt, attempts_log
    return None, None, last_msg, last_tag, max_retries, attempts_log


# ============================================================
# Reward function: judge + Soft Overlong Punishment (DAPO §5)
# ============================================================
_judge_executor = None
_step_judge_stats = {"calls": 0, "failed": 0}
_tokenizer = None
_use_length_penalty = True            # R10 length_factor (kept for back-compat; default off in R12 launcher)
_use_soft_overlong = True             # R12 NEW
_soft_overlong_max_len = 384          # R12 NEW: max_completion_length used by sampler
_soft_overlong_buffer = 64            # R12 NEW: last N tokens get linear penalty


def length_factor(n_tokens):
    """R10 length_factor (B-relaxed): unused by default in R12 (Soft Overlong replaces it)."""
    if n_tokens <= 200:
        return 0.95 + 0.05 * (n_tokens / 200.0)
    elif n_tokens <= 700:
        return 1.00
    else:
        return 1.00 - 0.25 * min(1.0, (n_tokens - 700) / 324.0)


def soft_overlong_penalty(n_tokens, max_len=None, buffer=None):
    """DAPO Soft Overlong: linear ramp from 1.0 → 0.0 across the last `buffer`
    tokens of `max_len`.

    Region |  start = max_len - buffer
        n_tokens <= start         → factor = 1.0   (no penalty, reward unchanged)
        start < n_tokens < max_len → factor = 1 - (n_tokens - start) / buffer
        n_tokens >= max_len        → factor = 0.0   (clipped to zero reward)

    Defaults pick up module-level globals (_soft_overlong_max_len / _buffer)
    set in main() from CLI flags."""
    if max_len is None:
        max_len = _soft_overlong_max_len
    if buffer is None:
        buffer = _soft_overlong_buffer
    if buffer <= 0:
        return 1.0
    start = max_len - buffer
    if n_tokens <= start:
        return 1.0
    if n_tokens >= max_len:
        return 0.0
    return 1.0 - (n_tokens - start) / buffer


def judge_reward(completions, answer, prompts=None, completion_ids=None, **kwargs):
    """R12 judge reward = R10 judge × Soft Overlong factor.

    Falls back to length_factor() (R10 style) only if --no_length_penalty is NOT
    set AND --enable_soft_overlong is NOT set.  Default R12 launcher uses
    soft_overlong_only (length_factor disabled).
    """
    global _judge_executor, _step_judge_stats
    if _judge_executor is None:
        _judge_executor = ThreadPoolExecutor(max_workers=64)

    n = len(completions)
    rewards = [0.0] * n
    judge_scores = [None] * n

    texts = [c if isinstance(c, str) else c[0]["content"] for c in completions]
    questions = [
        (q[0]["content"] if isinstance(q, list) else q) if q else ""
        for q in (prompts if prompts else [None] * n)
    ]

    t0 = time.time()
    futures = {
        _judge_executor.submit(judge_one, questions[i], answer[i], texts[i]): i
        for i in range(n)
    }
    threshold_k = max(int(n * EARLY_TERM_PCT), 1)
    n_failed = 0
    n_judged_ok = 0
    fail_tags = {}
    n_attempts = []
    fail_samples = []
    retry_records = []
    early_term = False
    judge_is_correct = [None] * n
    for fut in as_completed(futures):
        i = futures[fut]
        try:
            score, is_corr, info, tag, attempt, attempts_log = fut.result(timeout=600)
        except Exception as e:
            score, is_corr, info, tag, attempt, attempts_log = None, None, str(e)[:200], "future_" + type(e).__name__, 0, []
        n_attempts.append(attempt)
        if score is None:
            n_failed += 1
            fail_tags[tag] = fail_tags.get(tag, 0) + 1
            if len(fail_samples) < 3:
                fail_samples.append({"tag": tag, "attempt": attempt, "info": info[:300]})
        else:
            n_judged_ok += 1
        if attempt > 1 or attempts_log:
            retry_records.append({
                "i": i, "final_status": tag, "n_attempts": attempt,
                "attempts": attempts_log,
            })
        judge_scores[i] = score
        judge_is_correct[i] = is_corr
        if not early_term and n_judged_ok >= threshold_k:
            early_term = True
            for f in futures:
                if not f.done():
                    f.cancel()
            break
    judge_dt = time.time() - t0
    n_early_term = sum(1 for s in judge_scores if s is None) - n_failed if early_term else 0
    if fail_samples:
        JUDGE_FAIL_SAMPLES.parent.mkdir(parents=True, exist_ok=True)
        with open(JUDGE_FAIL_SAMPLES, "a") as f:
            for s in fail_samples:
                s["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
                f.write(json.dumps(s) + "\n")
    if retry_records:
        JUDGE_RETRY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(JUDGE_RETRY_LOG, "a") as f:
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            cfg_tag = os.environ.get("CFG_TAG", "")
            for rec in retry_records:
                rec["ts"] = ts
                rec["config"] = cfg_tag
                f.write(json.dumps(rec) + "\n")

    outcomes = [False] * n
    n_toks = [0] * n
    raw_scores = [None] * n
    soft_factors = [1.0] * n  # R12: log per-sample soft overlong penalty
    for i in range(n):
        outcomes[i] = math_equal_numerical(extract_answer(texts[i]), answer[i])
        score = judge_scores[i]
        is_corr = judge_is_correct[i]
        # n_tokens — prefer real completion_ids (TRL passes them), else tokenizer encode, else char/4
        if completion_ids is not None and i < len(completion_ids) and completion_ids[i] is not None:
            n_toks[i] = len(completion_ids[i])
        elif _tokenizer is not None:
            n_toks[i] = len(_tokenizer.encode(texts[i], add_special_tokens=False))
        else:
            n_toks[i] = len(texts[i]) // 4
        if score is None:
            rewards[i] = float("nan")
        else:
            # Cap-enforcement guard (judge may sometimes break the prompt rules)
            if is_corr is False and score > 0.4:
                score = 0.4
            elif is_corr is True and score < 0.5:
                score = 0.5
            raw_scores[i] = score
            # Apply post-processing reward shaping
            if _use_soft_overlong:
                soft_factors[i] = soft_overlong_penalty(n_toks[i])
                rewards[i] = score * soft_factors[i]
            elif _use_length_penalty:
                soft_factors[i] = length_factor(n_toks[i])
                rewards[i] = score * soft_factors[i]
            else:
                rewards[i] = score

    judged_rewards = [rewards[i] for i in range(n) if judge_scores[i] is not None]
    if judged_rewards:
        group_mean = sum(judged_rewards) / len(judged_rewards)
        for i in range(n):
            if judge_scores[i] is None:
                rewards[i] = group_mean
    else:
        group_mean = float("nan")
        for i in range(n):
            rewards[i] = 1.0 if outcomes[i] else 0.0

    _step_judge_stats["calls"] += n
    _step_judge_stats["failed"] += n_failed
    _step_judge_stats["reward_call_idx"] = _step_judge_stats.get("reward_call_idx", 0) + 1
    call_idx = _step_judge_stats["reward_call_idx"]
    success_rate = 1 - (n_failed / max(n, 1))

    if call_idx % SAMPLE_EVERY == 1:
        SAMPLE_LOG.parent.mkdir(parents=True, exist_ok=True)
        order = sorted(range(n), key=lambda i: rewards[i])
        picks = list({order[0], order[-1], n // 2})
        with open(SAMPLE_LOG, "a") as f:
            for i in picks:
                f.write(json.dumps({
                    "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                    "call_idx": call_idx,
                    "config": os.environ.get("CFG_TAG", ""),
                    "i": i,
                    "prompt": questions[i][:200],
                    "completion": texts[i],
                    "reward": round(rewards[i], 4),
                    "raw_score": round(raw_scores[i], 4) if raw_scores[i] is not None else None,
                    "soft_factor": round(soft_factors[i], 4),
                    "n_tokens": n_toks[i],
                    "judge_score": round(judge_scores[i], 4) if judge_scores[i] is not None else None,
                    "judge_is_correct": judge_is_correct[i],
                    "extractor_correct": outcomes[i],
                    "completion_len_chars": len(texts[i]),
                }) + "\n")

    n_fallback = sum(1 for s in judge_scores if s is None)
    n_overlong_clipped = sum(1 for f in soft_factors if f == 0.0)
    n_soft_penalized = sum(1 for f in soft_factors if 0.0 < f < 1.0)

    JUDGE_STATS.parent.mkdir(parents=True, exist_ok=True)
    with open(JUDGE_STATS, "a") as f:
        f.write(json.dumps({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "step_calls": n,
            "step_failed": n_failed,
            "step_success_rate": round(success_rate, 4),
            "judge_dt_s": round(judge_dt, 2),
            "fail_tags": fail_tags,
            "max_attempts": max(n_attempts) if n_attempts else 0,
            "mean_attempts": round(sum(n_attempts) / max(len(n_attempts), 1), 2),
            "early_term": early_term,
            "n_judged_ok": n_judged_ok,
            "n_fallback": n_fallback,
            "group_mean": round(group_mean, 4) if group_mean == group_mean else None,
            "n_overlong_clipped": n_overlong_clipped,
            "n_soft_penalized": n_soft_penalized,
            "mean_n_tokens": round(sum(n_toks) / max(n, 1), 1),
            "max_n_tokens": max(n_toks) if n_toks else 0,
            "cum_calls": _step_judge_stats["calls"],
            "cum_failed": _step_judge_stats["failed"],
            "cum_success_rate": round(1 - _step_judge_stats["failed"] / max(_step_judge_stats["calls"], 1), 4),
        }) + "\n")

    return rewards


# ============================================================
# Standard model loading + LoRA setup — verbatim from R10
# ============================================================
def lr_str(lr): return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
def beta_str(b): return f"{b:g}"


def load_grpo_dataset(path):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "prompt": r["prompt"],
                "answer": gold_from_completion(r["completion"]),
            })
    return Dataset.from_list(rows)


def load_base_model(model_path):
    try:
        import flash_attn  # noqa
        attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"
    print(f"[model] attn={attn}")
    return AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation=attn,
    )


def load_with_init_adapter(base_path, init_adapter_path):
    base = load_base_model(base_path)
    print(f"[init] loading SFT adapter from {init_adapter_path}")
    model = PeftModel.from_pretrained(base, str(init_adapter_path), is_trainable=True)
    model.print_trainable_parameters()
    return model


def make_grpo_config(args, output_dir, eos_ids):
    extra = {}
    if args.use_vllm:
        extra["use_vllm"] = True
        extra["vllm_mode"] = args.vllm_mode
        extra["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
        gen_kwargs = {"stop_token_ids": eos_ids}
    else:
        gen_kwargs = {"eos_token_id": eos_ids}
    return GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=args.lr, beta=args.beta,
        loss_type=args.loss_type, scale_rewards="group",
        num_iterations=args.num_iterations, epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        mask_truncated_completions=args.mask_truncated_completions,
        num_generations=args.group_size,
        max_completion_length=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
        generation_kwargs=gen_kwargs,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        optim="adamw_torch_fused", weight_decay=0.01,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        logging_steps=args.logging_steps, log_completions=False,
        report_to="none", seed=args.seed,
        dataloader_num_workers=0,
        **extra,
    )


# ============================================================
# R12 NEW: DynamicSamplingGRPOTrainer  (DAPO §3.2 Dynamic Sampling)
# ============================================================
class DynamicSamplingGRPOTrainer(GRPOTrainer):
    """Subclass that filters out groups whose reward std == 0 (all-correct or
    all-wrong) and over-samples from the dataloader to keep the generation batch
    full of *informative* groups (DAPO Eq. 11).

    Why this matters under DAPO loss:
      The DAPO normalizer is `num_items_in_batch / num_processes` (sum of
      completion tokens across the WHOLE generation batch). Zero-std groups
      have advantage=0 → contribute zero numerator but full denominator,
      diluting the per-token signal of informative groups. Dropping them
      makes each step's gradient strictly stronger per informative token.

    Mechanism:
      1. Call parent `_generate_and_score_completions(inputs)` to get a full
         output dict (rewards, advantages already computed).
      2. Reshape advantages to (-1, G); flag zero-std groups
         (advantages.abs().sum(dim=1) ≈ 0 ⇔ rewards in group all equal).
      3. Slice every tensor in output dict + the inputs list to keep only
         informative rows.
      4. If kept rows < target_size, pull next batch of inputs from our
         dataloader iterator and repeat. Cap at `max_num_gen_batches`.
      5. Concatenate accumulated dicts, truncate to exactly target_size.
      6. Recompute `num_items_in_batch` from kept completion_mask so the
         DAPO normalizer reflects the *kept* batch.

    Single-GPU only (we run num_processes=1). Multi-GPU support would need to
    reconcile filtering decisions across processes — left as TODO.

    Limitations / known caveats (verified vs unverified):
      - VERIFIED: parent's reward kwargs from inputs[0].keys() — we pass back
        the same list-of-dicts so reward func still sees `answer` etc.
      - VERIFIED: `_calculate_rewards` already gathers rewards across procs
        (line 1158); our local-only filtering will diverge across procs in
        multi-GPU — that's why we restrict to single-GPU.
      - VERIFIED: parent uses `len(inputs)` for the local batch_size (line
        1781-1783). Truncating the output dict to len(target) keeps slicing
        consistent. We also re-pad to a uniform dim-1 shape across iterations
        because each parent call may produce different P+C width.
      - UNVERIFIED: importance_sampling_ratio when num_iterations=2 — first
        call's IS ratio is recomputed by the parent on each generate step.
        Concatenating IS ratios from 2+ separate parent calls produces a
        *correct-per-row* tensor; loss uses it row-wise, no cross-row interp.
      - UNVERIFIED: parent appends to `self._metrics` and `self._logs` —
        calling it K times per real step inflates these by Kx. The avg-at-log
        denominator (line 2145 sum/len) absorbs this correctly but `step_time`
        will look ~K× larger because each parent call records its own time.
      - TODO: zero_std_eps tuning. Threshold 1e-6 catches "all rewards equal"
        because advantages = (r - r.mean) / (std + 1e-4). When std=0, advantage
        = 0/1e-4 = 0 exactly. Group is zero-std iff every advantage in the
        group is exactly 0. We use 1e-6 to be safe vs floating-point noise.
    """

    def __init__(self, *args, max_num_gen_batches: int = 10, dapo_zero_std_eps: float = 1e-6, **kw):
        super().__init__(*args, **kw)
        self.max_num_gen_batches = max_num_gen_batches
        self.dapo_zero_std_eps = dapo_zero_std_eps
        self._dapo_iter = None
        self._dapo_dl = None  # reference to the train dataloader (set lazily)
        # log file stream (opened on first call)
        self._dapo_log_path = DAPO_DS_LOG

    # ---------- helpers ----------

    def _ensure_iter(self):
        """Lazily grab the train dataloader from the callback handler (set by
        Trainer._inner_training_loop) and keep our own iterator over it. We
        recreate when exhausted so over-sampling never blocks training."""
        if self._dapo_iter is not None:
            return
        # Trainer.callback_handler.train_dataloader is set in _inner_training_loop.
        # If we're called before that (shouldn't happen for normal training),
        # fall back to building one fresh.
        dl = getattr(self.callback_handler, "train_dataloader", None)
        if dl is None:
            dl = self.get_train_dataloader()
        self._dapo_dl = dl
        self._dapo_iter = iter(dl)

    def _next_input_batch(self):
        """Fetch one generation batch (list of dicts) from our private iterator.
        Recreate the iter on StopIteration so we can over-sample indefinitely."""
        self._ensure_iter()
        try:
            return next(self._dapo_iter)
        except StopIteration:
            self._dapo_iter = iter(self._dapo_dl)
            return next(self._dapo_iter)

    @staticmethod
    def _identify_keep_groups(advantages: torch.Tensor, group_size: int, eps: float):
        """advantages: (B,) where B = N_groups * group_size.
        Returns: tensor[bool] of shape (B,) — True for rows in informative groups.
        A group is *informative* iff at least one |advantage| > eps."""
        n_groups = advantages.numel() // group_size
        adv_grp = advantages.view(n_groups, group_size)
        # group is informative iff any |adv| > eps (equivalently, std > 0)
        keep_grp = (adv_grp.abs() > eps).any(dim=1)        # (n_groups,)
        # Expand back to row-level mask
        keep_rows = keep_grp.repeat_interleave(group_size)  # (B,)
        return keep_rows, keep_grp

    @staticmethod
    def _slice_output_dict(out: dict, keep_rows: torch.Tensor) -> dict:
        """Slice every tensor along dim 0 by keep_rows. Pass through scalars / non-tensors."""
        new = {}
        for k, v in out.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.size(0) == keep_rows.size(0):
                new[k] = v[keep_rows]
            else:
                # scalar (num_items_in_batch) or unrelated tensor — keep as-is
                new[k] = v
        return new

    @staticmethod
    def _pad_to_width(t: torch.Tensor, width: int, pad_value):
        """Right-pad a 2D tensor's dim-1 to `width` using pad_value. No-op if already ≥ width."""
        if t.dim() != 2 or t.size(1) >= width:
            return t
        extra = width - t.size(1)
        pad = torch.full((t.size(0), extra), pad_value, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad], dim=1)

    def _concat_dicts(self, dicts: list) -> dict:
        """Concatenate output dicts row-wise. Re-pad dim-1 tensors to the max
        width across iterations so torch.cat works.

        Padding values:
          prompt_ids, completion_ids → pad_token_id (left-pad already in parent
            for prompt; right-pad for completion → we right-pad both here, but
            mask is what loss uses, so the prompt pad position matters less).
          prompt_mask, completion_mask, tool_mask → 0
          per-token logp tensors → 0.0
          importance_sampling_ratio → 1.0 (neutral, but masked out anyway)

        Note: prompt_ids are left-padded in parent (line 1571) so right-padding
        them here might place pad on the wrong side — but they go through
        attention_mask and only affect the prompt KV cache during the train
        forward. To be safe, we left-pad prompt_ids/prompt_mask and right-pad
        the completion_* tensors.
        """
        if len(dicts) == 1:
            return dicts[0]
        keys = dicts[0].keys()
        out = {}
        pad_id = self.pad_token_id
        for k in keys:
            vals = [d[k] for d in dicts]
            # ----- non-tensor or 0-d tensor -----
            t0 = vals[0]
            is_tensor = isinstance(t0, torch.Tensor)
            if not is_tensor or t0.dim() == 0:
                # 0-d tensor (e.g. num_items_in_batch = total completion tokens) or
                # plain Python scalar. We sum these — they get RECOMPUTED below from
                # the final kept completion_mask anyway.
                if k == "num_items_in_batch":
                    try:
                        if is_tensor:
                            out[k] = torch.stack(vals).sum()
                        else:
                            out[k] = sum(vals)
                    except Exception:
                        out[k] = vals[0]
                else:
                    out[k] = vals[0]
                continue

            if t0.dim() == 1:
                # (B,) — just concat
                out[k] = torch.cat(vals, dim=0)
                continue

            # 2D tensor (B, seq) — pad to common width then concat.
            max_w = max(v.size(1) for v in vals)
            if k in ("prompt_ids",):
                # left-pad
                padded = []
                for v in vals:
                    if v.size(1) >= max_w:
                        padded.append(v); continue
                    extra = max_w - v.size(1)
                    pad = torch.full((v.size(0), extra), pad_id, dtype=v.dtype, device=v.device)
                    padded.append(torch.cat([pad, v], dim=1))
                out[k] = torch.cat(padded, dim=0)
            elif k in ("prompt_mask",):
                padded = []
                for v in vals:
                    if v.size(1) >= max_w:
                        padded.append(v); continue
                    extra = max_w - v.size(1)
                    pad = torch.zeros((v.size(0), extra), dtype=v.dtype, device=v.device)
                    padded.append(torch.cat([pad, v], dim=1))
                out[k] = torch.cat(padded, dim=0)
            else:
                # right-pad. Pick pad value by key
                if k in ("completion_ids",):
                    pv = pad_id
                elif k in ("completion_mask", "tool_mask"):
                    pv = 0
                elif k in ("importance_sampling_ratio",):
                    pv = 1.0
                else:  # logp tensors and anything else → 0.0
                    pv = 0.0
                padded = [self._pad_to_width(v, max_w, pv) for v in vals]
                out[k] = torch.cat(padded, dim=0)
        return out

    @staticmethod
    def _truncate_dict(out: dict, n: int) -> dict:
        """Truncate every (B, ...) tensor to first n rows. Scalars pass through."""
        new = {}
        for k, v in out.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.size(0) > n:
                new[k] = v[:n]
            else:
                new[k] = v
        return new

    def _log_dapo_step(self, summary: dict):
        try:
            self._dapo_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._dapo_log_path, "a") as f:
                summary["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
                summary["config"] = os.environ.get("CFG_TAG", "")
                summary["global_step"] = int(self.state.global_step) if hasattr(self, "state") else -1
                f.write(json.dumps(summary) + "\n")
        except Exception as e:
            print(f"[dapo_log WARN] {e}")

    # ---------- main override ----------

    def _generate_and_score_completions(self, inputs):
        """Override: dynamic sampling — drop zero-std groups, over-sample.

        Returns the same dict structure as the parent."""
        target_size = len(inputs)                         # B (e.g. 384)
        G = self.num_generations                          # G (e.g. 8)
        assert target_size % G == 0, f"target_size {target_size} not divisible by G={G}"

        # Special case: max_num_gen_batches == 1 disables dynamic sampling entirely.
        # We just call parent once and return as-is. Saves the filter/concat overhead.
        if self.max_num_gen_batches <= 1:
            return super()._generate_and_score_completions(inputs)

        # First-pass: parent generates + scores the original input batch
        accumulated_dicts = []     # list of slice dicts (each row-aligned, kept rows only)
        accumulated_rows = 0
        cur_inputs = inputs
        n_iter = 0
        n_total_groups_seen = 0
        n_kept_groups_total = 0
        last_full_out = None       # most recent parent call's full (unfiltered) dict — used as final pad-up
        t_total0 = time.time()

        while n_iter < self.max_num_gen_batches:
            n_iter += 1
            t0 = time.time()
            out = super()._generate_and_score_completions(cur_inputs)
            t_parent = time.time() - t0
            last_full_out = out

            # Identify informative rows / groups
            adv = out["advantages"]                       # (B_local,)
            assert adv.numel() % G == 0, (
                f"advantages length {adv.numel()} not divisible by G={G}; "
                f"DynamicSamplingGRPOTrainer expects single-GPU. "
                f"For multi-GPU support see TODO in subclass docstring."
            )
            keep_rows, keep_grp = self._identify_keep_groups(adv, G, self.dapo_zero_std_eps)
            n_kept = int(keep_grp.sum().item())
            n_total = keep_grp.numel()
            n_total_groups_seen += n_total
            n_kept_groups_total += n_kept

            print(f"[dapo_ds] iter {n_iter}/{self.max_num_gen_batches}: "
                  f"groups kept {n_kept}/{n_total} "
                  f"(rows {int(keep_rows.sum().item())}/{keep_rows.numel()}) "
                  f"parent_dt={t_parent:.1f}s")

            if n_kept > 0:
                kept_dict = self._slice_output_dict(out, keep_rows)
                accumulated_dicts.append(kept_dict)
                accumulated_rows += int(keep_rows.sum().item())

            if accumulated_rows >= target_size:
                break

            # Need more — pull next batch of inputs and loop
            if n_iter < self.max_num_gen_batches:
                cur_inputs = self._next_input_batch()
                # Defensive: dataloader yields list-of-dicts. If size doesn't match
                # what RepeatSampler set up, it's a dataloader contract bug — parent
                # will error before we hit it.

        # If we never collected enough informative rows (cap reached or all zero-std),
        # pad up using the most recent parent call's UNFILTERED output. This guarantees
        # we always return >= target_size rows so the rest of the trainer doesn't break.
        if accumulated_rows < target_size and last_full_out is not None:
            shortfall = target_size - accumulated_rows
            print(f"[dapo_ds] cap reached, need {shortfall} more rows; padding from last full batch")
            # Take first `shortfall` rows from the last unfiltered output. Keep group
            # integrity: round up to next G multiple so we never split a group.
            pad_rows = ((shortfall + G - 1) // G) * G
            pad_keep = torch.zeros(last_full_out["advantages"].size(0),
                                   dtype=torch.bool, device=last_full_out["advantages"].device)
            pad_keep[:pad_rows] = True
            pad_dict = self._slice_output_dict(last_full_out, pad_keep)
            accumulated_dicts.append(pad_dict)
            accumulated_rows += pad_rows

        # Concatenate kept slices
        if len(accumulated_dicts) == 1:
            merged = accumulated_dicts[0]
        else:
            merged = self._concat_dicts(accumulated_dicts)

        # Truncate to exactly target_size (G-multiple boundaries preserved because
        # all our slicing operates at row-level inside G-groups)
        if any(isinstance(v, torch.Tensor) and v.dim() >= 1 and v.size(0) > target_size for v in merged.values()):
            merged = self._truncate_dict(merged, target_size)

        # Recompute num_items_in_batch from the *kept* completion_mask. DAPO loss
        # divides by this exact value (parent line 2081), so getting it right is
        # essential for correct gradient magnitude.
        if "completion_mask" in merged and isinstance(merged["completion_mask"], torch.Tensor):
            # If tool_mask present, multiply (matches parent line 1959 mask combo)
            cm = merged["completion_mask"]
            if "tool_mask" in merged and isinstance(merged.get("tool_mask"), torch.Tensor):
                cm = cm * merged["tool_mask"]
            merged["num_items_in_batch"] = int(cm.sum().item())

        # Stats log
        elapsed = time.time() - t_total0
        self._log_dapo_step({
            "n_iter": n_iter,
            "groups_seen": n_total_groups_seen,
            "groups_kept": n_kept_groups_total,
            "keep_rate": round(n_kept_groups_total / max(n_total_groups_seen, 1), 4),
            "target_size": target_size,
            "final_rows": int(next(iter(merged.values())).size(0)) if any(isinstance(v, torch.Tensor) for v in merged.values()) else target_size,
            "num_items_in_batch": merged.get("num_items_in_batch"),
            "total_dt_s": round(elapsed, 1),
        })

        return merged


# ============================================================
# Callbacks — verbatim from R10
# ============================================================
class ResumeStateCallback(TrainerCallback):
    def __init__(self, resume_dir):
        self.resume_dir = Path(resume_dir)

    def on_save(self, args, state, control, model=None, optimizer=None, lr_scheduler=None, **kwargs):
        if model is None or optimizer is None:
            return control
        tmp = self.resume_dir.parent / (self.resume_dir.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained(str(tmp))
        except Exception as e:
            print(f"[resume_save WARN] adapter save: {e}")
        try:
            torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
            if lr_scheduler is not None:
                torch.save(lr_scheduler.state_dict(), tmp / "scheduler.pt")
        except Exception as e:
            print(f"[resume_save WARN] optim/sched: {e}")
        try:
            import numpy as _np
            rng = {
                "python": random.getstate(),
                "numpy": _np.random.get_state(),
                "cpu": torch.random.get_rng_state(),
            }
            if torch.cuda.is_available():
                rng["cuda"] = torch.cuda.random.get_rng_state_all()
            torch.save(rng, tmp / "rng_state.pth")
        except Exception as e:
            print(f"[resume_save WARN] rng: {e}")
        try:
            state.save_to_json(str(tmp / "trainer_state.json"))
        except Exception as e:
            print(f"[resume_save WARN] state: {e}")
        try:
            if self.resume_dir.exists():
                shutil.rmtree(self.resume_dir)
            tmp.rename(self.resume_dir)
            print(f"[resume_save] step {state.global_step} → {self.resume_dir}")
        except Exception as e:
            print(f"[resume_save WARN] swap: {e}")
        return control


class EmptyCacheCallback(TrainerCallback):
    def __init__(self, every_n_steps=50):
        self.every = every_n_steps

    def on_step_end(self, args, state, control, **kw):
        if state.global_step > 0 and state.global_step % self.every == 0:
            gc.collect()
            torch.cuda.empty_cache()
            try:
                free, total = torch.cuda.mem_get_info()
                print(f"[empty_cache] step {state.global_step} | free={free/1e9:.2f}GB / total={total/1e9:.2f}GB")
            except Exception:
                pass
        return control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--init_adapter", default="",
                    help="path to SFT LoRA adapter to start from; empty = fresh LoRA on base")
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--train_file", default=str(TRAIN_FILE))
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--group_size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=20)
    ap.add_argument("--logging_steps", type=int, default=1)
    ap.add_argument("--warmup_steps", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_total_limit", type=int, default=None)
    ap.add_argument("--save_only_model", action="store_true", default=False)
    ap.add_argument("--no_save_only_model", dest="save_only_model", action="store_false")
    ap.add_argument("--auto_resume", action="store_true", default=False)
    ap.add_argument("--use_vllm", action="store_true", default=False)
    ap.add_argument("--vllm_mode", type=str, default="colocate", choices=["colocate", "server"])
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.4)
    ap.add_argument("--no_length_penalty", action="store_true", default=False,
                    help="Disable R10 length_factor (default: enabled UNLESS --enable_soft_overlong is set)")
    ap.add_argument("--num_iterations", type=int, default=1,
                    help="PPO inner-loop μ. R12 default launcher uses 2 to actually trigger asymmetric clip.")
    ap.add_argument("--loss_type", default="grpo",
                    choices=["grpo", "dapo", "dr_grpo", "bnpo", "cispo", "sapo", "luspo", "vespo"])
    ap.add_argument("--epsilon_high", type=float, default=None)
    ap.add_argument("--mask_truncated_completions", action="store_true", default=False)
    # ===== R12 NEW =====
    ap.add_argument("--max_num_gen_batches", type=int, default=10,
                    help="DAPO Dynamic Sampling cap: max number of generation rounds per real step "
                         "(includes the original round). 1 = disabled (single round, equivalent to R10).")
    ap.add_argument("--enable_soft_overlong", action="store_true", default=False,
                    help="Apply DAPO Soft Overlong Punishment to judge_reward "
                         "(linear ramp 1→0 over last `--soft_overlong_buffer` tokens of `--max_new_tokens`).")
    ap.add_argument("--soft_overlong_buffer", type=int, default=64,
                    help="Token buffer width for Soft Overlong (default 64, per DAPO §5).")
    ap.add_argument("--soft_overlong_max_len", type=int, default=None,
                    help="Override max_len for Soft Overlong (default = --max_new_tokens).")
    ap.add_argument("--dapo_zero_std_eps", type=float, default=1e-6,
                    help="Threshold for declaring a group's advantage 'zero' (drop if all |adv| < eps).")
    args = ap.parse_args()

    global _use_length_penalty, _use_soft_overlong, _soft_overlong_buffer, _soft_overlong_max_len
    _use_soft_overlong = bool(args.enable_soft_overlong)
    # If Soft Overlong is on, default to disabling the R10 length_factor
    # (they're competing reward shapers); user can still force length_factor by
    # leaving --no_length_penalty unset AND --enable_soft_overlong unset.
    if _use_soft_overlong:
        _use_length_penalty = False
    else:
        _use_length_penalty = not args.no_length_penalty
    _soft_overlong_buffer = int(args.soft_overlong_buffer)
    _soft_overlong_max_len = int(args.soft_overlong_max_len) if args.soft_overlong_max_len else int(args.max_new_tokens)

    print(f"[reward] length_factor={_use_length_penalty}  soft_overlong={_use_soft_overlong} "
          f"max_len={_soft_overlong_max_len} buffer={_soft_overlong_buffer}")

    if "GOOGLE_API_KEY" not in os.environ:
        raise SystemExit("GOOGLE_API_KEY env var required")

    eff_batch = args.batch_size * args.accum
    if eff_batch % args.group_size != 0:
        raise SystemExit(f"per_device_batch * accum ({eff_batch}) must divide group_size ({args.group_size})")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = CKPT_BASE / f"lr{lr_str(args.lr)}_b{beta_str(args.beta)}_dapo_r12"
    output_dir.mkdir(parents=True, exist_ok=True)

    final_ckpt = output_dir / f"checkpoint-{args.max_steps}"
    if final_ckpt.exists():
        print(f"[done] {final_ckpt} exists; skipping")
        return

    resume_dir = output_dir / "_resume_state"
    resume_from_ckpt = None
    if args.auto_resume and resume_dir.exists() and (resume_dir / "optimizer.pt").exists():
        resume_from_ckpt = str(resume_dir)
        print(f"[resume] using resume_state dir → {resume_from_ckpt}")
    else:
        existing_ckpts = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )
        if existing_ckpts:
            if args.auto_resume:
                resume_from_ckpt = str(existing_ckpts[-1])
                print(f"[resume] no resume_state, fallback to {resume_from_ckpt}")
            else:
                print(f"[abort] partial ckpts in {output_dir} but --auto_resume not set; aborting")
                raise SystemExit(1)

    print("\n[load] dataset...")
    ds = load_grpo_dataset(Path(args.train_file))

    print(f"\n[config] lr={args.lr:g} β={args.beta:g} G={args.group_size} T={args.temperature}")
    print(f"  init adapter: {args.init_adapter}")
    print(f"  output: {output_dir}")
    print(f"  reward: Gemini 2.5 Flash-Lite judge + Soft Overlong"
          f" (max_len={_soft_overlong_max_len}, buffer={_soft_overlong_buffer})"
          if _use_soft_overlong else
          f"  reward: Gemini 2.5 Flash-Lite judge"
          + (" + length_factor" if _use_length_penalty else " (no length shaping)"))
    print(f"  dynamic sampling: max_num_gen_batches={args.max_num_gen_batches} "
          f"(1 = disabled / R10 mode)")

    print("\n[load] tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    global _tokenizer
    _tokenizer = tokenizer
    use_init_adapter = bool(args.init_adapter and args.init_adapter.strip())
    if use_init_adapter:
        print(f"[init] using SFT init adapter: {args.init_adapter}")
        model = load_with_init_adapter(Path(args.model), Path(args.init_adapter))
        peft_cfg = None
    else:
        print(f"[init] no adapter — fresh LoRA r={args.lora_r} α={args.lora_alpha} on base")
        model = load_base_model(Path(args.model))
        peft_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            bias="none", task_type="CAUSAL_LM",
        )

    eos_ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if eot is not None and eot != tokenizer.unk_token_id and eot not in eos_ids:
        eos_ids.append(eot)
    print(f"[gen] eos_token_id={eos_ids}")

    cfg = make_grpo_config(args, output_dir, eos_ids)
    # === R12 KEY DIFF: use DynamicSamplingGRPOTrainer subclass ===
    trainer = DynamicSamplingGRPOTrainer(
        model=model,
        reward_funcs=[judge_reward],
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_cfg,
        max_num_gen_batches=args.max_num_gen_batches,
        dapo_zero_std_eps=args.dapo_zero_std_eps,
    )

    if eot is not None and eot != tokenizer.unk_token_id and trainer.eos_token_id != eot:
        prev = trainer.eos_token_id
        trainer.eos_token_id = eot
        print(f"[fix] trainer.eos_token_id: {prev} → {eot} (<end_of_turn>) for Gemma2 chat template")

    if args.save_only_model:
        trainer.add_callback(ResumeStateCallback(output_dir / "_resume_state"))
        print(f"[resume_state] callback armed; latest full state → {output_dir}/_resume_state")

    trainer.add_callback(EmptyCacheCallback(every_n_steps=args.save_steps))
    print(f"[empty_cache] callback armed; gc + empty_cache every {args.save_steps} steps")

    print("\n[train] starting GRPO with judge reward + Dynamic Sampling (R12)...")
    t0 = time.time()
    if resume_from_ckpt:
        trainer.train(resume_from_checkpoint=resume_from_ckpt)
    else:
        trainer.train()
    final_step = trainer.state.global_step
    final_ckpt_dir = output_dir / f"checkpoint-{final_step}"
    if not final_ckpt_dir.exists():
        print(f"[save] final ckpt at step {final_step} → {final_ckpt_dir}")
        trainer.save_model(str(final_ckpt_dir))
    duration = time.time() - t0
    trainer.save_model(str(output_dir))

    final_success_rate = 1 - _step_judge_stats["failed"] / max(_step_judge_stats["calls"], 1)
    TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_LOG, "a") as f:
        f.write(json.dumps({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "stage": "v3_grpo_stage1_v3_r12",
            "lr": args.lr, "beta": args.beta,
            "init_adapter": str(args.init_adapter),
            "judge_model": JUDGE_MODEL,
            "duration_s": round(duration, 1),
            "judge_total_calls": _step_judge_stats["calls"],
            "judge_total_failed": _step_judge_stats["failed"],
            "judge_success_rate": round(final_success_rate, 4),
            "max_num_gen_batches": args.max_num_gen_batches,
            "soft_overlong": _use_soft_overlong,
            "soft_overlong_max_len": _soft_overlong_max_len,
            "soft_overlong_buffer": _soft_overlong_buffer,
        }) + "\n")
    print(f"\ndone in {duration/60:.1f} min, judge success rate = {final_success_rate*100:.2f}%")


if __name__ == "__main__":
    main()

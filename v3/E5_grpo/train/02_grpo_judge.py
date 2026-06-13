"""v3 E5 Stage 1 v3: GRPO with Gemini LLM-as-judge reward.

Key changes vs 01_grpo.py:
  - Reward = Gemini 2.5 Flash-Lite judge (with outcome guard + API fallback)
  - Concurrent judge calls (50 parallel) via ThreadPoolExecutor
  - Per-step API success rate logging
  - Starting ckpt = SFT lr=5e-4 step 130 (PEFT adapter)

Usage:
  GOOGLE_API_KEY=... python3 02_grpo_judge.py --lr 1e-5 --beta 0.04
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

ROOT = Path(__file__).resolve().parents[3]
TRAIN_FILE = ROOT / "v3" / "shared" / "data" / "sft" / "train.jsonl"
DEFAULT_MODEL = ROOT / "models" / "gemma-2-2b-it"
DEFAULT_INIT_ADAPTER = ROOT / "v3" / "E2_sft" / "checkpoints" / "sft_lr5e-4_r64" / "checkpoint-130"
CKPT_BASE = ROOT / "v3" / "E5_grpo" / "checkpoints" / "fastgrid" / "stage1_v3"
TRAIN_LOG = ROOT / "v3" / "E5_grpo" / "outputs" / "fastgrid" / "stage1_v3_train_log.jsonl"
JUDGE_STATS = ROOT / "v3" / "E5_grpo" / "outputs" / "fastgrid" / "stage1_v3_judge_stats.jsonl"
JUDGE_FAIL_SAMPLES = ROOT / "v3" / "E5_grpo" / "outputs" / "fastgrid" / "stage1_v3_judge_fail_samples.jsonl"
JUDGE_RETRY_LOG = ROOT / "v3" / "E5_grpo" / "outputs" / "fastgrid" / "stage1_v3_judge_retry_log.jsonl"
SAMPLE_LOG = ROOT / "v3" / "E5_grpo" / "outputs" / "fastgrid" / "stage1_v3_samples.jsonl"
SAMPLE_EVERY = 5  # log 3 samples every N reward calls

sys.path.insert(0, str(ROOT / "v3" / "shared"))
from answer_extraction import extract_answer, math_equal_numerical, gold_from_completion


# ============================================================
# LLM Judge (Gemini 2.5 Flash-Lite)
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
    # Strip ```json ... ``` markdown
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s[:-3].strip()
        elif "```" in s:
            s = s.rsplit("```", 1)[0].strip()
    # Try direct
    try:
        return json.loads(s)
    except Exception:
        pass
    # raw_decode: scan for first valid JSON object, ignore trailing garbage
    dec = json.JSONDecoder()
    i = s.find("{")
    while 0 <= i < len(s):
        try:
            obj, _ = dec.raw_decode(s[i:])
            return obj
        except Exception:
            i = s.find("{", i + 1)
    return None


PER_CALL_TIMEOUT_S = 60  # hard cap per single Gemini API call
EARLY_TERM_PCT = 0.875   # cancel stragglers once 87.5% (14/16) judged ok


def judge_one(question, gold, response, max_retries=15):
    """Returns (score in [0,1] or None, is_correct or None, info, fail_tag, n_attempt, attempts_log).
    Now extracts both score and is_correct from judge JSON output.
    attempts_log: list of (attempt, tag, dt_seconds, msg[:150]) for failed attempts.
    Retries transient errors (rate_limit/503/timeout/network/json_parse) with exp
    backoff capped at 30s + jitter. Deterministic errors (score_range/score_type)
    are NOT retried — model is genuinely refusing/garbage."""
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, response=response[:1500])
    last_tag, last_msg = "unknown", ""
    attempts_log = []
    for attempt in range(1, max_retries + 1):
        temp = 0.0 if attempt == 1 or last_tag != "json_parse" else min(0.3 + 0.1 * (attempt - 1), 1.0)
        cfg = types.GenerateContentConfig(
            temperature=temp,
            response_mime_type="application/json",
            http_options=types.HttpOptions(timeout=PER_CALL_TIMEOUT_S * 1000),  # ms
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
            is_correct = None  # malformed
        if 0 <= score <= 10:
            return score / 10.0, is_correct, text, "ok", attempt, attempts_log
        return None, is_correct, str(data)[:200], "score_range", attempt, attempts_log
    return None, None, last_msg, last_tag, max_retries, attempts_log


# ============================================================
# Reward function (judge-only with outcome guard + fallback)
# ============================================================
_judge_executor = None
_step_judge_stats = {"calls": 0, "failed": 0}
_tokenizer = None  # set in main() for length penalty
_use_length_penalty = True  # set False via --no_length_penalty


def length_factor(n_tokens):
    """B-relaxed length penalty: 250-700 flat at 1.0, only penalize >700.
    Short (<200): 0.95 floor. Long (>700): linear → 0.75 at 1024."""
    if n_tokens <= 200:
        return 0.95 + 0.05 * (n_tokens / 200.0)
    elif n_tokens <= 700:
        return 1.00
    else:
        return 1.00 - 0.25 * min(1.0, (n_tokens - 700) / 324.0)


def judge_reward(completions, answer, prompts=None, **kwargs):
    global _judge_executor, _step_judge_stats
    if _judge_executor is None:
        _judge_executor = ThreadPoolExecutor(max_workers=64)

    n = len(completions)
    rewards = [0.0] * n
    judge_scores = [None] * n

    # Build inputs
    texts = [c if isinstance(c, str) else c[0]["content"] for c in completions]
    questions = [
        (q[0]["content"] if isinstance(q, list) else q) if q else ""
        for q in (prompts if prompts else [None] * n)
    ]

    # Submit all judge calls concurrently
    t0 = time.time()
    futures = {
        _judge_executor.submit(judge_one, questions[i], answer[i], texts[i]): i
        for i in range(n)
    }
    threshold_k = max(int(n * EARLY_TERM_PCT), 1)  # 14 if n=16, X=87.5%
    n_failed = 0
    n_judged_ok = 0
    fail_tags = {}
    n_attempts = []
    fail_samples = []
    retry_records = []
    early_term = False
    judge_is_correct = [None] * n  # judge's verdict (separate from extractor)
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
        # Early termination: ≥ threshold_k judged → cancel rest, treat as fallback
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

    # First pass: compute rewards. New design: trust judge's is_correct + score
    # Hard caps in prompt enforce: is_correct=False → score ≤ 4 → reward ≤ 0.4
    # Hard caps in prompt enforce: is_correct=True  → score ≥ 5 → reward ≥ 0.5
    # No outcome_guard wrapper needed; judge's hard caps + extractor as backup
    # NEW: length penalty post-processing (continuous, see length_factor)
    outcomes = [False] * n
    n_toks = [0] * n
    raw_scores = [None] * n  # pre-length-penalty score for logging
    for i in range(n):
        outcomes[i] = math_equal_numerical(extract_answer(texts[i]), answer[i])
        score = judge_scores[i]
        is_corr = judge_is_correct[i]
        # Count tokens for length penalty
        if _tokenizer is not None:
            n_toks[i] = len(_tokenizer.encode(texts[i], add_special_tokens=False))
        else:
            n_toks[i] = len(texts[i]) // 4  # char-based fallback
        if score is None:
            rewards[i] = float("nan")  # placeholder, replaced below with group_mean
        else:
            # Backup safety net: if judge's is_correct conflicts with prompt cap
            # rules (e.g., judge gave score=8 but is_correct=False), enforce caps
            if is_corr is False and score > 0.4:
                score = 0.4
            elif is_corr is True and score < 0.5:
                score = 0.5
            raw_scores[i] = score
            # Apply length penalty (if enabled)
            if _use_length_penalty:
                rewards[i] = score * length_factor(n_toks[i])
            else:
                rewards[i] = score

    # Second pass: assign group_mean to fallback samples (zero advantage, no bias)
    judged_rewards = [rewards[i] for i in range(n) if judge_scores[i] is not None]
    if judged_rewards:
        group_mean = sum(judged_rewards) / len(judged_rewards)
        for i in range(n):
            if judge_scores[i] is None:
                rewards[i] = group_mean
    else:
        # Catastrophic: 0 judged → fall back to outcome guard
        group_mean = float("nan")
        for i in range(n):
            rewards[i] = 1.0 if outcomes[i] else 0.0

    # Update stats
    _step_judge_stats["calls"] += n
    _step_judge_stats["failed"] += n_failed
    _step_judge_stats["reward_call_idx"] = _step_judge_stats.get("reward_call_idx", 0) + 1
    call_idx = _step_judge_stats["reward_call_idx"]
    success_rate = 1 - (n_failed / max(n, 1))

    # Sample logging: every SAMPLE_EVERY reward calls, dump 3 completions (mix of 1 best/1 worst/1 random)
    if call_idx % SAMPLE_EVERY == 1:
        SAMPLE_LOG.parent.mkdir(parents=True, exist_ok=True)
        order = sorted(range(n), key=lambda i: rewards[i])
        picks = list({order[0], order[-1], n // 2})  # worst, best, mid
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
                    "length_factor": round(length_factor(n_toks[i]), 4),
                    "n_tokens": n_toks[i],
                    "judge_score": round(judge_scores[i], 4) if judge_scores[i] is not None else None,
                    "judge_is_correct": judge_is_correct[i],
                    "extractor_correct": outcomes[i],
                    "completion_len_chars": len(texts[i]),
                }) + "\n")

    n_fallback = sum(1 for s in judge_scores if s is None)

    # Append to per-step JSONL log
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
            "group_mean": round(group_mean, 4) if group_mean == group_mean else None,  # NaN-safe
            "cum_calls": _step_judge_stats["calls"],
            "cum_failed": _step_judge_stats["failed"],
            "cum_success_rate": round(1 - _step_judge_stats["failed"] / max(_step_judge_stats["calls"], 1), 4),
        }) + "\n")

    return rewards


# ============================================================
# Standard model loading + LoRA setup (with SFT init)
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
    """Load base + init from existing SFT LoRA adapter, then make it trainable."""
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
        # vLLM mode: SamplingParams uses stop_token_ids, not eos_token_id
        gen_kwargs = {"stop_token_ids": eos_ids}
    else:
        # PyTorch generate uses eos_token_id
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


class ResumeStateCallback(TrainerCallback):
    """每次 TRL save 后，把 full state (adapter + optimizer + scheduler + RNG + trainer_state)
    存到独立 resume_dir，覆盖。配合 save_only_model=True 用：trajectory ckpts 只 adapter，
    resume_dir 只保留最新一个 full state。disk = 87×0.6GB + 1×1.5GB ≈ 53 GB"""
    def __init__(self, resume_dir):
        self.resume_dir = Path(resume_dir)

    def on_save(self, args, state, control, model=None, optimizer=None, lr_scheduler=None, **kwargs):
        if model is None or optimizer is None:
            return control
        tmp = self.resume_dir.parent / (self.resume_dir.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        # adapter (PEFT saves adapter_model.safetensors + adapter_config.json)
        try:
            model.save_pretrained(str(tmp))
        except Exception as e:
            print(f"[resume_save WARN] adapter save: {e}")
        # optimizer + scheduler
        try:
            torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
            if lr_scheduler is not None:
                torch.save(lr_scheduler.state_dict(), tmp / "scheduler.pt")
        except Exception as e:
            print(f"[resume_save WARN] optim/sched: {e}")
        # RNG state
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
        # trainer state
        try:
            state.save_to_json(str(tmp / "trainer_state.json"))
        except Exception as e:
            print(f"[resume_save WARN] state: {e}")
        # atomic swap
        try:
            if self.resume_dir.exists():
                shutil.rmtree(self.resume_dir)
            tmp.rename(self.resume_dir)
            print(f"[resume_save] step {state.global_step} → {self.resume_dir}")
        except Exception as e:
            print(f"[resume_save WARN] swap: {e}")
        return control


class EmptyCacheCallback(TrainerCallback):
    """Periodically free the PyTorch caching allocator + Python GC.
    Mitigates fragmentation that vLLM colocate accumulates over long runs."""
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
    ap.add_argument("--epsilon", type=float, default=0.2,
                    help="PPO clip ratio (1±epsilon). Default 0.2; raise to 0.5 to relax IS clip under vLLM colocate.")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_total_limit", type=int, default=None,
                    help="Max ckpts to keep (None=unlimited, 1=rolling overwrite)")
    ap.add_argument("--save_only_model", action="store_true", default=False,
                    help="Save only model (not optimizer/scheduler) — set False for resume support")
    ap.add_argument("--no_save_only_model", dest="save_only_model", action="store_false")
    ap.add_argument("--auto_resume", action="store_true", default=False,
                    help="Auto-resume from latest checkpoint in output_dir")
    ap.add_argument("--use_vllm", action="store_true", default=False,
                    help="Use vLLM for generation (much faster than PyTorch native generate)")
    ap.add_argument("--vllm_mode", type=str, default="colocate", choices=["colocate", "server"])
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.4,
                    help="Fraction of GPU mem allocated to vLLM (rest is for training)")
    ap.add_argument("--no_length_penalty", action="store_true", default=False,
                    help="Disable length_factor() in reward (default: enabled)")
    ap.add_argument("--num_iterations", type=int, default=1,
                    help="PPO inner-loop iterations per generation (μ in GRPO paper). Default 1; raise to 2+ to activate ε clipping.")
    ap.add_argument("--loss_type", default="grpo",
                    choices=["grpo", "dapo", "dr_grpo", "bnpo", "cispo", "sapo", "luspo", "vespo"],
                    help="Loss formulation. dapo = token-level batch norm (DAPO paper, no length bias).")
    ap.add_argument("--epsilon_high", type=float, default=None,
                    help="Upper-bound epsilon for asymmetric clip. Default None (= epsilon). DAPO recommends 0.28.")
    ap.add_argument("--mask_truncated_completions", action="store_true", default=False,
                    help="Mask out truncated completions from loss (DAPO recommendation).")
    args = ap.parse_args()
    global _use_length_penalty
    _use_length_penalty = not args.no_length_penalty
    print(f"[lp] length_penalty = {_use_length_penalty}")

    if "GOOGLE_API_KEY" not in os.environ:
        raise SystemExit("GOOGLE_API_KEY env var required")

    eff_batch = args.batch_size * args.accum
    if eff_batch % args.group_size != 0:
        raise SystemExit(f"per_device_batch * accum ({eff_batch}) must divide group_size ({args.group_size})")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = CKPT_BASE / f"lr{lr_str(args.lr)}_b{beta_str(args.beta)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    final_ckpt = output_dir / f"checkpoint-{args.max_steps}"
    if final_ckpt.exists():
        print(f"[done] {final_ckpt} exists; skipping")
        return

    # Auto-resume: prefer resume_dir (latest full state), fall back to checkpoint-*
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
    print(f"  reward: Gemini 2.5 Flash-Lite judge + outcome guard")

    print("\n[load] tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    global _tokenizer
    _tokenizer = tokenizer  # for length penalty token counting
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
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[judge_reward],
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )

    # Gemma2-IT chat template ends with <end_of_turn>=107, not <eos>=1.
    # TRL's GRPOTrainer uses self.eos_token_id (single int) to detect truncation:
    #   is_truncated = ids[-1] not in [self.eos_token_id, self.pad_token_id]
    # If left at tokenizer default (=1), 100% of Gemma2-IT completions are flagged
    # truncated → mask_truncated_completions zeros all → loss=0, grad=0.
    # Patch to 107 so chat-template-terminated completions count as terminated.
    if eot is not None and eot != tokenizer.unk_token_id and trainer.eos_token_id != eot:
        prev = trainer.eos_token_id
        trainer.eos_token_id = eot
        print(f"[fix] trainer.eos_token_id: {prev} → {eot} (<end_of_turn>) for Gemma2 chat template")

    # Hybrid checkpoint: TRL saves adapter-only every save_steps (trajectory),
    # callback saves latest full state to resume_dir for spot-interrupt recovery
    if args.save_only_model:
        trainer.add_callback(ResumeStateCallback(output_dir / "_resume_state"))
        print(f"[resume_state] callback armed; latest full state → {output_dir}/_resume_state")

    trainer.add_callback(EmptyCacheCallback(every_n_steps=args.save_steps))
    print(f"[empty_cache] callback armed; gc + empty_cache every {args.save_steps} steps")

    print("\n[train] starting GRPO with judge reward...")
    t0 = time.time()
    if resume_from_ckpt:
        trainer.train(resume_from_checkpoint=resume_from_ckpt)
    else:
        trainer.train()
    # Save final ckpt as numbered checkpoint (TRL doesn't auto-save final unless
    # max_steps is multiple of save_steps)
    final_step = trainer.state.global_step
    final_ckpt_dir = output_dir / f"checkpoint-{final_step}"
    if not final_ckpt_dir.exists():
        print(f"[save] final ckpt at step {final_step} → {final_ckpt_dir}")
        trainer.save_model(str(final_ckpt_dir))
    duration = time.time() - t0
    trainer.save_model(str(output_dir))

    # Final stats
    final_success_rate = 1 - _step_judge_stats["failed"] / max(_step_judge_stats["calls"], 1)
    TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_LOG, "a") as f:
        f.write(json.dumps({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "stage": "v3_grpo_stage1_v3",
            "lr": args.lr, "beta": args.beta,
            "init_adapter": str(args.init_adapter),
            "judge_model": JUDGE_MODEL,
            "duration_s": round(duration, 1),
            "judge_total_calls": _step_judge_stats["calls"],
            "judge_total_failed": _step_judge_stats["failed"],
            "judge_success_rate": round(final_success_rate, 4),
        }) + "\n")
    print(f"\ndone in {duration/60:.1f} min, judge success rate = {final_success_rate*100:.2f}%")


if __name__ == "__main__":
    main()

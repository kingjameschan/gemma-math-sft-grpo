"""R13 rule-based reward — strict 4-layer extraction (no last-number fallback).

Binary 0/1 reward, no judge LLM, no partial credit.
Reuses v3/shared answer extraction infra but skips layer 5 (last-number fallback)
to prevent reward-hack via "any number in response wins".

Extract layers (priority order, first match wins):
  1. "final answer is $X$. I hope"  (MATH/Hendrycks convention)
  2. \\boxed{X}                       (★ primary GSM8K format)
  3. "he answer is X"                 ("The answer is..." natural language)
  4. ```output X```                   (TIR/PoT mode)
  ✗ layer 5 (last number) — REMOVED for hack-prevention

verl contract: compute_score(data_source, solution_str, ground_truth, extra_info)
  → returns float in [0, 1]
"""
import json
import os
import re
import sys

# Reuse battle-tested helpers from v3/shared (same as eval pipeline)
sys.path.insert(0, os.environ.get("V3_SHARED", "/workspace/v3/shared"))
from answer_extraction import (
    extract_boxed_answers,
    extract_program_output,
    strip_string,
    math_equal_numerical,
)

LOG_PATH = os.environ.get("R13_REWARD_LOG", "/home/ubuntu/r13_reward_log.jsonl")


def extract_answer_strict(text: str) -> str:
    """Layers 1-4 only. Returns "" if no formal answer marker matched."""
    if not text:
        return ""
    # Layer 1: "final answer is $X$. I hope"
    if "final answer is $" in text and "$. I hope" in text:
        ans = text.split("final answer is $", 1)[1].split("$. I hope", 1)[0].strip()
    # Layer 2: \boxed{X}
    elif "boxed" in text:
        boxed = extract_boxed_answers(text)
        ans = boxed[-1] if boxed else ""
    # Layer 3: "he answer is X"  (catches "The answer is..." / "the answer is...")
    elif "he answer is" in text:
        ans = text.split("he answer is")[-1].strip()
    # Layer 4: program output ```output ... ```
    elif "```output" in text:
        ans = extract_program_output(text)
    else:
        # Layer 5 REMOVED — no last-number fallback (hack prevention)
        return ""
    if not ans:
        return ""
    ans = ans.strip().split("\n")[0].lstrip(":").rstrip(".").rstrip("/")
    return strip_string(ans)


def _log_call(step, correct, pred, gold, resp, has_boxed):
    """Append per-call log to JSONL for distribution analysis."""
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps({
                "step": step,
                "correct": bool(correct),
                "has_boxed": bool(has_boxed),  # for format_rate metric
                "pred": str(pred) if pred else None,
                "gold": str(gold),
                "resp_len": len(resp),  # char-len; tokenize at analysis time
                "empty": not resp.strip(),
                "no_extract": not pred,
            }) + "\n")
    except Exception:
        pass  # never fail the training due to log I/O


async def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """verl-compatible binary reward. 1.0 if correct, 0.0 otherwise."""
    pred = extract_answer_strict(solution_str)
    correct = bool(pred) and math_equal_numerical(pred, ground_truth)
    # format_rate metric: literal "\boxed{" presence (ignores extraction success)
    has_boxed = "\\boxed{" in solution_str

    step = (extra_info or {}).get("global_step", -1)
    _log_call(step, correct, pred, ground_truth, solution_str, has_boxed)

    return 1.0 if correct else 0.0

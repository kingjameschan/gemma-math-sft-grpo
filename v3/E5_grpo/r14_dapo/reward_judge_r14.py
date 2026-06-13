"""R14 rule-based reward — R13 + \\frac/\\dfrac/simple-fraction normalization.

Same as R13:
  - 4-layer extract (boxed > "final answer is" > "the answer is" > program output)
  - Binary 0/1 reward, no judge LLM, no partial credit
  - Per-call JSONL log

R14 ADDS:
  - canonical(): \\frac{a}{b} → a/b numeric, "20/3" → 6.667, "50\\%" → 50%
  - math_equal_v2: tries R13 path first, falls back to canonical forms
  - Lets "1/2" == "\\frac{1}{2}" == "0.5" == "0.500" all judge equal
  - Critical for MATH dataset's 668 \\frac answers (10.8% of MATH).

verl contract: compute_score(data_source, solution_str, ground_truth, extra_info)
"""
import json
import os
import re
import sys

sys.path.insert(0, os.environ.get("V3_SHARED", "/workspace/v3/shared"))
from answer_extraction import (
    extract_boxed_answers,
    extract_program_output,
    strip_string,
    math_equal_numerical,
)

LOG_PATH = os.environ.get("R14_REWARD_LOG", "/home/ubuntu/r14_reward_log.jsonl")

FRAC_RE = re.compile(
    r'^(-?)\s*\\(?:d|t)?frac\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}$'
)
SIMPLE_FRAC_RE = re.compile(r'^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$')


def canonical(s: str) -> str:
    """Canonicalize fraction-like answer to numeric string. Returns input if no match."""
    if not s:
        return s
    s = s.strip().strip('$').strip()
    s = s.replace('\\%', '%').replace('\\!', '').replace('\\,', '').strip()
    # Try \frac{a}{b} or \dfrac/\tfrac
    m = FRAC_RE.match(s)
    if m:
        sign = -1.0 if m.group(1) == '-' else 1.0
        try:
            num = float(m.group(2)); den = float(m.group(3))
            if den != 0:
                return repr(sign * num / den)
        except Exception: pass
    # Try simple a/b
    m = SIMPLE_FRAC_RE.match(s)
    if m:
        try:
            num = float(m.group(1)); den = float(m.group(2))
            if den != 0:
                return repr(num / den)
        except Exception: pass
    return s


def math_equal_v2(pred: str, ref: str) -> bool:
    """R14 equality: R13 numerical first, then canonical-form retry."""
    if pred == "" or pred is None:
        return False
    if math_equal_numerical(pred, ref):
        return True
    p_canon = canonical(pred)
    r_canon = canonical(ref)
    if p_canon != pred or r_canon != ref:
        if math_equal_numerical(p_canon, r_canon): return True
        if math_equal_numerical(p_canon, ref): return True
        if math_equal_numerical(pred, r_canon): return True
    return False


def extract_answer_strict(text: str) -> str:
    """Layers 1-4 only. Returns '' if no formal answer marker matched."""
    if not text:
        return ""
    if "final answer is $" in text and "$. I hope" in text:
        ans = text.split("final answer is $", 1)[1].split("$. I hope", 1)[0].strip()
    elif "boxed" in text:
        boxed = extract_boxed_answers(text)
        ans = boxed[-1] if boxed else ""
    elif "he answer is" in text:
        ans = text.split("he answer is")[-1].strip()
    elif "```output" in text:
        ans = extract_program_output(text)
    else:
        return ""
    if not ans:
        return ""
    ans = ans.strip().split("\n")[0].lstrip(":").rstrip(".").rstrip("/")
    return strip_string(ans)


def _log_call(step, correct, pred, gold, resp, has_boxed):
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps({
                "step": step,
                "correct": bool(correct),
                "has_boxed": bool(has_boxed),
                "pred": str(pred) if pred else None,
                "gold": str(gold),
                "resp_len": len(resp),
                "empty": not resp.strip(),
                "no_extract": not pred,
            }) + "\n")
    except Exception:
        pass


async def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    pred = extract_answer_strict(solution_str)
    correct = bool(pred) and math_equal_v2(pred, ground_truth)
    has_boxed = "\\boxed{" in solution_str
    step = (extra_info or {}).get("global_step", -1)
    _log_call(step, correct, pred, ground_truth, solution_str, has_boxed)
    return 1.0 if correct else 0.0

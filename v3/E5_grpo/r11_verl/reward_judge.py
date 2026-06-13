"""verl-compatible Gemini judge reward (port from R10's 02_grpo_judge.py).

Signature follows verl custom_reward_function contract:
    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

verl calls this per-completion (RewardLoopWorker uses asyncio.gather over the batch).
We make it async so 384 trajectories/step can run concurrently against Gemini API.

Set verl config:
    custom_reward_function.path=/workspace/v3/E5_grpo/r11_verl/reward_judge.py
    custom_reward_function.name=compute_score
"""
import asyncio
import json
import os
import random
import re
import time

from google import genai
from google.genai import types

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-2.5-flash-lite")
PER_CALL_TIMEOUT_S = 30
MAX_RETRIES = 8
_TRANSIENT_TAGS = {"rate_limit", "503", "timeout", "network", "json_parse"}

# Same prompt as R10 (kept verbatim for reward continuity in ablation comparison)
JUDGE_PROMPT = """You grade a student's solution to a GSM8K math problem.

Question: {question}
Gold answer (numeric): {gold}
Student response: {response}

PROCEDURE:
STEP A — recompute every "A op B = C" line; note where student first diverges.
STEP B — extract student's stated final answer; strip units/commas; compare to gold (±0.001).
STEP C — score band:
   IF correct → score in {{5..10}}: 9-10 clean, 7-8 minimal, 5-6 lucky.
   IF wrong   → score in {{0..4}}: 4 one slip, 3 multiple errors, 2 misread, 1 contradiction, 0 empty.

Return JSON: {{"is_correct": true|false, "score": <0-10 int>, "reasoning": "<brief>"}}
"""


_client = None
def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _client


def _classify_err(e):
    s = str(e).lower()
    if "429" in s or "rate" in s: return "rate_limit", str(e)[:200]
    if "503" in s or "unavailable" in s: return "503", str(e)[:200]
    if "timeout" in s or "deadline" in s: return "timeout", str(e)[:200]
    if "network" in s or "connection" in s: return "network", str(e)[:200]
    return "unknown", str(e)[:200]


def _extract_json(text):
    m = re.search(r"\{[^{}]*\"score\"[^{}]*\}", text, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(0))
    except: return None


def _judge_sync(question, gold, response):
    """Sync judge call with retry. Returns (score in [0,1] or None, is_correct, tag)."""
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, response=response[:1500])
    for attempt in range(1, MAX_RETRIES + 1):
        temp = 0.0 if attempt == 1 else min(0.3 + 0.1 * (attempt - 1), 1.0)
        cfg = types.GenerateContentConfig(
            temperature=temp,
            response_mime_type="application/json",
            http_options=types.HttpOptions(timeout=PER_CALL_TIMEOUT_S * 1000),
        )
        try:
            r = _get_client().models.generate_content(
                model=JUDGE_MODEL, contents=prompt, config=cfg)
        except Exception as e:
            tag, _ = _classify_err(e)
            if tag in _TRANSIENT_TAGS and attempt < MAX_RETRIES:
                time.sleep(min(2 ** (attempt - 1), 30) + random.random())
                continue
            return None, None, tag

        try: data = json.loads(r.text or "")
        except: data = _extract_json(r.text or "")
        if data is None:
            if attempt < MAX_RETRIES:
                time.sleep(min(0.5 * attempt, 5) + random.random() * 0.5)
                continue
            return None, None, "json_parse"

        try: score = float(data.get("score", -1))
        except: return None, None, "score_type"
        is_correct = data.get("is_correct", None)
        if is_correct not in (True, False): is_correct = None
        if 0 <= score <= 10:
            return score / 10.0, is_correct, "ok"
        return None, is_correct, "score_range"
    return None, None, "max_retries"


# Outcome guard: if extracted answer matches gold, force minimum reward
_BOXED = re.compile(r"\\boxed\{([^}]+)\}")
def _final_number(text):
    m = list(_BOXED.finditer(text))
    if m:
        s = m[-1].group(1).strip().replace(",", "").replace("$", "")
        try: return float(s)
        except: pass
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if nums:
        try: return float(nums[-1])
        except: pass
    return None


def _outcome_correct(response, gold):
    pred = _final_number(response)
    if pred is None: return False
    try: g = float(str(gold).replace(",", "").replace("$", ""))
    except: return False
    return abs(pred - g) < 0.001


async def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """verl reward fn entry point. Async — verl RewardLoopWorker uses asyncio.gather."""
    question = (extra_info or {}).get("question", "")
    score, is_correct, tag = await asyncio.to_thread(
        _judge_sync, question, ground_truth, solution_str)

    # Outcome guard: judge failed → fall back to rule-based correctness
    if score is None:
        return 1.0 if _outcome_correct(solution_str, ground_truth) else 0.0

    # Outcome consistency: if rule says correct but judge gave very low → bump to 0.5 floor
    rule_correct = _outcome_correct(solution_str, ground_truth)
    if rule_correct and score < 0.5:
        score = 0.5
    if (not rule_correct) and score > 0.5:
        score = min(score, 0.4)
    return float(score)

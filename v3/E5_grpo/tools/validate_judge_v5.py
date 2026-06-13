"""Re-judge the SAME 200 stratified rows with v5 prompt + Gemini 3.1 Flash-Lite Preview.

Inputs : /tmp/judge_validation_200.jsonl  (v4 baseline)
Outputs: /tmp/judge_validation_200_v5.jsonl  (v5 results, same schema + prompt_version)

Concurrency: 20 workers, per-call timeout 60s, up to 3 retries on transient errors.
If 503 rate >10%, fall back remaining rows to gemini-2.5-flash-lite.
"""
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai
from google.genai import types

INPUT = Path("/tmp/judge_validation_200.jsonl")
OUTPUT = Path("/tmp/judge_validation_200_v5.jsonl")
LIVE_LOG = Path("/tmp/judge_validation_200_v5.log")

JUDGE_MODEL_PRIMARY = "gemini-3.1-flash-lite-preview"
JUDGE_MODEL_FALLBACK = "gemini-2.5-flash-lite"
PER_CALL_TIMEOUT_S = 60
MAX_RETRIES = 3
N_WORKERS = 20
FALLBACK_503_THRESHOLD = 0.10  # 10%

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


_client = None
def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _client


def _classify_err(e):
    msg = str(e)[:200]
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return "rate_limit", msg
    if "503" in msg or "UNAVAILABLE" in msg:
        return "server_503", msg
    if "timeout" in msg.lower() or "deadline" in msg.lower():
        return "timeout", msg
    if "connect" in msg.lower() or "ssl" in msg.lower():
        return "network", msg
    return type(e).__name__, msg


_TRANSIENT = {"rate_limit", "server_503", "timeout", "network", "json_parse"}


def _try_extract_json(text):
    if not text:
        return None
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


def judge_one(question, gold, response, model):
    """Returns (data_or_none, raw_text, err_tag, err_msg, n_attempts, dt_total)."""
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, response=response[:1500])
    last_tag, last_msg = "unknown", ""
    raw_text = ""
    t_total_start = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        cfg = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            http_options=types.HttpOptions(timeout=PER_CALL_TIMEOUT_S * 1000),
        )
        try:
            r = _get_client().models.generate_content(
                model=model, contents=prompt, config=cfg,
            )
        except Exception as e:
            last_tag, last_msg = _classify_err(e)
            if last_tag in _TRANSIENT and attempt < MAX_RETRIES:
                time.sleep(min(2 ** (attempt - 1), 30) + random.random())
                continue
            return None, "", last_tag, last_msg, attempt, time.time() - t_total_start
        raw_text = r.text or ""
        try:
            data = json.loads(raw_text)
        except Exception:
            data = _try_extract_json(raw_text)
        if data is None:
            last_tag, last_msg = "json_parse", raw_text[:300]
            if attempt < MAX_RETRIES:
                time.sleep(min(0.5 * attempt, 5) + random.random() * 0.5)
                continue
            return None, raw_text, last_tag, last_msg, attempt, time.time() - t_total_start
        return data, raw_text, "ok", "", attempt, time.time() - t_total_start
    return None, raw_text, last_tag, last_msg, MAX_RETRIES, time.time() - t_total_start


def process_row(row, model):
    """Returns enriched row dict with v5 fields."""
    t0 = time.time()
    data, raw_text, tag, msg, n_attempts, _dt = judge_one(
        row["question"], row["gold"], row["response"], model
    )
    dt = time.time() - t0
    out = dict(row)
    # Drop v4 judge fields, replace with v5
    for k in ("judge_is_correct", "judge_score", "judge_reason", "judge_raw", "judge_error"):
        out.pop(k, None)
    out["prompt_version"] = "v5"
    out["judge_model"] = model
    out["latency_s"] = round(dt, 3)
    out["n_attempts"] = n_attempts
    out["err_tag"] = tag
    out["err_msg"] = msg if tag != "ok" else ""
    out["judge_raw"] = raw_text
    if data is not None:
        is_corr = data.get("is_correct")
        if is_corr not in (True, False):
            is_corr = None
        try:
            score = int(data.get("score", -1))
        except Exception:
            score = -1
        out["judge_is_correct"] = is_corr
        out["judge_score"] = score if 0 <= score <= 10 else None
        out["judge_reason"] = data.get("reason", "")[:500]
    else:
        out["judge_is_correct"] = None
        out["judge_score"] = None
        out["judge_reason"] = ""
    return out


def main():
    if not INPUT.exists():
        print(f"ERROR: {INPUT} missing", file=sys.stderr)
        sys.exit(1)
    rows = [json.loads(l) for l in open(INPUT)]
    print(f"Loaded {len(rows)} rows from {INPUT}")
    print(f"Primary model : {JUDGE_MODEL_PRIMARY}")
    print(f"Fallback model: {JUDGE_MODEL_FALLBACK} (if 503-rate >{FALLBACK_503_THRESHOLD*100:.0f}%)")
    print(f"Workers: {N_WORKERS}, per-call timeout: {PER_CALL_TIMEOUT_S}s, retries: {MAX_RETRIES}")
    print()

    # Phase 1: try primary model on ALL rows; track 503 rate
    results = {}
    n_503 = 0
    n_done = 0
    t_phase1 = time.time()

    LIVE_LOG.write_text("")  # clear

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(process_row, r, JUDGE_MODEL_PRIMARY): r["idx"] for r in rows}
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                out = fut.result(timeout=PER_CALL_TIMEOUT_S * (MAX_RETRIES + 1))
            except Exception as e:
                # last-ditch: build a failed row
                src = next(r for r in rows if r["idx"] == idx)
                out = dict(src)
                for k in ("judge_is_correct", "judge_score", "judge_reason", "judge_raw", "judge_error"):
                    out.pop(k, None)
                out.update(prompt_version="v5", judge_model=JUDGE_MODEL_PRIMARY,
                           latency_s=None, n_attempts=MAX_RETRIES,
                           err_tag="future_" + type(e).__name__, err_msg=str(e)[:200],
                           judge_raw="", judge_is_correct=None, judge_score=None, judge_reason="")
            results[idx] = out
            if out["err_tag"] == "server_503":
                n_503 += 1
            n_done += 1
            if n_done % 20 == 0:
                err = sum(1 for r in results.values() if r["err_tag"] != "ok")
                print(f"  [{n_done}/{len(rows)}] errs={err}, 503s={n_503}, elapsed={time.time()-t_phase1:.0f}s", flush=True)
                with open(LIVE_LOG, "a") as f:
                    f.write(f"{n_done}/{len(rows)} errs={err} 503s={n_503} t={time.time()-t_phase1:.0f}s\n")

    p1_dt = time.time() - t_phase1
    print(f"\nPhase 1 done in {p1_dt:.1f}s: 503s={n_503}/{len(rows)} ({n_503/len(rows)*100:.1f}%)")

    # Phase 2: re-judge any failed rows on fallback if 503 rate exceeded
    failed_idxs = [idx for idx, r in results.items() if r["err_tag"] != "ok"]
    used_fallback = False
    if n_503 / max(len(rows), 1) > FALLBACK_503_THRESHOLD and failed_idxs:
        used_fallback = True
        print(f"503 rate exceeds {FALLBACK_503_THRESHOLD*100:.0f}%; "
              f"falling back {len(failed_idxs)} failed rows to {JUDGE_MODEL_FALLBACK}")
        rows_by_idx = {r["idx"]: r for r in rows}
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(process_row, rows_by_idx[idx], JUDGE_MODEL_FALLBACK): idx
                    for idx in failed_idxs}
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    out = fut.result(timeout=PER_CALL_TIMEOUT_S * (MAX_RETRIES + 1))
                except Exception as e:
                    out = results[idx]
                    out["err_tag"] = "fallback_future_" + type(e).__name__
                    out["err_msg"] = str(e)[:200]
                results[idx] = out

    # Write output (preserve original idx ordering)
    OUTPUT.write_text("")
    with open(OUTPUT, "w") as f:
        for r in rows:
            out = results.get(r["idx"])
            if out is None:
                # shouldn't happen
                continue
            f.write(json.dumps(out) + "\n")

    # summary
    n_ok = sum(1 for r in results.values() if r["err_tag"] == "ok")
    n_fail = len(rows) - n_ok
    avg_lat = sum(r["latency_s"] or 0 for r in results.values()) / max(n_ok, 1)
    print(f"\nFinal: {n_ok}/{len(rows)} ok, {n_fail} failed; avg_latency={avg_lat:.2f}s; "
          f"used_fallback={used_fallback}")
    print(f"Written: {OUTPUT}")


if __name__ == "__main__":
    main()

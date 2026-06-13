"""Judge accuracy audit for v3 GRPO baseit_5e6 run.

1. Filter samples to baseit_5e6 (ts >= 2026-05-06T02:46).
2. Build gold map from train.jsonl using question prefix.
3. Stratified sample 30 entries.
4. Compute boxed-extraction gt_correct.
5. Re-judge with Gemini 3.1 Flash-Lite using v5 JUDGE_PROMPT.
6. Compare and report.
"""
import ast
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

random.seed(42)

_THIS = Path(__file__).resolve()
ROOT = _THIS.parents[3]  # .../fine-tuning
SAMPLES_FILE = _THIS.parent / "stage1_v3_samples.jsonl"
TRAIN_FILE = ROOT / "v3" / "shared" / "data" / "sft" / "train.jsonl"
OUT_FILE = _THIS.parent / "baseit_5e6_judge_audit.jsonl"

# v5 judge prompt verbatim from 02_grpo_judge.py
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


# ---------- Step 1: filter samples ----------
def load_baseit_samples():
    out = []
    with open(SAMPLES_FILE) as f:
        for line in f:
            d = json.loads(line)
            if d["ts"] >= "2026-05-06T02:46":
                out.append(d)
    return out


# ---------- Step 1b: build gold map from train.jsonl ----------
def build_gold_map():
    """key = first 150 chars of question, val = (full_question, gold_str, gold_float)"""
    m = {}
    boxed_re = re.compile(r"\\boxed\{([^}]+)\}")
    with open(TRAIN_FILE) as f:
        for line in f:
            d = json.loads(line)
            prompt = d["prompt"]
            completion = d["completion"]
            if isinstance(prompt, str):
                # JSON-encoded list
                prompt = ast.literal_eval(prompt) if prompt.startswith("[") else prompt
            if isinstance(completion, str):
                completion = ast.literal_eval(completion) if completion.startswith("[") else completion
            q = prompt[0]["content"] if isinstance(prompt, list) else prompt
            a = completion[0]["content"] if isinstance(completion, list) else completion
            # Extract gold from \boxed{...}
            mb = boxed_re.search(a)
            if not mb:
                continue
            gold_s = mb.group(1).strip()
            try:
                gf = float(gold_s.replace(",", "").replace("$", "").replace("%", "").strip())
            except Exception:
                gf = None
            # question before "\nPlease reason step by step"
            qkey = q.split("\nPlease reason")[0].strip()
            key = qkey[:150]
            if key not in m:
                m[key] = (qkey, gold_s, gf, a)  # store gold completion too
    return m


# ---------- Step 2: gt extraction ----------
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_final_number(text):
    """Try \\boxed{N}, then 'Answer:', then last number in last non-empty line."""
    if not text:
        return None
    # 1) boxed
    pieces = text.split("boxed{")
    if len(pieces) > 1:
        for piece in pieces[1:]:
            n = 0
            for i, ch in enumerate(piece):
                if ch == "{":
                    n += 1
                elif ch == "}":
                    n -= 1
                    if n < 0:
                        candidate = piece[:i].strip()
                        # Try to numericize
                        cleaned = re.sub(r"[\$,%\s]", "", candidate)
                        cleaned = cleaned.rstrip(".")
                        try:
                            return float(cleaned)
                        except Exception:
                            # Try to grab a number out of it
                            mm = NUM_RE.search(candidate)
                            if mm:
                                try:
                                    return float(mm.group(0).replace(",", ""))
                                except Exception:
                                    pass
                        break
    # 2) Answer:
    m = re.search(r"[Aa]nswer\s*[:\-=]?\s*\$?\s*(-?\d[\d,]*\.?\d*)", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except Exception:
            pass
    # 3) last number in last non-empty line
    lines = [ln for ln in text.strip().split("\n") if ln.strip()]
    if lines:
        nums = NUM_RE.findall(lines[-1])
        if nums:
            try:
                return float(nums[-1].replace(",", ""))
            except Exception:
                pass
    return None


def gt_check(completion, gold_float):
    val = extract_final_number(completion)
    if val is None or gold_float is None:
        return False, val
    return abs(val - gold_float) <= 0.001, val


# ---------- Step 3: re-judge via Gemini ----------
from google import genai
from google.genai import types

_client = None
def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _client


JUDGE_MODEL = "gemini-3.1-flash-lite-preview"


def judge_one(question, gold, response):
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, response=response[:1500])
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        http_options=types.HttpOptions(timeout=60000),
    )
    for attempt in range(3):
        try:
            r = _get_client().models.generate_content(
                model=JUDGE_MODEL, contents=prompt, config=cfg,
            )
            text = r.text or ""
            try:
                data = json.loads(text)
            except Exception:
                # tolerant
                s = text.strip()
                if s.startswith("```"):
                    s = s.split("\n", 1)[-1]
                    if s.endswith("```"):
                        s = s[:-3].strip()
                    elif "```" in s:
                        s = s.rsplit("```", 1)[0].strip()
                try:
                    data = json.loads(s)
                except Exception:
                    dec = json.JSONDecoder()
                    i = s.find("{")
                    data = None
                    while 0 <= i < len(s):
                        try:
                            data, _ = dec.raw_decode(s[i:])
                            break
                        except Exception:
                            i = s.find("{", i + 1)
            if data is None:
                continue
            try:
                score = int(data.get("score", -1))
            except Exception:
                continue
            ic = data.get("is_correct", None)
            reason = data.get("reason", "") or ""
            if not (0 <= score <= 10):
                continue
            return {"is_correct": bool(ic) if ic in (True, False) else None,
                    "score": score, "reason": reason, "raw": text[:500]}
        except Exception as e:
            time.sleep(min(2 ** attempt, 10))
            continue
    return {"is_correct": None, "score": None, "reason": None, "raw": "FAIL"}


# ---------- Main ----------
def main():
    print("Loading samples...", flush=True)
    samples = load_baseit_samples()
    print(f"  {len(samples)} baseit_5e6 samples", flush=True)

    print("Building gold map from train.jsonl...", flush=True)
    gold_map = build_gold_map()
    print(f"  {len(gold_map)} train questions in gold map", flush=True)

    # Resolve gold for each sample
    resolved = []
    n_unresolved = 0
    for s in samples:
        # prompt is truncated to 200 chars, key is 150 chars
        pk = s["prompt"][:150]
        if pk in gold_map:
            qfull, gold_s, gold_f, gold_completion = gold_map[pk]
        else:
            n_unresolved += 1
            continue
        s["_gold_full_q"] = qfull
        s["_gold_str"] = gold_s
        s["_gold_float"] = gold_f
        resolved.append(s)
    print(f"  resolved={len(resolved)}, unresolved={n_unresolved}", flush=True)

    # Step 1: stratified pick 30
    true_pool = [s for s in resolved if s.get("judge_is_correct") is True]
    false_pool = [s for s in resolved if s.get("judge_is_correct") is False]

    # Shuffle within each pool but ensure call_idx span
    def stratify(pool, n_target):
        # group by call_idx, pick round-robin
        by_call = {}
        for s in pool:
            by_call.setdefault(s["call_idx"], []).append(s)
        keys = sorted(by_call.keys())
        random.shuffle(keys)  # diversify call_idx selection
        for k in by_call:
            random.shuffle(by_call[k])
        out = []
        i = 0
        while len(out) < n_target and any(by_call[k] for k in keys):
            k = keys[i % len(keys)]
            if by_call[k]:
                out.append(by_call[k].pop())
            i += 1
        return out

    pick_true = stratify(true_pool, 15)
    pick_false = stratify(false_pool, 15)
    print(f"\nPicked: {len(pick_true)} True + {len(pick_false)} False", flush=True)
    print(f"  True call_idx: {sorted(set(s['call_idx'] for s in pick_true))}", flush=True)
    print(f"  False call_idx: {sorted(set(s['call_idx'] for s in pick_false))}", flush=True)
    print(f"  True logged scores: {sorted(s['judge_score'] for s in pick_true)}", flush=True)
    print(f"  False logged scores: {sorted(s['judge_score'] for s in pick_false)}", flush=True)

    picks = pick_true + pick_false
    print(f"\nTotal picks: {len(picks)}", flush=True)

    # Step 2: gt extraction
    for s in picks:
        ok, val = gt_check(s["completion"], s["_gold_float"])
        s["_gt_correct"] = ok
        s["_gt_extracted"] = val

    # Step 3: re-judge sequentially (small N, easier to debug)
    print("\nRe-judging via Gemini 3.1 Flash-Lite...", flush=True)
    t0 = time.time()
    # Use a small thread pool for speed
    def judge_task(idx_s):
        idx, s = idx_s
        return idx, judge_one(s["_gold_full_q"], s["_gold_str"], s["completion"])
    results = [None] * len(picks)
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(judge_task, (i, s)): i for i, s in enumerate(picks)}
        n_done = 0
        for f in as_completed(futs):
            idx, jd = f.result()
            results[idx] = jd
            n_done += 1
            if n_done % 5 == 0 or n_done == len(picks):
                print(f"  {n_done}/{len(picks)} done in {time.time()-t0:.1f}s", flush=True)

    for i, s in enumerate(picks):
        s["_re_judge"] = results[i]

    # Step 4: emit table + flags
    rows = []
    for s in picks:
        rj = s["_re_judge"]
        logged_ic = s.get("judge_is_correct")
        logged_sc = s.get("judge_score")
        gt = s["_gt_correct"]
        re_ic = rj.get("is_correct")
        re_sc = rj.get("score")
        re_reason = rj.get("reason") or ""
        # Flags
        is_correct_drift = (logged_ic != re_ic) if (logged_ic is not None and re_ic is not None) else False
        gt_disagree = (logged_ic != gt) if (logged_ic is not None) else False
        # Cap violation: wrong→score≥0.5 means logged is_correct=False but score≥0.5
        # Or correct→score≤0.4 (logged_score is normalized to [0,1])
        band_violation = False
        if logged_ic is False and isinstance(logged_sc, (int, float)) and logged_sc >= 0.5:
            band_violation = True
        if logged_ic is True and isinstance(logged_sc, (int, float)) and logged_sc <= 0.4:
            band_violation = True
        if re_ic is False and isinstance(re_sc, int) and re_sc >= 5:
            band_violation = True
        if re_ic is True and isinstance(re_sc, int) and re_sc <= 4:
            band_violation = True

        row = {
            "call_idx": s["call_idx"],
            "i": s["i"],
            "gold": s["_gold_str"],
            "gt_extracted": s["_gt_extracted"],
            "gt_correct": gt,
            "logged_ic": logged_ic,
            "logged_score": logged_sc,
            "re_ic": re_ic,
            "re_score": re_sc,
            "re_reason": re_reason[:120],
            "is_correct_drift": is_correct_drift,
            "gt_disagree": gt_disagree,
            "band_violation": band_violation,
            "completion_excerpt": s["completion"][:300],
            "question_excerpt": s["_gold_full_q"][:200],
        }
        rows.append(row)

    # Save
    with open(OUT_FILE, "w") as f:
        for s, row in zip(picks, rows):
            full = {**row,
                    "completion_full": s["completion"],
                    "question_full": s["_gold_full_q"],
                    "re_judge_raw": s["_re_judge"].get("raw", "")[:400],
                    "ts": s["ts"],
                    "config": s["config"]}
            f.write(json.dumps(full) + "\n")

    # Aggregate
    agg = {
        "n": len(rows),
        "logged_judge_acc": sum(1 for r in rows if r["logged_ic"] == r["gt_correct"]) / len(rows),
        "fresh_consistency": sum(1 for r in rows
                                 if r["logged_ic"] is not None and r["re_ic"] is not None
                                 and r["logged_ic"] == r["re_ic"]) / len(rows),
        "n_drift": sum(1 for r in rows if r["is_correct_drift"]),
        "n_gt_disagree": sum(1 for r in rows if r["gt_disagree"]),
        "n_band_violation": sum(1 for r in rows if r["band_violation"]),
    }

    print("\n" + "=" * 100, flush=True)
    print("AGGREGATE", flush=True)
    print(json.dumps(agg, indent=2), flush=True)

    print("\n" + "=" * 100, flush=True)
    print("ROWS", flush=True)
    print(f"{'call':>4} {'i':>4} {'gold':>6} {'gt':>4} {'log_ic':>6} {'log_s':>5} {'re_ic':>5} {'re_s':>4} flags  reason", flush=True)
    for r in rows:
        flags = []
        if r["is_correct_drift"]: flags.append("DRIFT")
        if r["gt_disagree"]: flags.append("GT_DIS")
        if r["band_violation"]: flags.append("BAND")
        fl = ",".join(flags) or "-"
        gtv = r["gt_extracted"]
        gt_disp = f"{int(gtv)}" if gtv is not None and gtv == int(gtv) else (f"{gtv}" if gtv is not None else "?")
        print(f"{r['call_idx']:>4} {r['i']:>4} {r['gold']:>6} "
              f"{gt_disp:>4} {str(r['logged_ic'])[:5]:>6} {r['logged_score']!s:>5} "
              f"{str(r['re_ic'])[:5]:>5} {r['re_score']!s:>4} {fl:>15} {r['re_reason'][:80]}",
              flush=True)

    return rows, agg


if __name__ == "__main__":
    main()

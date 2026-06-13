"""DS-CoT pass@k + maj@k eval on Gemma2-2B-IT (or +LoRA).

Sample K completions per question (temperature > 0), then compute:
  - pass@1, pass@2, pass@4, pass@8, pass@16, pass@32  (using unbiased estimator)
  - maj@k numeric (majority over 5-layer extracted preds)
  - maj@k boxed   (majority over boxed-only preds)

pass@k formula (Chen et al. HumanEval):
  pass@k = 1 - C(n-c, k) / C(n, k)
  where n = total samples, c = correct samples per question
  This is the UNBIASED probability of getting ≥1 correct in k samples
  drawn without replacement from the n-sample pool.

Same DS-CoT format as 01_eval_ds_cot.py:
  - user content = "{q}\\nPlease reason step by step, and put your final answer within \\boxed{}."
  - no system prompt; Gemma2 native chat template
  - 5-layer answer extraction; math_equal_numerical comparison

Usage:
  ~/vllm-env/bin/python v3/eval/03_eval_pass_at_k.py --ckpt base --k 32
  ~/vllm-env/bin/python v3/eval/03_eval_pass_at_k.py --ckpt base --k 32 --limit 64  # smoke
"""
import argparse
import datetime
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import regex
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
BASE_MODEL = ROOT / "models" / "gemma-2-2b-it"
DEFAULT_TEST = ROOT / "data" / "gsm8k" / "test.jsonl"
DEFAULT_OUTDIR = ROOT / "v3" / "E1_baseline" / "outputs"

USER_INSTRUCTION_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


# ============================================================================
# Answer extraction (verbatim from 01_eval_ds_cot.py)
# ============================================================================

def _fix_fracs(string):
    substrs = string.split("\\frac"); new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2: return string
                a, b = substr[0], substr[1]
                if b != "{":
                    new_str += "{" + a + "}{" + b + "}" + (substr[2:] if len(substr)>2 else "")
                else:
                    new_str += "{" + a + "}" + b + (substr[2:] if len(substr)>2 else "")
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2: return string
    a, b = string.split("/")
    try:
        if "sqrt" not in a: a = int(a)
        if "sqrt" not in b: b = int(b)
        if string == f"{a}/{b}": return f"\\frac{{{a}}}{{{b}}}"
    except Exception: pass
    return string


def _fix_sqrt(s):
    s = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", s)
    s = re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", s)
    return s


def strip_string(s):
    s = str(s).strip().replace("\n", "").rstrip(".")
    s = s.replace("\\!", "")
    if s.startswith("\\text{") and s.endswith("}"): s = s.split("{",1)[1][:-1]
    s = s.replace("tfrac","frac").replace("dfrac","frac").replace("cfrac","frac")
    s = s.replace("\\left","").replace("\\right","")
    _s = re.sub(r"\\text{.*?}$", "", s).strip()
    if _s and _s != s: s = _s
    s = s.replace("^{\\circ}","").replace("^\\circ","")
    s = regex.sub(r"\{(c|m)?m\}(\^(2|3))?", "", s).strip()
    s = regex.sub(r"p\.m\.$", "", s).strip()
    s = regex.sub(r"(\d)\s*t$", r"\1", s).strip()
    s = s.replace("\\$","").replace("$","").replace("x\\in","")
    s = s.replace("\\%","%")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    s = s.replace("\\cdot","")
    s = s.replace("infinity","\\infty")
    if "\\infty" not in s: s = s.replace("inf","\\infty")
    s = s.replace("+\\inity","\\infty").replace("\\mathbf","").replace("\\mathrm","")
    s = re.sub(r"\\mbox{.*?}", "", s)
    s = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", s)
    s = re.sub(r"(\d+)\.0+$", r"\1", s)
    if not s: return s
    if s[0] == ".": s = "0" + s
    s = _fix_sqrt(s); s = s.replace(" ","")
    s = _fix_fracs(s); s = _fix_a_slash_b(s)
    s = regex.sub(r"(\\|,|\.)+$", "", s)
    return s


def extract_boxed_answers(text):
    answers = []
    for piece in text.split("boxed{")[1:]:
        n = 0
        for i, ch in enumerate(piece):
            if ch == "{": n += 1
            elif ch == "}":
                n -= 1
                if n < 0:
                    if i+1 < len(piece) and piece[i+1] == "%":
                        answers.append(piece[:i+1])
                    else:
                        answers.append(piece[:i])
                    break
    return answers


def extract_program_output(s):
    if "```output" not in s: return ""
    s = s.split("```output")[-1]
    if "```" in s: s = s.split("```")[0]
    return s.strip()


def extract_answer(text):
    if "final answer is $" in text and "$. I hope" in text:
        ans = text.split("final answer is $",1)[1].split("$. I hope",1)[0].strip()
    elif "boxed" in text:
        boxed = extract_boxed_answers(text)
        ans = boxed[-1] if boxed else ""
    elif "he answer is" in text:
        ans = text.split("he answer is")[-1].strip()
    else:
        po = extract_program_output(text)
        if po: ans = po
        else:
            nums = re.findall(r"-?\d*\.?\d+", text.replace(",",""))
            ans = nums[-1] if nums else ""
    if not ans: return ""
    ans = ans.strip().split("\n")[0].lstrip(":").rstrip(".").rstrip("/")
    return strip_string(ans)


def extract_boxed_only(text):
    boxed = extract_boxed_answers(text)
    if not boxed: return ""
    return strip_string(boxed[-1].strip().split("\n")[0].rstrip(".").rstrip("/"))


def is_digit(s):
    try: float(str(s).replace(",","").replace("%",""))
    except Exception: return False
    return True


def parse_digit(s):
    s = str(s).replace(",","")
    return float(s[:-1])/100 if s.endswith("%") else float(s)


def math_equal_numerical(pred, ref, abs_tol=1e-3):
    if pred == "" or pred is None: return False
    if str(pred) == str(ref): return True
    if not (is_digit(pred) and is_digit(ref)): return False
    try:
        p = parse_digit(pred); r = parse_digit(ref)
    except Exception: return False
    for variant in (r, r/100, r*100):
        try:
            if math.isclose(p, variant, abs_tol=abs_tol): return True
        except Exception: continue
    return False


GSM_GOLD_RE = re.compile(r"####\s*(-?[0-9.,]+)")


def extract_gsm_gold(answer_field):
    m = GSM_GOLD_RE.search(answer_field)
    return strip_string(m.group(1).replace(",","").rstrip(".")) if m else ""


# ============================================================================
# pass@k unbiased estimator (Chen et al.)
# ============================================================================

def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of k samples drawn (without replacement)
    from a pool of n samples (with c correct) is correct. Numerically stable.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.exp(sum(math.log(n - c - i) - math.log(n - i) for i in range(k)))


def majority_vote(preds: list[str]) -> tuple[str, int]:
    """Plurality vote over non-empty preds. Returns (winner, count)."""
    valid = [p for p in preds if p != ""]
    if not valid:
        return "", 0
    cnt = Counter(valid)
    winner, count = cnt.most_common(1)[0]
    return winner, count


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="base")
    ap.add_argument("--base_model", default=None,
                    help="override base model path (default gemma-2-2b-it). For pretrain-base eval.")
    ap.add_argument("--k", type=int, default=32, help="samples per question")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_lora_rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_file", default=str(DEFAULT_TEST),
                    help="Path to test jsonl. Schema autodetect: GSM8K (question/answer) or MATH (problem/gold)")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTDIR),
                    help="Where to save pass_at_k_{ts}/ folder + append eval_log.jsonl")
    ap.add_argument("--task_tag", default=None,
                    help="Extra tag prefix in filename (e.g. 'math'). Default: inferred from test_file.")
    ap.add_argument("--chunk_size", type=int, default=0,
                    help="If >0, generate in chunks of N prompts, saving partials/ for resume. "
                         "0 = single-batch (legacy) mode.")
    ap.add_argument("--resume_dir", default=None,
                    help="Existing pass_at_k_*/ dir to resume from. Reads partials/. "
                         "If set, skips done chunks. Default: new dir each run.")
    args = ap.parse_args()

    test_path = Path(args.test_file)
    output_dir = Path(args.output_dir)
    eval_log = output_dir / "eval_log.jsonl"

    # --base_model override (default it). For pretrain-base experiments.
    base_model = Path(args.base_model) if getattr(args, "base_model", None) else BASE_MODEL

    # Resolve checkpoint. ckpt: "base" → no LoRA; <dir with adapter_config.json> → LoRA;
    #                      <full model dir> → treat as base, no LoRA.
    if args.ckpt == "base":
        lora_req = None
        tag = "base_" + base_model.name
    else:
        ckpt = Path(args.ckpt)
        if not ckpt.exists():
            raise SystemExit(f"missing: {ckpt}")
        if (ckpt / "adapter_config.json").exists():
            tag = ckpt.parent.name + "_" + ckpt.name
            lora_req = LoRARequest(lora_name=tag, lora_int_id=1, lora_path=str(ckpt))
        else:
            base_model = ckpt
            lora_req = None
            tag = "base_" + ckpt.name

    tok = AutoTokenizer.from_pretrained(base_model)
    with open(test_path) as f:
        data = [json.loads(line) for line in f]
    if args.limit:
        data = data[: args.limit]
    N = len(data)

    # Schema autodetect: GSM8K uses {question, answer}; MATH numeric uses {problem, gold}
    if "question" in data[0] and "answer" in data[0]:
        schema = "gsm8k"
        get_q = lambda ex: ex["question"]
        get_gold = lambda ex: extract_gsm_gold(ex["answer"])
    elif "problem" in data[0] and "gold" in data[0]:
        schema = "math"
        get_q = lambda ex: ex["problem"]
        get_gold = lambda ex: ex["gold"]
    else:
        raise SystemExit(f"unknown schema: keys={list(data[0].keys())}")
    print(f"[data] schema={schema} · path={test_path.name}")

    prompts = []
    for ex in data:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": get_q(ex) + USER_INSTRUCTION_SUFFIX}],
            tokenize=False, add_generation_prompt=True,
        )
        prompts.append(chat)

    print(f"[data] {N} samples · k={args.k} · total gens = {N * args.k}")
    print(f"[vLLM] base={base_model.name}  LoRA={tag if lora_req else 'none'}")
    llm = LLM(
        model=str(base_model),
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_lora=(lora_req is not None),
        max_lora_rank=args.max_lora_rank if lora_req else 8,
    )

    stop_ids = {tok.eos_token_id}
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    if eot is not None and eot != tok.unk_token_id:
        stop_ids.add(eot)

    sampling = SamplingParams(
        n=args.k, temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop_token_ids=list(stop_ids),
        seed=args.seed,
    )

    class _FakeOut:
        def __init__(self, text): self.text = text
    class _FakeRes:
        def __init__(self, texts): self.outputs = [_FakeOut(t) for t in texts]

    if args.chunk_size > 0:
        # ===== Chunked mode (resumable) =====
        ts0 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        task_tag_for_dir = args.task_tag or schema
        if args.resume_dir:
            out_dir = Path(args.resume_dir)
            if not out_dir.exists():
                raise SystemExit(f"resume_dir not found: {out_dir}")
            print(f"[resume] using existing dir: {out_dir}")
        else:
            out_dir = Path(args.output_dir) / f"pass_at_k_{task_tag_for_dir}_{ts0}"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[chunked] output dir: {out_dir}")
        partials_dir = out_dir / "partials"
        partials_dir.mkdir(parents=True, exist_ok=True)

        chunks = [(i, min(i + args.chunk_size, N)) for i in range(0, N, args.chunk_size)]
        print(f"[chunked] {len(chunks)} chunks × ≤{args.chunk_size} prompts each")
        all_texts_by_idx: dict[int, list[str]] = {}
        t0 = time.time()
        for ci, (s, e) in enumerate(chunks):
            pf = partials_dir / f"chunk_{ci:04d}.json"
            if pf.exists():
                try:
                    loaded = json.load(open(pf))
                    if loaded.get("end") == e and len(loaded.get("samples", [])) == (e - s):
                        for item in loaded["samples"]:
                            all_texts_by_idx[item["idx"]] = item["responses"]
                        print(f"  [resume] chunk {ci:04d} [{s}:{e}] loaded ({len(loaded['samples'])} prompts)")
                        continue
                    else:
                        print(f"  [warn] chunk {ci:04d} partial incomplete, regenerating")
                except Exception as ex:
                    print(f"  [warn] chunk {ci:04d} partial unreadable ({ex}), regenerating")
            chunk_prompts = prompts[s:e]
            tc = time.time()
            if lora_req:
                chunk_results = llm.generate(chunk_prompts, sampling, lora_request=lora_req)
            else:
                chunk_results = llm.generate(chunk_prompts, sampling)
            samples = []
            for local_i, res in enumerate(chunk_results):
                idx = s + local_i
                texts = [o.text for o in res.outputs]
                all_texts_by_idx[idx] = texts
                samples.append({"idx": idx, "responses": texts})
            dt_c = time.time() - tc
            with open(pf, "w", encoding="utf-8") as f:
                json.dump({"start": s, "end": e, "n": e - s, "samples": samples}, f)
            print(f"  [done] chunk {ci:04d} [{s}:{e}] {len(samples)} prompts {dt_c:.0f}s "
                  f"({(e-s)*args.k/dt_c:.1f} resp/s)")
        dt = time.time() - t0
        # Reassemble results in original order
        results = [_FakeRes(all_texts_by_idx[i]) for i in range(N)]
    else:
        # ===== Single-batch mode (legacy) =====
        print(f">>> sampling {args.k} responses per question on {N} prompts ...")
        t0 = time.time()
        if lora_req:
            results = llm.generate(prompts, sampling, lora_request=lora_req)
        else:
            results = llm.generate(prompts, sampling)
        dt = time.time() - t0
        out_dir = None  # will be set later in legacy code path
    total = N * args.k
    print(f"[gen] {dt:.1f}s · {total/dt:.1f} resp/s")

    # Score
    K = args.k
    levels = [k for k in [1, 2, 4, 8, 16, 32, 64, 128] if k <= K]

    pass_numeric = {k: 0.0 for k in levels}
    pass_boxed = {k: 0.0 for k in levels}
    sum_avg_per_resp_numeric = 0
    sum_avg_per_resp_boxed = 0
    n_maj_numeric = 0
    n_maj_boxed = 0
    n_total_resp = 0
    n_resp_have_boxed = 0
    samples_dump = []

    for ex, res in zip(data, results):
        gold = get_gold(ex)
        outs = res.outputs  # list of K
        n_total_resp += len(outs)

        # extract per-sample preds
        any_preds = []
        boxed_preds = []
        any_correct_count = 0
        boxed_correct_count = 0
        n_with_boxed = 0
        for o in outs:
            text = o.text
            ap_ = extract_answer(text)
            bp_ = extract_boxed_only(text)
            any_preds.append(ap_)
            boxed_preds.append(bp_)
            if bp_:
                n_with_boxed += 1
            if math_equal_numerical(ap_, gold):
                any_correct_count += 1
            if bp_ and math_equal_numerical(bp_, gold):
                boxed_correct_count += 1

        n_resp_have_boxed += n_with_boxed
        sum_avg_per_resp_numeric += any_correct_count / K
        sum_avg_per_resp_boxed += boxed_correct_count / K

        for k in levels:
            pass_numeric[k] += pass_at_k(K, any_correct_count, k)
            pass_boxed[k] += pass_at_k(K, boxed_correct_count, k)

        # maj@K
        maj_n_pred, _ = majority_vote(any_preds)
        maj_b_pred, _ = majority_vote(boxed_preds)
        if maj_n_pred and math_equal_numerical(maj_n_pred, gold):
            n_maj_numeric += 1
        if maj_b_pred and math_equal_numerical(maj_b_pred, gold):
            n_maj_boxed += 1

        # Save per-question structured info for ALL N questions (keep all K
        # responses so we can audit any sample). For K=32 this is ~30MB JSON
        # but worth it for full traceability.
        samples_dump.append({
            "question": get_q(ex),
            "gold": gold,
            "any_correct_per_K": any_correct_count,
            "boxed_correct_per_K": boxed_correct_count,
            "boxed_present_per_K": n_with_boxed,
            "maj_numeric_pred": maj_n_pred,
            "maj_boxed_pred": maj_b_pred,
            "any_preds": any_preds,
            "boxed_preds": boxed_preds,
            "responses": [o.text for o in outs],
        })

    metrics = {
        f"pass_at_{k}_numeric": round(pass_numeric[k] / N, 4) for k in levels
    }
    metrics.update({
        f"pass_at_{k}_boxed": round(pass_boxed[k] / N, 4) for k in levels
    })
    metrics["pass_at_1_numeric_avg"] = round(sum_avg_per_resp_numeric / N, 4)  # alt formula
    metrics["pass_at_1_boxed_avg"] = round(sum_avg_per_resp_boxed / N, 4)
    metrics[f"maj_at_{K}_numeric"] = round(n_maj_numeric / N, 4)
    metrics[f"maj_at_{K}_boxed"] = round(n_maj_boxed / N, 4)
    metrics["boxed_rate_per_resp"] = round(n_resp_have_boxed / n_total_resp, 4)
    metrics["duration_s"] = round(dt, 2)
    metrics["resp_per_sec"] = round(total / dt, 2)

    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    prompt_hash = hashlib.sha1(USER_INSTRUCTION_SUFFIX.encode()).hexdigest()[:8]
    task_tag = args.task_tag or schema
    config = {
        "engine": "vllm-ds-pass_at_k",
        "tag": tag,
        "ckpt": args.ckpt,
        "task": task_tag,
        "test_file": str(test_path),
        "samples": N,
        "K": args.k,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "prompt_hash": prompt_hash,
        "timestamp": timestamp,
    }

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if out_dir is None:
        # legacy single-batch mode — make a new dir
        out_dir = output_dir / f"pass_at_k_{task_tag}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{tag}_k{args.k}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"config": config, "metrics": metrics, "samples": samples_dump},
                  f, ensure_ascii=False, indent=2)

    log_row = {
        "timestamp": timestamp,
        "engine": "vllm-pass_at_k",
        "model": Path(base_model).name,
        "tag": tag,
        "ckpt": args.ckpt,
        "task": task_tag,
        "samples": N,
        "K": args.k,
        "temperature": args.temperature,
        "prompt_hash": prompt_hash,
        **{f"pass@{k}": metrics[f"pass_at_{k}_numeric"] for k in levels},
        f"maj@{K}_numeric": metrics[f"maj_at_{K}_numeric"],
        f"maj@{K}_boxed": metrics[f"maj_at_{K}_boxed"],
        "duration_s": metrics["duration_s"],
        "output": str(out_file.relative_to(ROOT)),
    }
    eval_log.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_row, ensure_ascii=False) + "\n")

    print()
    print(f"=== pass@k + maj@K eval · {tag} (K={K}, T={args.temperature}, {dt:.0f}s) ===")
    print(f"  per-resp boxed rate    : {n_resp_have_boxed/n_total_resp:.2%}")
    print(f"  pass@1 (any) avg       : {metrics['pass_at_1_numeric_avg']:.4f}")
    print()
    print(f"  {'k':>3}  {'pass@k numeric':>14}  {'pass@k boxed':>13}")
    for k in levels:
        print(f"  {k:>3}  {metrics[f'pass_at_{k}_numeric']:>13.4f}  {metrics[f'pass_at_{k}_boxed']:>12.4f}")
    print()
    print(f"  maj@{K} numeric : {metrics[f'maj_at_{K}_numeric']:.4f}")
    print(f"  maj@{K} boxed   : {metrics[f'maj_at_{K}_boxed']:.4f}")
    print(f"  saved: {out_file}")


if __name__ == "__main__":
    main()

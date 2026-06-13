"""DS-style Tool-Integrated Reasoning (TIR) eval on Gemma2-2B-IT.

Faithful port of v2/refs/DeepSeekMath/evaluation/infer/run_tool_integrated_eval.py
adapted to Gemma2's native chat template.

Inference loop (DS protocol):
  1. Generate with stop=["```output", EOS, <end_of_turn>]
  2. For each unfinished response:
     - hit EOS/EOT → mark finished
     - hit "```output" stop string → extract aggregated code (DS extract_code) →
       run in subprocess (timeout 10s) → append "\\n```output\\n{result}\\n```\\n"
       → re-queue for next iteration
  3. Repeat n_iters times (DS uses 2 by default for math; configurable)

Code aggregation across iterations: DS extract_code re-collects code from ALL
historical ```python blocks. For all but the LAST block: keeps only imports,
def lines, indented lines, and state-changing lines (no print). Last block is
kept in full. Subprocess re-runs the full aggregate each iteration. This is
how state is preserved across multi-step code without leaking between problems.

Final answer: same 5-layer chain as CoT eval (boxed → "the answer is" → ...).
Plus DS-style 'program_accuracy' from last ```output``` block.

Usage (vLLM env):
  ~/vllm-env/bin/python v3/eval/02_eval_ds_tir.py --ckpt base
  ~/vllm-env/bin/python v3/eval/02_eval_ds_tir.py --ckpt base --limit 32  # smoke
  ~/vllm-env/bin/python v3/eval/02_eval_ds_tir.py --ckpt base --max_iters 4
"""
import argparse
import datetime
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import regex
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
BASE_MODEL = ROOT / "models" / "gemma-2-2b-it"
TEST_FILE = ROOT / "data" / "gsm8k" / "test.jsonl"
OUTPUT_DIR = ROOT / "v3" / "E1_baseline" / "outputs"
EVAL_LOG = OUTPUT_DIR / "eval_log.jsonl"

# DS README L196 — exact instruction
USER_INSTRUCTION_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


# ============================================================================
# Answer extraction — copied verbatim from v3/eval/01_eval_ds_cot.py
# (which itself is faithful port of DSMath/evaluation/data_processing/answer_extraction.py)
# ============================================================================

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    if len(substr) > 2:
                        new_str += "{" + a + "}{" + b + "}" + substr[2:]
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        new_str += "{" + a + "}" + b + substr[2:]
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        if string == f"{a}/{b}":
            return f"\\frac{{{a}}}{{{b}}}"
    except Exception:
        pass
    return string


def _fix_sqrt(s):
    s = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", s)
    s = re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", s)
    return s


def strip_string(s):
    s = str(s).strip().replace("\n", "").rstrip(".")
    s = s.replace("\\!", "")
    if s.startswith("\\text{") and s.endswith("}"):
        s = s.split("{", 1)[1][:-1]
    s = s.replace("tfrac", "frac").replace("dfrac", "frac").replace("cfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    _s = re.sub(r"\\text{.*?}$", "", s).strip()
    if _s and _s != s:
        s = _s
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = regex.sub(r"\{(c|m)?m\}(\^(2|3))?", "", s).strip()
    s = regex.sub(r"p\.m\.$", "", s).strip()
    s = regex.sub(r"(\d)\s*t$", r"\1", s).strip()
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("x\\in", "")
    s = s.replace("\\%", "%").replace("\\%", "%")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    s = s.replace("\\cdot", "")
    s = s.replace("infinity", "\\infty")
    if "\\infty" not in s:
        s = s.replace("inf", "\\infty")
    s = s.replace("+\\inity", "\\infty")
    s = s.replace("\\mathbf", "").replace("\\mathrm", "")
    s = re.sub(r"\\mbox{.*?}", "", s)
    s = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", s)
    s = re.sub(r"(\d+)\.0+$", r"\1", s)
    if not s:
        return s
    if s[0] == ".":
        s = "0" + s
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    s = _fix_a_slash_b(s)
    s = regex.sub(r"(\\|,|\.)+$", "", s)
    return s


def extract_boxed_answers(text):
    answers = []
    for piece in text.split("boxed{")[1:]:
        n = 0
        for i, ch in enumerate(piece):
            if ch == "{":
                n += 1
            elif ch == "}":
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == "%":
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    return answers


def extract_program_output(s):
    """Output from the LAST ```output...``` block (DS code)."""
    if "```output" not in s:
        return ""
    s = s.split("```output")[-1]
    if "```" in s:
        s = s.split("```")[0]
    return s.strip()


def extract_answer(text):
    """5-layer fallback: 'final answer is $...' > boxed > 'the answer is' > program_output > last number."""
    if "final answer is $" in text and "$. I hope" in text:
        tmp = text.split("final answer is $", 1)[1]
        ans = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in text:
        boxed = extract_boxed_answers(text)
        ans = boxed[-1] if boxed else ""
    elif "he answer is" in text:
        ans = text.split("he answer is")[-1].strip()
    else:
        po = extract_program_output(text)
        if po:
            ans = po
        else:
            nums = re.findall(r"-?\d*\.?\d+", text.replace(",", ""))
            ans = nums[-1] if nums else ""
    if not ans:
        return ""
    ans = ans.strip().split("\n")[0].lstrip(":").rstrip(".").rstrip("/")
    return strip_string(ans)


def extract_boxed_only(text):
    boxed = extract_boxed_answers(text)
    if not boxed:
        return ""
    return strip_string(boxed[-1].strip().split("\n")[0].rstrip(".").rstrip("/"))


# ============================================================================
# Numerical comparison
# ============================================================================

def is_digit(s):
    try:
        float(str(s).replace(",", "").replace("%", ""))
        return True
    except Exception:
        return False


def parse_digit(s):
    s = str(s).replace(",", "")
    if s.endswith("%"):
        return float(s[:-1]) / 100
    return float(s)


def math_equal_numerical(pred, ref, abs_tol=1e-3):
    if pred == "" or pred is None:
        return False
    if str(pred) == str(ref):
        return True
    if not (is_digit(pred) and is_digit(ref)):
        return False
    try:
        p = parse_digit(pred)
        r = parse_digit(ref)
    except Exception:
        return False
    for variant in (r, r / 100, r * 100):
        try:
            if math.isclose(p, variant, abs_tol=abs_tol):
                return True
        except Exception:
            continue
    return False


GSM_GOLD_RE = re.compile(r"####\s*(-?[0-9.,]+)")


def extract_gsm_gold(answer_field):
    m = GSM_GOLD_RE.search(answer_field)
    return strip_string(m.group(1).replace(",", "").rstrip(".")) if m else ""


# ============================================================================
# DS extract_code — verbatim port of run_tool_integrated_eval.py:26-43
# ============================================================================

def extract_code(text):
    """Aggregate code from all ```python blocks. Last block in full,
    earlier blocks keep imports/defs/non-print state lines."""
    if not text.strip().endswith("```"):
        return ""
    if text.startswith("```python"):
        text = "hey\n" + text  # DS quirk: pad if leads with code block
    blocks = [block.split("```", 1)[0].strip()
              for block in text.split("```python") if "```" in block]
    blocks = [block for block in blocks if block]
    if not blocks:
        return ""
    code = []
    for block in blocks[:-1]:
        for line in block.split("\n"):
            if line.startswith("    ") or line.startswith("import") or line.startswith("def "):
                code.append(line)
            elif "print(" not in line:
                code.append(line)
    code = "\n".join(code) + "\n" + blocks[-1]
    return code.strip()


# ============================================================================
# Subprocess executor (replaces DS pebble.ProcessPool)
# ============================================================================

def safe_exec(code, timeout=10.0):
    """Run accumulated code in fresh subprocess. Returns (stdout, error_msg)."""
    if "input(" in code or "os.system(" in code:
        return "", "blocked: input/os.system not allowed"
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "", "Timeout Error"
    except Exception as e:
        return "", f"spawn:{type(e).__name__}"
    if r.returncode != 0:
        err = (r.stderr or "").strip().splitlines()
        msg = err[-1][:200] if err else f"rc={r.returncode}"
        return "", f"Runtime errors: {msg}"
    return r.stdout.strip(), ""


def truncate_result(s, max_chars=100):
    if len(s) <= max_chars:
        return s
    return s[:50] + "..." + s[-50:]


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="base", help="'base' or LoRA adapter dir")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_iters", type=int, default=2,
                    help="DS default for math = 2. More = more chances for code-fix loops")
    ap.add_argument("--max_new_tokens", type=int, default=1024,
                    help="per-iteration generation budget")
    ap.add_argument("--max_model_len", type=int, default=4096,
                    help="needs > sum of iter outputs + tool outputs")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_lora_rank", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--exec_timeout", type=float, default=10.0,
                    help="DS uses 10s")
    ap.add_argument("--exec_workers", type=int, default=16)
    args = ap.parse_args()

    # Resolve checkpoint
    if args.ckpt == "base":
        lora_req = None
        tag = "base_gemma-2-2b-it"
    else:
        ckpt = Path(args.ckpt)
        if not ckpt.exists():
            raise SystemExit(f"missing: {ckpt}")
        tag = ckpt.parent.name + "_" + ckpt.name
        lora_req = LoRARequest(lora_name=tag, lora_int_id=1, lora_path=str(ckpt))

    # Load data
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    with open(TEST_FILE) as f:
        data = [json.loads(line) for line in f]
    if args.limit:
        data = data[: args.limit]
    N = len(data)

    # Build initial prompts (Gemma2 chat template)
    initial_prompts = []
    for ex in data:
        user_msg = ex["question"] + USER_INSTRUCTION_SUFFIX
        chat = tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
        initial_prompts.append(chat)

    print(f"[data] {N} GSM8K samples")
    print(f"[chat] prompt_tokens≈{len(tok(initial_prompts[0])['input_ids'])}")

    # vLLM
    print(f"[vLLM] base={BASE_MODEL.name}  LoRA={tag if lora_req else 'none'}")
    llm = LLM(
        model=str(BASE_MODEL),
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_lora=(lora_req is not None),
        max_lora_rank=args.max_lora_rank if lora_req else 8,
    )

    eos_id = tok.eos_token_id
    eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
    stop_token_ids = sorted({tid for tid in [eos_id, eot_id] if tid is not None and tid != tok.unk_token_id})

    sampling = SamplingParams(
        n=1, temperature=args.temperature, top_p=1.0,
        max_tokens=args.max_new_tokens,
        stop=["```output"],          # DS stop string → triggers exec
        stop_token_ids=stop_token_ids,
    )

    # Per-problem state
    prompts_acc = list(initial_prompts)  # full prompt sent to LLM (grows)
    model_outputs = [""] * N             # accumulated assistant text only
    finished = [False] * N
    n_exec_calls = [0] * N

    # Iteration loop
    iter_stats = []
    t_start = time.time()
    for it in range(args.max_iters):
        unfinished_ids = [i for i in range(N) if not finished[i]]
        if not unfinished_ids:
            print(f"[iter {it}] all finished")
            break
        print(f"[iter {it}] generating for {len(unfinished_ids)} unfinished prompts ...")

        batch_prompts = [prompts_acc[i] for i in unfinished_ids]
        t0 = time.time()
        results = llm.generate(batch_prompts, sampling)
        gen_dt = time.time() - t0

        codes_to_run = []  # (orig_idx, code)
        for i_orig, res in zip(unfinished_ids, results):
            out = res.outputs[0]
            text = out.text

            model_outputs[i_orig] += text
            prompts_acc[i_orig] += text

            # finish detection: hit EOS/EOT?
            last_tok = out.token_ids[-1] if out.token_ids else None
            if last_tok in stop_token_ids:
                finished[i_orig] = True
                continue

            # else: stopped on "```output" stop string OR max_tokens
            # Try DS extract_code on accumulated assistant output
            code = extract_code(model_outputs[i_orig])
            if not code:
                # no code to exec → either max_tokens or weirdly stopped
                finished[i_orig] = True
                continue

            codes_to_run.append((i_orig, code))

        # Execute codes in parallel
        exec_dt = 0.0
        if codes_to_run:
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=args.exec_workers) as pool:
                results_exec = list(pool.map(
                    lambda c: safe_exec(c[1], args.exec_timeout),
                    codes_to_run,
                ))
            exec_dt = time.time() - t0
            for (i_orig, _), (out, err) in zip(codes_to_run, results_exec):
                exec_result = out if out else err if err else "(no output)"
                exec_result = truncate_result(exec_result, 100)
                tail = f"\n```output\n{exec_result.strip()}\n```\n"
                model_outputs[i_orig] += tail
                prompts_acc[i_orig] += tail
                n_exec_calls[i_orig] += 1

        iter_stats.append({
            "iter": it,
            "n_unfinished_in": len(unfinished_ids),
            "n_codes_run": len(codes_to_run),
            "gen_dt": round(gen_dt, 1),
            "exec_dt": round(exec_dt, 1),
        })
        print(f"[iter {it}] gen={gen_dt:.1f}s, codes_run={len(codes_to_run)}, exec={exec_dt:.1f}s")

    # mark anything still unfinished after max_iters as done (truncated)
    for i in range(N):
        finished[i] = True
    total_dt = time.time() - t_start

    # ---- Score ----
    n_boxed = n_boxed_correct = n_numeric = n_program_correct = 0
    samples = []
    for ex, mo in zip(data, model_outputs):
        gold = extract_gsm_gold(ex["answer"])
        boxed_pred = extract_boxed_only(mo)
        any_pred = extract_answer(mo)
        program_out = extract_program_output(mo)
        program_pred = strip_string(program_out.split("\n")[-1].strip()) if program_out else ""

        boxed_ok = (boxed_pred != "") and math_equal_numerical(boxed_pred, gold)
        numeric_ok = math_equal_numerical(any_pred, gold)
        program_ok = math_equal_numerical(program_pred, gold)

        if boxed_pred != "":
            n_boxed += 1
        if boxed_ok:
            n_boxed_correct += 1
        if numeric_ok:
            n_numeric += 1
        if program_ok:
            n_program_correct += 1

        samples.append({
            "question": ex["question"],
            "gold": gold,
            "boxed_pred": boxed_pred,
            "any_pred": any_pred,
            "program_pred": program_pred,
            "boxed_ok": boxed_ok,
            "numeric_ok": numeric_ok,
            "program_ok": program_ok,
            "n_exec_calls": n_exec_calls[data.index(ex)] if ex in data else 0,
            "model_output": mo,
        })

    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    prompt_hash = hashlib.sha1(USER_INSTRUCTION_SUFFIX.encode()).hexdigest()[:8]
    n_used_exec = sum(1 for c in n_exec_calls if c > 0)
    metrics = {
        "boxed_rate": round(n_boxed / N, 4),
        "boxed_accuracy": round(n_boxed_correct / N, 4),
        "numeric_accuracy": round(n_numeric / N, 4),     # 5-layer chain (DS literature number)
        "program_accuracy": round(n_program_correct / N, 4),  # last ```output``` only
        "exec_usage_rate": round(n_used_exec / N, 4),    # % problems that ran any code
        "mean_exec_calls_per_q": round(sum(n_exec_calls) / max(1, n_used_exec), 2),
        "total_duration_s": round(total_dt, 2),
        "iters_done": len(iter_stats),
    }
    config = {
        "engine": "vllm-ds-tir",
        "tag": tag,
        "ckpt": args.ckpt,
        "base_model": str(BASE_MODEL),
        "samples": N,
        "max_iters": args.max_iters,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "exec_timeout": args.exec_timeout,
        "temperature": args.temperature,
        "user_suffix": USER_INSTRUCTION_SUFFIX,
        "prompt_hash": prompt_hash,
        "no_system_prompt": True,
        "chat_template": "gemma2_native",
        "answer_extraction": "ds_priority(boxed>the_answer_is>last_number)",
        "code_aggregation": "ds_extract_code(state_lines+full_last_block)",
        "comparison": "math_equal_numerical(isclose 1e-3, percentage variants)",
        "iter_stats": iter_stats,
        "timestamp": timestamp,
    }

    # Save (truncate samples to first 200 for size)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / f"ds_tir_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{tag}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"config": config, "metrics": metrics, "samples": samples},
                  f, ensure_ascii=False, indent=2)

    log_row = {
        "timestamp": timestamp,
        "engine": "vllm-ds-tir",
        "model": Path(BASE_MODEL).name,
        "tag": tag,
        "ckpt": args.ckpt,
        "samples": N,
        "prompt_hash": prompt_hash,
        "boxed_rate": metrics["boxed_rate"],
        "boxed_accuracy": metrics["boxed_accuracy"],
        "numeric_accuracy": metrics["numeric_accuracy"],
        "program_accuracy": metrics["program_accuracy"],
        "exec_usage_rate": metrics["exec_usage_rate"],
        "max_iters": args.max_iters,
        "duration_s": metrics["total_duration_s"],
        "output": str(out_file.relative_to(ROOT)),
    }
    EVAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_row, ensure_ascii=False) + "\n")

    print()
    print(f"=== DS-TIR eval · {tag} (max_iters={args.max_iters}, total {total_dt:.1f}s) ===")
    print(f"exec_usage_rate    : {n_used_exec/N:.2%}  (problems where model ran any code)")
    print(f"mean_exec_calls/q  : {sum(n_exec_calls)/max(1,n_used_exec):.2f}  (avg code-run iters per code-using problem)")
    print(f"boxed_rate         : {n_boxed/N:.2%}  ({n_boxed}/{N})")
    print(f"boxed_accuracy     : {n_boxed_correct/N:.2%}  ({n_boxed_correct}/{N})  <-- strict")
    print(f"numeric_accuracy   : {n_numeric/N:.2%}  ({n_numeric}/{N})  <-- DS literature metric (5-layer)")
    print(f"program_accuracy   : {n_program_correct/N:.2%}  ({n_program_correct}/{N})  <-- last ```output``` only")
    print(f"saved: {out_file}")


if __name__ == "__main__":
    main()

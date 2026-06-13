"""DeepSeekMath-style CoT eval on Gemma2-2B-IT (or +LoRA adapter).

Replicates DSMath's CoT evaluation protocol (run_cot_eval.py + answer_extraction.py)
adapted for Gemma2-IT's native chat template:
  - user message = "{question}\\nPlease reason step by step, and put your final
    answer within \\boxed{}." (DSMath README L196)
  - NO system prompt (DSMath README L192: "we DO NOT RECOMMEND including the
    system prompt"; Gemma2 chat template has no system role anyway)
  - apply_chat_template (Gemma2 native: <start_of_turn>user / model) — equivalent
    to what DS does for DSMath-Instruct's "User: ... Assistant:" template
  - answer extraction priority (matches DSMath): boxed → "the answer is" →
    last number; with strip_string normalization
  - comparison via math_equal: numerical equality with 1e-3 tolerance, includes
    percentage variants (x, x/100, x*100)

Reports 3 metrics:
  - boxed_rate     : % responses that contain \\boxed{...}
  - boxed_accuracy : extracted-from-boxed pred matches gold (analog to our "strict")
  - numeric_acc    : pred extracted from any source matches gold

Usage (vLLM env in WSL2):
  ~/vllm-env/bin/python v2/eval/07_eval_ds_cot.py --ckpt base
  ~/vllm-env/bin/python v2/eval/07_eval_ds_cot.py --ckpt v2/checkpoints/.../checkpoint-800 --max_lora_rank 32
  ~/vllm-env/bin/python v2/eval/07_eval_ds_cot.py --ckpt base --limit 32   # smoke
"""

import argparse
import datetime
import hashlib
import json
import math
import re
import time
from pathlib import Path

import regex  # DS uses `regex` for some patterns; install via vllm env
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
BASE_MODEL = ROOT / "models" / "gemma-2-2b-it"
TEST_FILE = ROOT / "data" / "gsm8k" / "test.jsonl"
OUTPUT_DIR = ROOT / "v3" / "E1_baseline" / "outputs"
EVAL_LOG = OUTPUT_DIR / "eval_log.jsonl"

# DSMath README L196 — exact text for English math eval
USER_INSTRUCTION_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


# ============================================================================
# Answer extraction — copied from DSMath/evaluation/data_processing/answer_extraction.py
# (kept as faithful as possible; only minor cleanup)
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
    """Normalize a math answer string. Faithful port of DS strip_string."""
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
    """All `\\boxed{...}` content with proper brace matching (DS code)."""
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
    """Output between last ```output and following ```"""
    if "```output" not in s:
        return ""
    s = s.split("```output")[-1]
    if "```" in s:
        s = s.split("```")[0]
    return s.strip()


def extract_answer(text, exhaust=False):
    """DS extract_answer: priority = 'final answer is $...' > boxed > 'the answer is' > program_output > last number."""
    pred = []
    if "final answer is $" in text and "$. I hope" in text:
        tmp = text.split("final answer is $", 1)[1]
        pred = [tmp.split("$. I hope", 1)[0].strip()]
    elif "boxed" in text:
        pred = extract_boxed_answers(text)
    elif "he answer is" in text:
        pred = [text.split("he answer is")[-1].strip()]
    else:
        po = extract_program_output(text)
        if po:
            pred = [po]
        else:
            ans = re.findall(r"-?\d*\.?\d+", text.replace(",", ""))
            pred = [ans[-1]] if ans else []

    out = []
    for ans in pred:
        ans = ans.strip().split("\n")[0].lstrip(":").rstrip(".").rstrip("/")
        out.append(strip_string(ans))
    if exhaust:
        return out
    return out[-1] if out else ""


def extract_boxed_only(text):
    """Only return the last \\boxed{...} content (or '' if none); for boxed_rate."""
    boxed = extract_boxed_answers(text)
    return strip_string(boxed[-1].strip().split("\n")[0].rstrip(".").rstrip("/")) if boxed else ""


# ============================================================================
# Numerical comparison — DS math_equal simplified for GSM8K (digit-level only)
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
    """Faithful port of DS math_equal numerical branch (drops sympy/symbolic)."""
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
    # DS includes percentage variants
    for variant in (r, r / 100, r * 100):
        try:
            if math.isclose(p, variant, abs_tol=abs_tol):
                return True
        except Exception:
            continue
    return False


# ============================================================================
# Gold extraction (GSM8K-specific)
# ============================================================================

GSM_GOLD_RE = re.compile(r"####\s*(-?[0-9.,]+)")


def extract_gsm_gold(answer_field):
    m = GSM_GOLD_RE.search(answer_field)
    if not m:
        return ""
    return strip_string(m.group(1).replace(",", "").rstrip("."))


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="base", help="'base' or LoRA adapter dir")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=1024,
                    help="DS uses 1024; CoT + boxed needs more room than #### N")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--max_lora_rank", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
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

    # Build prompts: Gemma2 native chat template, user content = question + DS suffix
    prompts = []
    for ex in data:
        user_msg = ex["question"] + USER_INSTRUCTION_SUFFIX
        chat = tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
        prompts.append(chat)

    print(f"[data] {N} GSM8K samples · prompt_len≈{len(tok(prompts[0])['input_ids'])} tok")

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

    stop_ids = {tok.eos_token_id}
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    if eot is not None and eot != tok.unk_token_id:
        stop_ids.add(eot)

    sampling = SamplingParams(
        temperature=args.temperature, top_p=1.0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=list(stop_ids),
    )

    print(f">>> generating on {N} samples ...")
    t0 = time.time()
    if lora_req:
        results = llm.generate(prompts, sampling, lora_request=lora_req)
    else:
        results = llm.generate(prompts, sampling)
    dt = time.time() - t0

    # Score
    n_boxed = n_boxed_correct = n_numeric = 0
    samples = []
    for ex, res in zip(data, results):
        text = res.outputs[0].text
        gold = extract_gsm_gold(ex["answer"])

        boxed_pred = extract_boxed_only(text)        # only-boxed extraction
        any_pred = extract_answer(text)              # full DS priority chain

        boxed_ok = (boxed_pred != "") and math_equal_numerical(boxed_pred, gold)
        numeric_ok = math_equal_numerical(any_pred, gold)

        if boxed_pred != "":
            n_boxed += 1
        if boxed_ok:
            n_boxed_correct += 1
        if numeric_ok:
            n_numeric += 1

        samples.append({
            "question": ex["question"],
            "gold": gold,
            "boxed_pred": boxed_pred,
            "any_pred": any_pred,
            "boxed_ok": boxed_ok,
            "numeric_ok": numeric_ok,
            "response": text,
        })

    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    prompt_hash = hashlib.sha1(USER_INSTRUCTION_SUFFIX.encode()).hexdigest()[:8]
    metrics = {
        "boxed_rate": round(n_boxed / N, 4),
        "boxed_accuracy": round(n_boxed_correct / N, 4),
        "numeric_accuracy": round(n_numeric / N, 4),
        "duration_s": round(dt, 2),
        "req_per_sec": round(N / dt, 2),
    }
    config = {
        "engine": "vllm-ds-cot",
        "tag": tag,
        "ckpt": args.ckpt,
        "base_model": str(BASE_MODEL),
        "samples": N,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "user_suffix": USER_INSTRUCTION_SUFFIX,
        "prompt_hash": prompt_hash,
        "no_system_prompt": True,
        "chat_template": "gemma2_native",
        "answer_extraction": "ds_priority(boxed>the_answer_is>last_number)",
        "comparison": "math_equal_numerical(isclose 1e-3, percentage variants)",
        "timestamp": timestamp,
    }

    # Save
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / f"ds_cot_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{tag}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"config": config, "metrics": metrics, "samples": samples},
                  f, ensure_ascii=False, indent=2)

    log_row = {
        "timestamp": timestamp,
        "engine": "vllm-ds-cot",
        "model": Path(BASE_MODEL).name,
        "tag": tag,
        "ckpt": args.ckpt,
        "samples": N,
        "prompt_hash": prompt_hash,
        "boxed_rate": metrics["boxed_rate"],
        "boxed_accuracy": metrics["boxed_accuracy"],
        "numeric_accuracy": metrics["numeric_accuracy"],
        "duration_s": metrics["duration_s"],
        "output": str(out_file.relative_to(ROOT)),
    }
    with open(EVAL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_row, ensure_ascii=False) + "\n")

    print()
    print(f"=== DS-CoT eval · {tag} ({dt:.1f}s, {N/dt:.1f} req/s) ===")
    print(f"boxed_rate      : {n_boxed/N:.2%}  ({n_boxed}/{N})")
    print(f"boxed_accuracy  : {n_boxed_correct/N:.2%}  ({n_boxed_correct}/{N})  <-- strict analog")
    print(f"numeric_accuracy: {n_numeric/N:.2%}  ({n_numeric}/{N})  <-- with fallback chain")
    print(f"saved: {out_file}")


if __name__ == "__main__":
    main()

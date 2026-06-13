"""DS-style 5-layer answer extraction + math_equal numerical comparison.

Pure-Python utilities (no torch/vllm/transformers deps) so they can be imported
from train scripts (train-env) without pulling vllm. Mirror the SFT-eval helpers
in v3/E2_sft/eval/03_eval_pass_at_k.py — these are the same functions, factored
out so GRPO reward computation can reuse them.

Layers tried in order:
  1. "final answer is $...$. I hope"
  2. \\boxed{...}        ← primary
  3. "he answer is ..."
  4. ```output ... ```   (program output)
  5. last number in text
"""
import math
import re

import regex


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
    """DS 5-layer fallback chain → final answer string."""
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
    """Strict: only \\boxed{} extraction."""
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
    """Robust numerical compare: handles units, percent, scaling 100x."""
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


def gold_from_completion(completion_messages):
    """Extract numeric gold from assistant content's \\boxed{N}."""
    text = completion_messages[0]["content"]
    m = re.search(r"\\boxed\{([^{}]+)\}", text)
    if not m:
        return ""
    s = m.group(1).strip().replace(",", "").rstrip(".")
    return s

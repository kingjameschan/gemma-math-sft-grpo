"""D_dev metrics evaluator for one checkpoint (Phase 2 / E2 SFT analysis).

For one ckpt, compute on D_dev (500 samples):
  - val_nll       : mean cross-entropy on completion tokens (Schulman-style)
  - pass@1        : greedy generate + 5-layer extract + math_equal
  - pass@8        : K=8 sample + count correct
  - mean_length   : mean response token count
  - boxed_rate    : fraction of responses containing \\boxed{}

vLLM-based: single LLM session per ckpt, 3 generation passes.

Usage:
  ~/vllm-env/bin/python v3/eval/04_dev_metrics.py --ckpt v3/checkpoints/sft_lr1e-4_r64/checkpoint-50
  ~/vllm-env/bin/python v3/eval/04_dev_metrics.py --ckpt base    # base IT for D_dev anchor
"""
import argparse
import datetime
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

# Reuse extraction & math_equal from 03_eval_pass_at_k
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
_mod = import_module("03_eval_pass_at_k")
extract_answer = _mod.extract_answer
extract_boxed_only = _mod.extract_boxed_only
math_equal_numerical = _mod.math_equal_numerical
pass_at_k = _mod.pass_at_k

ROOT = Path(__file__).resolve().parents[3]
BASE_MODEL = ROOT / "models" / "gemma-2-2b-it"
DEV_FILE = ROOT / "v3" / "shared" / "data" / "sft" / "dev.jsonl"
OUTPUT_DIR = ROOT / "v3" / "E2_sft" / "outputs" / "dev_eval"
EVAL_LOG = ROOT / "v3" / "shared" / "eval_log.jsonl"

USER_INSTRUCTION_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


def gold_from_completion(completion_messages: list[dict]) -> str:
    """Extract numeric gold from assistant content's \\boxed{N}."""
    text = completion_messages[0]["content"]
    m = re.search(r"\\boxed\{([^{}]+)\}", text)
    if not m:
        return ""
    s = m.group(1).strip().replace(",", "").rstrip(".")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help='ckpt path or "base" for base Gemma2-IT')
    ap.add_argument("--dev_file", default=str(DEV_FILE))
    ap.add_argument("--out_dir", default=str(OUTPUT_DIR))
    ap.add_argument("--k", type=int, default=16,
                    help="samples for entropy + pass@K (default 16, was 8 before adding AE)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_model_len", type=int, default=1280)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_lora_rank", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save_samples", action="store_true",
                    help="save full sampling records: greedy responses + K=16 sample answers + K=16 full text (~10MB/ckpt)")
    args = ap.parse_args()

    # Resolve ckpt
    if args.ckpt == "base":
        lora_req = None
        tag = "base_gemma-2-2b-it"
    else:
        ckpt = Path(args.ckpt)
        if not ckpt.exists():
            raise SystemExit(f"missing ckpt: {ckpt}")
        tag = ckpt.parent.name + "_" + ckpt.name
        lora_req = LoRARequest(lora_name=tag, lora_int_id=1, lora_path=str(ckpt))

    # Load D_dev
    with open(args.dev_file) as f:
        dev = [json.loads(line) for line in f]
    if args.limit:
        dev = dev[: args.limit]
    N = len(dev)

    # Build inputs
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    gen_prompts = []           # for pass@1 / pass@8 generation
    nll_full_texts = []        # for val_nll (prompt + gold completion)
    golds = []                 # gold strings for math_equal

    for ex in dev:
        # Generation prompt: user message + chat template + add_generation_prompt
        gen_prompt = tok.apply_chat_template(
            ex["prompt"], tokenize=False, add_generation_prompt=True,
        )
        gen_prompts.append(gen_prompt)

        # NLL input: full chat (prompt + completion)
        nll_text = tok.apply_chat_template(
            ex["prompt"] + ex["completion"],
            tokenize=False, add_generation_prompt=False,
        )
        nll_full_texts.append(nll_text)

        golds.append(gold_from_completion(ex["completion"]))

    print(f"[data] {N} D_dev samples · ckpt={tag}")
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

    # ---- Pass 1: greedy → pass@1, length, boxed_rate ----
    greedy_params = SamplingParams(
        n=1, temperature=0.0, top_p=1.0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=list(stop_ids),
        seed=args.seed,
    )
    print(">>> greedy generation ...")
    t0 = time.time()
    if lora_req:
        greedy_results = llm.generate(gen_prompts, greedy_params, lora_request=lora_req)
    else:
        greedy_results = llm.generate(gen_prompts, greedy_params)
    t_greedy = time.time() - t0
    print(f"[greedy] {t_greedy:.1f}s")

    n_greedy_correct = 0
    n_greedy_boxed = 0
    sum_resp_len = 0
    n_resp = 0
    greedy_responses = []   # full text per Q (1 per Q)
    greedy_extracted = []   # extracted answer per Q
    for ex, res, gold in zip(dev, greedy_results, golds):
        out = res.outputs[0]
        text = out.text
        # 5-layer extract
        ap_ = extract_answer(text)
        bp_ = extract_boxed_only(text)
        if math_equal_numerical(ap_, gold):
            n_greedy_correct += 1
        if bp_:
            n_greedy_boxed += 1
        # token-level length (use vLLM's output token count)
        sum_resp_len += len(out.token_ids)
        n_resp += 1
        greedy_responses.append(text)
        greedy_extracted.append(ap_)

    pass_at_1 = n_greedy_correct / N
    boxed_rate = n_greedy_boxed / N
    mean_length = sum_resp_len / n_resp

    # ---- Pass 2: K=8 sampling → pass@8 ----
    sample_params = SamplingParams(
        n=args.k, temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop_token_ids=list(stop_ids),
        seed=args.seed,
    )
    print(f">>> sampling K={args.k} ...")
    t0 = time.time()
    if lora_req:
        sample_results = llm.generate(gen_prompts, sample_params, lora_request=lora_req)
    else:
        sample_results = llm.generate(gen_prompts, sample_params)
    t_sample = time.time() - t0
    print(f"[sample K={args.k}] {t_sample:.1f}s")

    K = args.k
    pass_k_sum = 0.0
    sample_correct_per_q = []
    per_q_entropy = []          # answer entropy per question (nats)
    per_q_answers = []          # extracted answers per question (K per Q)
    per_q_responses = []        # full response text per question (K per Q)
    for ex, res, gold in zip(dev, sample_results, golds):
        n_correct = 0
        answers = []
        responses = []
        for o in res.outputs:
            ap_ = extract_answer(o.text)
            answers.append(ap_)
            responses.append(o.text)
            if math_equal_numerical(ap_, gold):
                n_correct += 1
        pass_k_sum += pass_at_k(K, n_correct, K)
        sample_correct_per_q.append(n_correct)
        # Answer entropy on this question's K samples
        # H = -Σ p(v) log p(v) where p(v) = count(v) / K (nats)
        freq = Counter(answers)
        H = 0.0
        for c in freq.values():
            p = c / K
            if p > 0:
                H -= p * math.log(p)
        per_q_entropy.append(H)
        per_q_answers.append(answers)
        per_q_responses.append(responses)

    pass_at_K = pass_k_sum / N
    mean_entropy = sum(per_q_entropy) / max(1, len(per_q_entropy))
    max_entropy_K = math.log(K)   # ceiling for normalization (log(K))

    # ---- Pass 3: val_nll via prompt_logprobs ----
    nll_params = SamplingParams(
        n=1, temperature=0.0, max_tokens=1,    # 1 dummy token
        prompt_logprobs=1,                     # request logprobs of prompt tokens
        seed=args.seed,
    )
    print(">>> val_nll via prompt_logprobs ...")
    t0 = time.time()
    if lora_req:
        nll_results = llm.generate(nll_full_texts, nll_params, lora_request=lora_req)
    else:
        nll_results = llm.generate(nll_full_texts, nll_params)
    t_nll = time.time() - t0
    print(f"[nll] {t_nll:.1f}s")

    # Compute mean NLL on completion tokens for each sample
    sum_nll = 0.0
    sum_completion_tokens = 0
    n_skipped = 0
    for ex, res in zip(dev, nll_results):
        prompt_only = tok.apply_chat_template(
            ex["prompt"], tokenize=True, add_generation_prompt=True,
        )
        n_prompt_tok = len(prompt_only)
        # res.prompt_logprobs is List[Optional[Dict[token_id, Logprob]]]
        # Index 0 is None; index i corresponds to logp(token_i | tokens_<i)
        prompt_logprobs = res.prompt_logprobs
        if prompt_logprobs is None:
            n_skipped += 1
            continue
        # Sum logprob of actual tokens at positions n_prompt_tok ... end-1
        # (these are the completion tokens)
        prompt_token_ids = res.prompt_token_ids
        for pos in range(n_prompt_tok, len(prompt_token_ids)):
            d = prompt_logprobs[pos]
            if d is None:
                continue
            actual_token_id = prompt_token_ids[pos]
            # Logprob of actual token
            lp_obj = d.get(actual_token_id)
            if lp_obj is None:
                # actual token was not in top-1; rarely happens
                continue
            sum_nll += -lp_obj.logprob
            sum_completion_tokens += 1

    val_nll = sum_nll / max(1, sum_completion_tokens)

    # ---- Compose result ----
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    metrics = {
        "val_nll": round(val_nll, 5),
        "pass_at_1": round(pass_at_1, 5),
        f"pass_at_{K}": round(pass_at_K, 5),
        "mean_entropy": round(mean_entropy, 5),
        "max_entropy_K": round(max_entropy_K, 5),
        "mean_response_length": round(mean_length, 1),
        "boxed_rate": round(boxed_rate, 5),
        "n_dev": N,
        "n_greedy_correct": n_greedy_correct,
        "n_greedy_boxed": n_greedy_boxed,
        "per_question_entropy": [round(h, 5) for h in per_q_entropy],
        "per_question_correct_count": sample_correct_per_q,
        "duration_s": {
            "greedy": round(t_greedy, 1),
            f"sample_k{K}": round(t_sample, 1),
            "nll": round(t_nll, 1),
            "total": round(t_greedy + t_sample + t_nll, 1),
        },
        "config": {
            "ckpt": args.ckpt,
            "tag": tag,
            "k": K,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "seed": args.seed,
            "n_skipped_nll": n_skipped,
        },
        "timestamp": timestamp,
    }
    if args.save_samples:
        metrics["greedy_responses"] = greedy_responses                 # 500 × full text
        metrics["greedy_extracted"] = greedy_extracted                  # 500 × extracted answer
        metrics["per_sample_answers"] = per_q_answers                  # 500 × K extracted answers
        metrics["per_sample_responses"] = per_q_responses              # 500 × K full text

    print()
    print(f"=== {tag} ===")
    print(f"  val_nll          : {metrics['val_nll']:.4f}")
    print(f"  pass@1 (greedy)  : {metrics['pass_at_1']*100:.2f}%")
    print(f"  pass@{K} (sample) : {metrics[f'pass_at_{K}']*100:.2f}%")
    print(f"  mean_entropy     : {metrics['mean_entropy']:.4f} nats (max={max_entropy_K:.4f})")
    print(f"  mean_length      : {metrics['mean_response_length']:.1f} tok")
    print(f"  boxed_rate       : {metrics['boxed_rate']*100:.2f}%")
    print(f"  total_time       : {metrics['duration_s']['total']:.1f}s")

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{tag}.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nsaved: {out_file}")

    # Append to eval_log
    EVAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "timestamp": timestamp,
        "engine": "vllm-dev-metrics",
        "ckpt": args.ckpt,
        "tag": tag,
        "k": K,
        "n_dev": N,
        "val_nll": metrics["val_nll"],
        "pass_at_1": metrics["pass_at_1"],
        f"pass_at_{K}": metrics[f"pass_at_{K}"],
        "mean_entropy": metrics["mean_entropy"],
        "mean_length": metrics["mean_response_length"],
        "output": str(out_file.relative_to(ROOT)) if out_file.is_relative_to(ROOT) else str(out_file),
    }
    with open(EVAL_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


if __name__ == "__main__":
    main()

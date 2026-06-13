"""Fastgrid eval: greedy pass@1 on D_dev for all fastgrid ckpts.

Single vLLM session, swap LoRA per ckpt to amortize base-model load.
Only computes pass@1 (greedy), boxed_rate, mean_length — skips K-sampling
and val_nll for speed (~30s/ckpt instead of ~150s).

Outputs: v3/outputs/fastgrid_eval/sft_lr{X}_r{Y}_checkpoint-{N}.json

Usage:
  ~/vllm-env/bin/python v3/eval/05_fastgrid_eval.py
  ~/vllm-env/bin/python v3/eval/05_fastgrid_eval.py --ckpt_root v3/checkpoints/fastgrid
"""
import argparse
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
_mod = import_module("03_eval_pass_at_k")
extract_answer = _mod.extract_answer
extract_boxed_only = _mod.extract_boxed_only
math_equal_numerical = _mod.math_equal_numerical

_dev = import_module("04_dev_metrics")
gold_from_completion = _dev.gold_from_completion

ROOT = Path(__file__).resolve().parents[3]
BASE_MODEL = ROOT / "models" / "gemma-2-2b-it"
DEV_FILE = ROOT / "v3" / "shared" / "data" / "sft" / "dev.jsonl"
OUTPUT_DIR = ROOT / "v3" / "E2_sft" / "outputs" / "fastgrid_eval"
CKPT_ROOT = ROOT / "v3" / "E2_sft" / "checkpoints" / "fastgrid"


def collect_ckpts(root: Path):
    ckpts = []
    if not root.exists():
        return ckpts
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        for ck in sorted(d.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[-1])):
            ckpts.append(ck)
    return ckpts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", default=str(CKPT_ROOT))
    ap.add_argument("--out_dir", default=str(OUTPUT_DIR))
    ap.add_argument("--dev_file", default=str(DEV_FILE))
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_model_len", type=int, default=1280)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_lora_rank", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allowed_steps", default=None,
                    help="comma-separated list of step numbers to eval; default all")
    args = ap.parse_args()

    ckpt_root = Path(args.ckpt_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = collect_ckpts(ckpt_root)
    if args.allowed_steps:
        allowed = {int(s) for s in args.allowed_steps.split(",")}
        ckpts = [c for c in ckpts if int(c.name.split("-")[-1]) in allowed]
        print(f"[filter] keeping {len(ckpts)} ckpts at steps {sorted(allowed)}")
    print(f"[scan] found {len(ckpts)} fastgrid ckpts under {ckpt_root}")

    todo = [c for c in ckpts
            if not (out_dir / f"{c.parent.name}_{c.name}.json").exists()]
    print(f"[plan] {len(todo)}/{len(ckpts)} need eval (skipping existing)")
    if not todo:
        print("[done] nothing to do")
        return

    # Load D_dev once
    with open(args.dev_file) as f:
        dev = [json.loads(line) for line in f]
    N = len(dev)
    print(f"[data] D_dev N={N}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    gen_prompts = []
    golds = []
    for ex in dev:
        gp = tok.apply_chat_template(
            ex["prompt"], tokenize=False, add_generation_prompt=True,
        )
        gen_prompts.append(gp)
        golds.append(gold_from_completion(ex["completion"]))

    # Single vLLM session for all ckpts
    print(f"[vLLM] loading base + enable_lora (max_lora_rank={args.max_lora_rank})...")
    t_load = time.time()
    llm = LLM(
        model=str(BASE_MODEL),
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
    )
    print(f"[vLLM] loaded in {time.time() - t_load:.1f}s")

    stop_ids = {tok.eos_token_id}
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    if eot is not None and eot != tok.unk_token_id:
        stop_ids.add(eot)

    greedy = SamplingParams(
        n=1, temperature=0.0, top_p=1.0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=list(stop_ids),
        seed=args.seed,
    )

    t_grid = time.time()
    for i, ck in enumerate(todo, 1):
        tag = f"{ck.parent.name}_{ck.name}"
        print(f"\n[{i}/{len(todo)}] {tag}")
        lora_req = LoRARequest(lora_name=tag, lora_int_id=i, lora_path=str(ck))

        t0 = time.time()
        results = llm.generate(gen_prompts, greedy, lora_request=lora_req)
        n_correct, n_boxed, lengths = 0, 0, []
        for ex, res, gold in zip(dev, results, golds):
            out = res.outputs[0]
            text = out.text
            ap_ = extract_answer(text)
            bp_ = extract_boxed_only(text)
            if math_equal_numerical(ap_, gold):
                n_correct += 1
            if bp_:
                n_boxed += 1
            lengths.append(len(out.token_ids))

        pass1 = n_correct / N
        boxed = n_boxed / N
        mean_len = sum(lengths) / N
        dur = time.time() - t0

        out = {
            "config": {
                "tag": tag,
                "ckpt": (str(ck.resolve().relative_to(ROOT)) if ROOT in ck.resolve().parents else str(ck)),
                "lora_rank_dir": ck.parent.name,
                "step": int(ck.name.split("-")[-1]),
            },
            "pass_at_1": round(pass1, 5),
            "boxed_rate": round(boxed, 5),
            "mean_response_length": round(mean_len, 1),
            "n_dev": N,
            "n_greedy_correct": n_correct,
            "n_greedy_boxed": n_boxed,
            "duration_s": round(dur, 1),
        }
        out_file = out_dir / f"{tag}.json"
        json.dump(out, open(out_file, "w"), indent=2)
        print(f"   pass@1={pass1*100:.2f}%  boxed={boxed*100:.1f}%  "
              f"mean_len={mean_len:.0f}  ({dur:.0f}s)")

    print(f"\n=== fastgrid eval done in {(time.time() - t_grid)/60:.1f} min ===")


if __name__ == "__main__":
    main()

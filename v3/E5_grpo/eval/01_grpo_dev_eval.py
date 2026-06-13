"""GRPO fastgrid D_dev pass@1 eval.

Single vLLM session, swap LoRA per ckpt. Imports answer-extraction helpers
from v3/shared/ (no module-name-starts-with-digit problem).

Iterates: for each subdir of --ckpt_root, eval each checkpoint-N inside.
Output: <out_dir>/<run_name>_checkpoint-<N>.json

Usage:
  python3 v3/E5_grpo/eval/01_grpo_dev_eval.py \
    --ckpt_root v3/E5_grpo/checkpoints/fastgrid/stage1 \
    --out_dir   v3/E5_grpo/outputs/fastgrid/stage1_eval \
    --dev_file  v3/shared/data/sft/dev.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
BASE_MODEL = ROOT / "models" / "gemma-2-2b-it"

sys.path.insert(0, str(ROOT / "v3" / "shared"))
from answer_extraction import extract_answer, math_equal_numerical, gold_from_completion


def collect_ckpts(root: Path):
    out = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        for ck in sorted(d.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[-1])):
            out.append(ck)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--dev_file",  required=True)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_model_len", type=int, default=1280)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_lora_rank", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ckpt_root = Path(args.ckpt_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = collect_ckpts(ckpt_root)
    print(f"[scan] found {len(ckpts)} ckpts under {ckpt_root}")

    todo = [c for c in ckpts
            if not (out_dir / f"{c.parent.name}_{c.name}.json").exists()]
    print(f"[plan] {len(todo)}/{len(ckpts)} need eval (skipping existing)")
    if not todo:
        print("[done] nothing to do")
        return

    with open(args.dev_file) as f:
        dev = [json.loads(l) for l in f]
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
        n_correct = 0
        n_boxed = 0
        lengths = []
        for ex, res, gold in zip(dev, results, golds):
            text = res.outputs[0].text
            pred = extract_answer(text)
            if math_equal_numerical(pred, gold):
                n_correct += 1
            if "\\boxed{" in text:
                n_boxed += 1
            lengths.append(len(res.outputs[0].token_ids))

        pass1 = n_correct / N
        boxed = n_boxed / N
        mean_len = sum(lengths) / N
        dur = time.time() - t0

        out = {
            "config": {
                "tag": tag,
                "ckpt": str(ck),
                "run_dir": ck.parent.name,
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

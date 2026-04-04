"""
GCP 独立评测脚本：Gemma2-2B checkpoint 三指标评测。
支持 SFT / DPO / GRPO 三种 checkpoint 类型。

用法：
  # SFT: base + SFT adapter
  python3 eval_grpo.py --mode sft \
    --base_model ~/models/gemma-2-2b-it \
    --checkpoint_dir checkpoints/sft-lr5e6 \
    --output_dir eval_results/sft-lr5e6

  # DPO: base + SFT merge + DPO adapter
  python3 eval_grpo.py --mode dpo \
    --base_model ~/models/gemma-2-2b-it \
    --sft_adapter checkpoints/gemma2-2b-it-sft-lr1e5-r8-fa2/checkpoint-50 \
    --checkpoint_dir checkpoints/gemma2-2b-it-dpo-ablation/dpo-beta01 \
    --output_dir eval_results/dpo-beta01

  # GRPO: base + SFT merge + GRPO adapter (default)
  python3 eval_grpo.py --mode grpo \
    --base_model ~/models/gemma-2-2b-it \
    --sft_adapter checkpoints/gemma2-2b-it-sft-lr1e5-r8-fa2/checkpoint-50 \
    --checkpoint_dir checkpoints/ablation-beta001 \
    --output_dir eval_results/ablation-beta001
"""
import argparse, json, os, re, shutil, gc
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Please solve the following math problem step by step "
    "and provide the final answer at the end preceded by ####."
)


def extract_answer_strict(text):
    """严格提取: #### 后只跟纯数字到行尾"""
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)\s*$", text, re.MULTILINE)
    if m:
        return m.group(1).replace(",", ""), True
    return None, False


def extract_answer_fallback(text):
    """宽松提取: 最后一个数字"""
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def get_gold(answer_text):
    m = re.search(r'####\s*(-?[\d,]*\d[\d,]*\.?\d*)', answer_text)
    return m.group(1).replace(',', '') if m else None


def has_hash_marker(text):
    return "####" in text


def merge_model(base_path, adapter_path, output_path, sft_path=None, mode="grpo"):
    """Merge adapters based on mode:
    - sft:  base + adapter
    - dpo:  base + SFT merge + adapter
    - grpo: base + SFT merge + adapter
    """
    print(f"[Merge] mode={mode}, adapter={Path(adapter_path).name} → {output_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    if mode in ("dpo", "grpo") and sft_path:
        model = PeftModel.from_pretrained(model, sft_path)
        model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    del model
    gc.collect()
    print(f"[Merge] Done")


def evaluate(merged_path, test_data, gpu_mem=0.85):
    from vllm import LLM, SamplingParams
    print(f"\n[Eval] {merged_path} ({len(test_data)} samples)")

    llm = LLM(
        model=str(merged_path),
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    # Gemma2: no system role, fold into user message
    prompts = []
    for item in test_data:
        messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + item["question"]}]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))

    stop_tokens = ["<end_of_turn>"]
    eos_ids = [tokenizer.convert_tokens_to_ids(t) for t in stop_tokens
               if tokenizer.convert_tokens_to_ids(t) != tokenizer.unk_token_id]
    sampling = SamplingParams(temperature=0, max_tokens=512,
                              stop=stop_tokens, stop_token_ids=eos_ids)
    outputs = llm.generate(prompts, sampling)

    results = []
    for item, out in zip(test_data, outputs):
        resp = out.outputs[0].text
        gold = get_gold(item["answer"])

        has_marker = has_hash_marker(resp)
        strict_pred, is_strict = extract_answer_strict(resp)
        fallback_pred = extract_answer_fallback(resp)

        # 数字正确（fallback）
        try:
            num_correct = fallback_pred is not None and gold is not None and \
                          abs(float(fallback_pred) - float(gold)) < 1e-4
        except (ValueError, TypeError):
            num_correct = False

        # 严格正确（#### + 纯数字 + 数值对）
        try:
            strict_correct = is_strict and strict_pred is not None and gold is not None and \
                             abs(float(strict_pred) - float(gold)) < 1e-4
        except (ValueError, TypeError):
            strict_correct = False

        results.append({
            "question": item["question"],
            "gold": gold,
            "has_marker": has_marker,
            "strict_pred": strict_pred,
            "fallback_pred": fallback_pred,
            "num_correct": num_correct,
            "strict_correct": strict_correct,
            "response": resp,
        })

    del llm
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    return results


def print_metrics(name, results):
    n = len(results)
    marker = sum(1 for r in results if r["has_marker"])
    num_ok = sum(1 for r in results if r["num_correct"])
    strict = sum(1 for r in results if r["strict_correct"])
    print(f"\n{'='*50}")
    print(f" {name} ({n} samples)")
    print(f"{'='*50}")
    print(f" ####率:     {marker:>5}/{n}  = {marker/n:.2%}")
    print(f" 数字正确:   {num_ok:>5}/{n}  = {num_ok/n:.2%}")
    print(f" 严格正确:   {strict:>5}/{n}  = {strict/n:.2%}")
    print(f"{'='*50}\n")
    return {"name": name, "hash_rate": marker/n, "num_acc": num_ok/n, "strict_acc": strict/n}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--sft_adapter", default=None, help="SFT adapter path (required for dpo/grpo mode)")
    parser.add_argument("--checkpoint_dir", required=True, help="包含 checkpoint-* 子目录")
    parser.add_argument("--mode", choices=["sft", "dpo", "grpo"], default="grpo")
    parser.add_argument("--output_dir", default="~/eval_results")
    parser.add_argument("--gpu_mem", type=float, default=0.85)
    parser.add_argument("--steps", type=int, nargs="+", default=None, help="只评指定步数")
    parser.add_argument("--num_samples", type=int, default=0, help="0=全量")
    parser.add_argument("--test_data", default=None, help="test.jsonl 路径")
    args = parser.parse_args()

    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_tmp_merged"

    # Load test data
    test_path = args.test_data or os.path.expanduser("~/data/test.jsonl")
    test_data = [json.loads(l) for l in open(test_path)]
    if args.num_samples > 0:
        test_data = test_data[:args.num_samples]
    print(f"Test data: {len(test_data)} samples")

    # Find checkpoints
    ckpt_dir = Path(os.path.expanduser(args.checkpoint_dir))
    checkpoints = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if args.steps:
        allowed = {f"checkpoint-{s}" for s in args.steps}
        checkpoints = [p for p in checkpoints if p.name in allowed]
    print(f"Checkpoints to evaluate: {[p.name for p in checkpoints]}")

    summary = []
    for ckpt in checkpoints:
        out_jsonl = output_dir / f"{ckpt.name}.jsonl"

        # Skip if already done
        if out_jsonl.exists():
            results = [json.loads(l) for l in open(out_jsonl)]
            metrics = print_metrics(ckpt.name, results)
            summary.append(metrics)
            continue

        # Merge
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        sft_path = os.path.expanduser(args.sft_adapter) if args.sft_adapter else None
        merge_model(
            os.path.expanduser(args.base_model),
            ckpt,
            tmp_dir,
            sft_path=sft_path,
            mode=args.mode,
        )

        # Evaluate
        results = evaluate(tmp_dir, test_data, args.gpu_mem)
        metrics = print_metrics(ckpt.name, results)
        summary.append(metrics)

        # Save
        with open(out_jsonl, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        shutil.rmtree(tmp_dir)

    # Final summary
    print("\n" + "=" * 60)
    print(f"{'Checkpoint':<20} {'####率':>8} {'数字正确':>10} {'严格正确':>10}")
    print("=" * 60)
    for s in summary:
        print(f"{s['name']:<20} {s['hash_rate']:>8.2%} {s['num_acc']:>10.2%} {s['strict_acc']:>10.2%}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_dir}/summary.json")

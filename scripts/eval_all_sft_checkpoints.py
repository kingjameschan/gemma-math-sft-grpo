"""
自动评测所有 SFT checkpoint。
流程：merge adapter → vLLM 评测 1319 题 → 记录准确率 → 删除临时 merged 模型

用法（WSL2 vllm-env）：
  ~/vllm-env/bin/python scripts/eval_all_sft_checkpoints.py \
    --base_model models/Qwen3-1.7B-Base \
    --checkpoint_dirs checkpoints/qwen3-1.7b-sft-lr3e4-r16 \
                      checkpoints/qwen3-1.7b-sft-lr2e4-r16 \
                      checkpoints/qwen3-1.7b-sft-lr1e4-r16 \
    --output_dir outputs/sft_ckpt_evals \
    --quantization bf16   # 1.7B 用 bf16；7B/8B 用 fp8
"""
import argparse, json, os, re, shutil
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Please solve the following math problem step by step "
    "and provide the final answer at the end preceded by ####."
)

QWEN3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{%- if message.role == 'system' %}{{- '<|im_start|>system\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'user' %}{{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'assistant' %}{{- '<|im_start|>assistant\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)

GEMMA2_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{%- if message.role == 'system' %}{# skip #}"
    "{%- elif message.role == 'user' %}{{- '<start_of_turn>user\\n' + message.content + '<end_of_turn>\\n' }}"
    "{%- elif message.role == 'assistant' %}{{- '<start_of_turn>model\\n' + message.content + '<end_of_turn>\\n' }}"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<start_of_turn>model\\n' }}{%- endif %}"
)

def extract_answer(text):
    m = re.search(r'####\s*(-?[\d,]*\d[\d,]*\.?\d*)', text)
    if m:
        return float(m.group(1).replace(',', ''))
    nums = re.findall(r'-?\d[\d,]*\.?\d*', text)
    return float(nums[-1].replace(',', '')) if nums else None

def get_gold(answer_text):
    m = re.search(r'####\s*(-?[\d,]*\d[\d,]*\.?\d*)', answer_text)
    return float(m.group(1).replace(',', '')) if m else None

def resolve(p, project_root):
    p = Path(p)
    return p if p.is_absolute() else project_root / p

def merge_adapter(base_model_path, adapter_path, output_path, sft_adapter_path=None):
    """Merge adapter into base model. If sft_adapter_path is given, do two-step merge (base→SFT→DPO)."""
    if sft_adapter_path:
        print(f"\n[Merge] Two-step: base + SFT({Path(sft_adapter_path).name}) + DPO({adapter_path.name}) → {output_path}")
    else:
        print(f"\n[Merge] {adapter_path.name} → {output_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True,
    )
    if sft_adapter_path:
        # Step 1: merge SFT adapter
        model = PeftModel.from_pretrained(model, sft_adapter_path)
        model = model.merge_and_unload()
        print(f"[Merge] SFT merged, now applying DPO adapter...")
    # Merge the target adapter (SFT or DPO)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    del model
    print(f"[Merge] Done → {output_path}")

def evaluate(merged_path, out_jsonl, test_data, quantization, gpu_mem, max_model_len=None):
    from vllm import LLM, SamplingParams
    print(f"\n[Eval] {merged_path.name} ({len(test_data)} samples)")
    quant = None if quantization == "bf16" else quantization
    llm_kwargs = dict(
        model=str(merged_path),
        dtype="bfloat16",
        quantization=quant,
        gpu_memory_utilization=gpu_mem,
        trust_remote_code=True,
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    cfg = AutoConfig.from_pretrained(str(merged_path), trust_remote_code=True)
    model_type = getattr(cfg, "model_type", "")
    if model_type == "qwen3":
        tokenizer.chat_template = QWEN3_CHAT_TEMPLATE
    elif model_type == "gemma2":
        tokenizer.chat_template = GEMMA2_CHAT_TEMPLATE

    stop_tokens = ["<end_of_turn>"] if model_type == "gemma2" else ["<|im_end|>"]

    prompts = []
    for item in test_data:
        if model_type == "gemma2":
            # Gemma2 不支持 system role，折入 user message
            messages = [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + item["question"]},
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": item["question"]},
            ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))

    eos_ids = [tokenizer.convert_tokens_to_ids(t) for t in stop_tokens
               if tokenizer.convert_tokens_to_ids(t) != tokenizer.unk_token_id]
    sampling = SamplingParams(temperature=0, max_tokens=512,
                              stop=stop_tokens, stop_token_ids=eos_ids)
    outputs = llm.generate(prompts, sampling)

    results, correct = [], 0
    for item, out in zip(test_data, outputs):
        resp = out.outputs[0].text
        pred = extract_answer(resp)
        gold = get_gold(item["answer"])
        ok   = (pred is not None and gold is not None and abs(pred - gold) < 1e-6)
        if ok:
            correct += 1
        results.append({
            "question": item["question"],
            "gold_val": gold, "pred_val": pred,
            "is_correct": ok, "full_response": resp,
        })

    acc = correct / len(test_data)
    with open(out_jsonl, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[Eval] Accuracy: {acc:.2%} ({correct}/{len(test_data)})")
    del llm
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",       type=str, required=True)
    parser.add_argument("--checkpoint_dirs",  type=str, nargs="+", required=True)
    parser.add_argument("--output_dir",       type=str, default="outputs/sft_ckpt_evals")
    parser.add_argument("--quantization",     type=str, default="bf16", choices=["bf16", "fp8"])
    parser.add_argument("--num_samples",      type=int, default=0, help="0=全量 1319")
    parser.add_argument("--gpu_mem",          type=float, default=0.88)
    parser.add_argument("--max_model_len",    type=int,   default=None, help="限制 vLLM 最大序列长度，8B fp8 建议 4096")
    parser.add_argument("--steps",            type=int, nargs="+", default=None, help="只评指定步数，如 --steps 700 900 1100")
    parser.add_argument("--sft_adapter",     type=str, default=None, help="SFT adapter 路径（DPO 评测时需要两步 merge：base→SFT→DPO）")
    args = parser.parse_args()

    base_model_path  = resolve(args.base_model, project_root)
    sft_adapter_path = resolve(args.sft_adapter, project_root) if args.sft_adapter else None
    output_dir       = resolve(args.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_merge_dir   = project_root / "models" / "_tmp_merged"

    # --- 数据 ---
    test_path = project_root / "data" / "gsm8k" / "test.jsonl"
    test_data = [json.loads(l) for l in open(test_path)]
    if args.num_samples > 0:
        test_data = test_data[:args.num_samples]

    # --- 收集所有 checkpoint ---
    checkpoints = []
    for ckpt_dir in args.checkpoint_dirs:
        ckpt_dir = resolve(ckpt_dir, project_root)
        run_name  = ckpt_dir.name
        for ckpt in sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])):
            checkpoints.append((run_name, ckpt))

    if args.steps:
        allowed = {f"checkpoint-{s}" for s in args.steps}
        checkpoints = [(r, p) for r, p in checkpoints if p.name in allowed]

    print(f"\n共 {len(checkpoints)} 个 checkpoint 待评测")

    summary = []
    for run_name, ckpt_path in checkpoints:
        step = ckpt_path.name  # e.g. checkpoint-100
        out_jsonl = output_dir / f"{run_name}_{step}.jsonl"

        if out_jsonl.exists():
            # 断点续评
            results = [json.loads(l) for l in open(out_jsonl)]
            acc = sum(r["is_correct"] for r in results) / len(results)
            print(f"[Skip] {run_name}/{step} — already done, acc={acc:.2%}")
            summary.append({"run": run_name, "step": step, "accuracy": acc})
            continue

        # merge
        if tmp_merge_dir.exists():
            shutil.rmtree(tmp_merge_dir)
        merge_adapter(base_model_path, ckpt_path, tmp_merge_dir, sft_adapter_path)

        # evaluate
        acc = evaluate(tmp_merge_dir, out_jsonl, test_data, args.quantization, args.gpu_mem, args.max_model_len)
        summary.append({"run": run_name, "step": step, "accuracy": acc})

        # cleanup
        shutil.rmtree(tmp_merge_dir)

    # --- 汇总 ---
    print("\n" + "="*60)
    print(f"{'Run':<35} {'Step':<18} {'Accuracy':>8}")
    print("="*60)
    for r in summary:
        print(f"{r['run']:<35} {r['step']:<18} {r['accuracy']:>8.2%}")

    # 保存汇总
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存至 {summary_path}")

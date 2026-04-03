"""
用 vLLM 批量评测模型（base / SFT-merged / DPO-merged）。

适用于：
  - Baseline 评测（base 模型，8-shot CoT）
  - SFT / DPO 评测（merged 模型，zero-shot）

输出格式：{id, question, gold_val, pred_val, is_correct, full_response}
可直接接 error_analysis.py / analyze_results.py。

用法：
  # Baseline（8-shot，base 模型）
  ~/vllm-env/bin/python scripts/evaluate_vllm.py \
    --model models/Qwen3-1.7B-Base \
    --output outputs/baseline_qwen3_1.7b.jsonl \
    --fewshot

  # SFT / DPO（zero-shot，merged 模型）
  ~/vllm-env/bin/python scripts/evaluate_vllm.py \
    --model models/Qwen2.5-7B-SFT-merged \
    --output outputs/eval_qwen2.5_7b_sft.jsonl

  # 调试（只跑前 50 题）
  ~/vllm-env/bin/python scripts/evaluate_vllm.py \
    --model models/Qwen3-1.7B-Base \
    --output outputs/debug.jsonl \
    --num_samples 50 --fewshot
"""
import argparse
import json
import os
import re
from tqdm import tqdm

# Gemma2 base 模型 chat template（与 gemma-2-*-it 格式一致）
GEMMA2_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{%- if message.role == 'user' %}"
    "{{- '<start_of_turn>user\\n' + message.content + '<end_of_turn>\\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{{- '<start_of_turn>model\\n' + message.content + '<end_of_turn>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<start_of_turn>model\\n' }}"
    "{%- endif %}"
)

# Qwen3 简化 ChatML 模板（禁用 thinking 块）
QWEN3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{%- if message.role == 'system' %}"
    "{{- '<|im_start|>system\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'user' %}"
    "{{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Please solve the following math problem step by step "
    "and provide the final answer at the end preceded by ####."
)

# 8-shot 示例固定索引（覆盖不同难度，seed=42 可复现）
FEW_SHOT_INDICES = [0, 1, 2, 7, 15, 30, 60, 120]


def extract_answer(text):
    match = re.search(r'####\s*(-?\d[\d,]*\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    numbers = re.findall(r'-?\d[\d,]*\.?\d*', text)
    return numbers[-1].replace(',', '') if numbers else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           type=str, required=True)
    parser.add_argument("--output",          type=str, required=True)
    parser.add_argument("--data",            type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("--train_data",      type=str, default="data/gsm8k/train.jsonl")
    parser.add_argument("--num_samples",     type=int, default=0,    help="0 = 全量")
    parser.add_argument("--max_tokens",      type=int, default=512)
    parser.add_argument("--batch_size",      type=int, default=50)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--gpu_mem",         type=float, default=0.85, help="gpu_memory_utilization")
    parser.add_argument("--quantization",    type=str,   default=None, help="量化方式：fp8 / bitsandbytes（7B/8B 节省显存用）")
    parser.add_argument("--fewshot",         action="store_true",   help="8-shot CoT（baseline 用）")
    args = parser.parse_args()

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(project_root, p)

    model_path  = resolve(args.model)
    output_path = resolve(args.output)
    data_path   = resolve(args.data)
    train_path  = resolve(args.train_data)

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    with open(data_path, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    if args.num_samples > 0:
        test_data = test_data[:args.num_samples]

    few_shot_examples = []
    if args.fewshot:
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = [json.loads(line) for line in f]
        few_shot_examples = [train_data[i] for i in FEW_SHOT_INDICES]
        print(f"8-shot 模式：使用 {len(few_shot_examples)} 道示例")

    print(f"模型:   {model_path}")
    print(f"输出:   {output_path}")
    print(f"题目数: {len(test_data)}")

    # ── 断点续跑 ──────────────────────────────────────────────────────────────
    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)['id'])
                except Exception:
                    pass
        if done_ids:
            print(f"已有 {len(done_ids)} 条，续跑...")

    pending = [(idx, item) for idx, item in enumerate(test_data) if idx not in done_ids]

    if not pending:
        print("所有样本已完成。")
    else:
        print(f"待推理: {len(pending)} 条")

        # ── 检测模型类型 ──────────────────────────────────────────────────────
        from vllm import LLM, SamplingParams
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = getattr(cfg, "model_type", "")

        # ── 初始化 vLLM ──────────────────────────────────────────────────────
        q_str = f" + {args.quantization}" if args.quantization else " bf16"
        print(f"\n初始化 vLLM ({q_str.strip()})...")
        llm_kwargs = dict(
            model=model_path,
            dtype="bfloat16",
            tensor_parallel_size=args.tensor_parallel,
            gpu_memory_utilization=args.gpu_mem,
            trust_remote_code=True,
            max_model_len=2048,
        )
        if args.quantization:
            llm_kwargs["quantization"] = args.quantization
            if args.quantization == "bitsandbytes":
                llm_kwargs["load_format"] = "bitsandbytes"
        llm = LLM(**llm_kwargs)

        # ── 设置 tokenizer 模板 + stop token ─────────────────────────────────
        tokenizer = llm.get_tokenizer()
        if model_type == "qwen3":
            tokenizer.chat_template = QWEN3_CHAT_TEMPLATE
            print("Qwen3 检测到，使用简化 chat template（thinking 已禁用）")
        elif model_type == "gemma2":
            tokenizer.chat_template = GEMMA2_CHAT_TEMPLATE
            print("Gemma2 检测到，使用 Gemma2 chat template")

        if model_type == "gemma2":
            stop_tokens = ["<end_of_turn>"]
        else:
            stop_tokens = ["<|im_end|>"]
        sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens, stop=stop_tokens)

        # ── 构建 prompts ───────────────────────────────────────────────────────
        prompts = []
        for _, item in pending:
            if model_type == "gemma2":
                # Gemma2 不支持 system role，将 system prompt 折入第一个 user message
                messages = []
                if few_shot_examples:
                    first_q = SYSTEM_PROMPT + "\n\n" + few_shot_examples[0]["question"]
                    messages.append({"role": "user", "content": first_q})
                    messages.append({"role": "assistant", "content": few_shot_examples[0]["answer"]})
                    for ex in few_shot_examples[1:]:
                        messages.append({"role": "user", "content": ex["question"]})
                        messages.append({"role": "assistant", "content": ex["answer"]})
                    messages.append({"role": "user", "content": item["question"]})
                else:
                    messages.append({"role": "user", "content": SYSTEM_PROMPT + "\n\n" + item["question"]})
            else:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for ex in few_shot_examples:
                    messages.append({"role": "user",      "content": ex["question"]})
                    messages.append({"role": "assistant", "content": ex["answer"]})
                messages.append({"role": "user", "content": item["question"]})
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))

        # ── 批量推理 ──────────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        n_batches = (len(pending) - 1) // args.batch_size + 1

        with open(output_path, 'a', encoding='utf-8') as f:
            for b in tqdm(range(n_batches), desc="Evaluating", unit="batch"):
                batch         = pending[b * args.batch_size:(b + 1) * args.batch_size]
                batch_prompts = prompts[b * args.batch_size:(b + 1) * args.batch_size]
                outputs       = llm.generate(batch_prompts, sampling_params)

                for (idx, item), output in zip(batch, outputs):
                    response   = output.outputs[0].text
                    gold_val   = extract_answer(item['answer'])
                    pred_val   = extract_answer(response)
                    is_correct = bool(pred_val and gold_val and float(pred_val) == float(gold_val))
                    f.write(json.dumps({
                        "id":            idx,
                        "question":      item['question'],
                        "gold_val":      gold_val,
                        "pred_val":      pred_val,
                        "is_correct":    is_correct,
                        "full_response": response,
                    }, ensure_ascii=False) + '\n')

    # ── 统计 ──────────────────────────────────────────────────────────────────
    all_records = []
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                all_records.append(json.loads(line))
            except Exception:
                pass

    correct = sum(1 for r in all_records if r['is_correct'])
    total   = len(all_records)
    print(f"\n{'='*40}")
    print(f"模型:   {os.path.basename(model_path)}")
    print(f"准确率: {correct/total:.2%}  ({correct}/{total})")
    print(f"结果:   {output_path}")


if __name__ == "__main__":
    main()

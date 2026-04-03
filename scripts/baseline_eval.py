"""
8-shot CoT baseline 评测脚本。用于评测裸基座模型在 GSM8K 上的准确率。

SFT/DPO 评测用 zero-shot；baseline 用 8-shot（学术标准做法，充分激发基座推理能力）。

用法：
  python scripts/baseline_eval.py \
    --model models/Qwen3-1.7B-Base \
    --output outputs/baseline_qwen3_1.7b.jsonl

  python scripts/baseline_eval.py \
    --model models/Qwen2.5-7B \
    --output outputs/baseline_qwen2.5_7b.jsonl
"""
import os
import re
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig

# Qwen3 简化 ChatML 模板（禁用 thinking 块），与 train_sft_v2.py 保持一致
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

# 8-shot 示例选取逻辑：
# 从训练集按推理步数（answer 中换行数）均匀采样，覆盖 1-step 到 5+ step 复杂题。
# 固定 seed=42，保证可复现。训练/测试集本身不重叠，无需额外去重。
FEW_SHOT_INDICES = [0, 1, 2, 7, 15, 30, 60, 120]  # 固定索引，覆盖不同难度


def extract_answer(text):
    match = re.search(r'####\s*(-?\d[\d,]*\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    numbers = re.findall(r'-?\d[\d,]*\.?\d*', text)
    return numbers[-1].replace(',', '') if numbers else None


def build_fewshot_prompt(tokenizer, few_shot_examples, question):
    """构造 8-shot prompt：system + 8 个示例（user/assistant 对）+ 当前问题"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in few_shot_examples:
        messages.append({"role": "user",      "content": ex["question"]})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, required=True, help="基座模型路径（相对项目根或绝对路径）")
    parser.add_argument("--output",      type=str, required=True, help="输出 jsonl 路径（相对项目根或绝对路径）")
    parser.add_argument("--data",        type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("--train_data",  type=str, default="data/gsm8k/train.jsonl")
    parser.add_argument("--num_samples", type=int, default=0, help="0 = 全量 1319 题")
    args = parser.parse_args()

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(project_root, p)

    model_path  = resolve(args.model)
    output_path = resolve(args.output)
    data_path   = resolve(args.data)
    train_path  = resolve(args.train_data)

    # --- 加载训练集，取 few-shot 示例 ---
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = [json.loads(line) for line in f]
    few_shot_examples = [train_data[i] for i in FEW_SHOT_INDICES]
    print(f"Few-shot 示例: {len(few_shot_examples)} 道（indices {FEW_SHOT_INDICES}）")

    # --- 加载测试集 ---
    with open(data_path, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    if args.num_samples > 0:
        test_data = test_data[:args.num_samples]
    print(f"模型:  {model_path}")
    print(f"输出:  {output_path}")
    print(f"题目数: {len(test_data)}")

    # --- 断点续跑 ---
    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)['id'])
                except Exception:
                    pass
        if done_ids:
            print(f"已有 {len(done_ids)} 条结果，跳过继续...")

    pending = [(idx, item) for idx, item in enumerate(test_data) if idx not in done_ids]
    if not pending:
        print("所有样本已完成。")
    else:
        print(f"待推理: {len(pending)} 条")

        # --- 加载分词器，处理 Qwen3 chat template ---
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")
        if model_type == "qwen3":
            tokenizer.chat_template = QWEN3_CHAT_TEMPLATE
            print(f"Qwen3 检测到，使用简化 chat template（thinking 已禁用）")

        # --- 加载 4-bit NF4 模型 ---
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("加载 4-bit 模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        # --- 逐条推理 ---
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'a', encoding='utf-8') as f_out:
            for i, (idx, item) in enumerate(tqdm(pending, desc="Baseline eval", unit="q")):

                prompt = build_fewshot_prompt(tokenizer, few_shot_examples, item['question'])
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.01,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                response = tokenizer.decode(
                    output_ids[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                gold_val   = extract_answer(item['answer'])
                pred_val   = extract_answer(response)
                is_correct = bool(pred_val and gold_val and float(pred_val) == float(gold_val))

                record = {
                    "id":            idx,
                    "question":      item['question'],
                    "gold_val":      gold_val,
                    "pred_val":      pred_val,
                    "is_correct":    is_correct,
                    "full_response": response,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

    # --- 统计 ---
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

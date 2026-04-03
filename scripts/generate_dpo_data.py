"""
用 SFT 最优模型在训练集上多次采样，生成 DPO 偏好对。

流程：
  1. 合并 SFT adapter → 临时 merged 模型
  2. vLLM 对每道训练题采样 N 次（temperature>0）
  3. 有对有错 → 取 1 chosen + 1 rejected
  4. 全对/全错 → 跳过
  5. 保存为 {prompt, chosen, rejected} 格式

用法：
  # Gemma2-2B-IT
  ~/vllm-env/bin/python scripts/generate_dpo_data.py \
    --base_model models/gemma-2-2b-it \
    --sft_adapter checkpoints/gemma2-2b-it-sft-lr1e5-r8-fa2/checkpoint-50 \
    --output data/dpo/dpo_gemma2_2b.jsonl

  # Qwen3-1.7B
  ~/vllm-env/bin/python scripts/generate_dpo_data.py \
    --base_model models/Qwen3-1.7B-Base \
    --sft_adapter checkpoints/qwen3-1.7b-sft-lr3e4-r8/checkpoint-200 \
    --output data/dpo/dpo_qwen3_1.7b.jsonl
"""
import argparse, json, re, shutil, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Please solve the following math problem step by step "
    "and provide the final answer at the end preceded by ####."
)

# Qwen3 base model 需要手动指定 chat template
QWEN3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{%- if message.role == 'system' %}{{- '<|im_start|>system\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'user' %}{{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'assistant' %}{{- '<|im_start|>assistant\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)

# 各模型的 stop token
STOP_TOKENS = {
    "gemma2": ["<end_of_turn>"],
    "qwen3":  ["<|im_end|>"],
    "qwen2":  ["<|im_end|>"],
}


def extract_answer(text):
    m = re.search(r'####\s*(-?[\d,]*\d[\d,]*\.?\d*)', text)
    if m:
        return float(m.group(1).replace(',', ''))
    nums = re.findall(r'-?\d[\d,]*\.?\d*', text)
    return float(nums[-1].replace(',', '')) if nums else None


def get_gold(answer_text):
    m = re.search(r'####\s*(-?[\d,]*\d[\d,]*\.?\d*)', answer_text)
    return float(m.group(1).replace(',', '')) if m else None


def merge_adapter(base_model_path, adapter_path, output_path):
    print(f"[Merge] {adapter_path} -> {output_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    del model
    torch.cuda.empty_cache()
    print(f"[Merge] Done")


def build_messages(question, model_type):
    """构建 chat messages，根据模型类型处理 system prompt。"""
    if model_type == "gemma2":
        # Gemma2 不支持 system role，折叠进 user
        return [{"role": "user", "content": SYSTEM_PROMPT + "\n\nProblem: " + question}]
    else:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",   type=str, required=True)
    parser.add_argument("--sft_adapter",  type=str, required=True)
    parser.add_argument("--output",       type=str, required=True)
    parser.add_argument("--n_samples",    type=int, default=5)
    parser.add_argument("--temperature",  type=float, default=0.7)
    parser.add_argument("--max_tokens",   type=int, default=384)
    parser.add_argument("--gpu_mem",      type=float, default=0.88)
    args = parser.parse_args()

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else project_root / p

    base_model_path  = resolve(args.base_model)
    sft_adapter_path = resolve(args.sft_adapter)
    output_path      = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_merge_dir    = project_root / "models" / "_tmp_merged_dpo"

    # 检测模型类型
    cfg = AutoConfig.from_pretrained(str(base_model_path), trust_remote_code=True)
    model_type = getattr(cfg, "model_type", "")
    print(f"模型类型: {model_type}")

    # 加载训练集（原始 GSM8K）
    raw_data = [json.loads(l) for l in open(project_root / "data/gsm8k/train.jsonl")]
    print(f"训练题数: {len(raw_data)}")

    # Merge SFT adapter
    if tmp_merge_dir.exists():
        shutil.rmtree(tmp_merge_dir)
    merge_adapter(base_model_path, sft_adapter_path, tmp_merge_dir)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(tmp_merge_dir),
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    if model_type == "qwen3":
        tokenizer.chat_template = QWEN3_CHAT_TEMPLATE

    # 构建 prompts
    prompts = []
    gold_vals = []
    raw_questions = []
    for item in raw_data:
        question = item["question"]
        gold = get_gold(item["answer"])
        raw_questions.append(question)
        gold_vals.append(gold)
        messages = build_messages(question, model_type)
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))

    stop_tokens = STOP_TOKENS.get(model_type, ["<|im_end|>"])
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.n_samples,
        stop=stop_tokens,
    )

    print(f"开始采样（{len(prompts)} 题 x {args.n_samples} 次，stop={stop_tokens}）...")
    outputs = llm.generate(prompts, sampling)

    # 生成 DPO 对
    pairs = []
    skipped_all_correct = 0
    skipped_all_wrong = 0

    # 取第一个 prompt 作为 prompt 模板示例
    prompt_example = prompts[0] if prompts else ""

    for i, (out, gold, question) in enumerate(zip(outputs, gold_vals, raw_questions)):
        responses = [o.text for o in out.outputs]
        correct = []
        wrong = []
        for r in responses:
            pred = extract_answer(r)
            ok = (pred is not None and gold is not None and abs(pred - gold) < 1e-6)
            if ok:
                correct.append(r)
            else:
                wrong.append(r)

        if not correct:
            skipped_all_wrong += 1
            continue
        if not wrong:
            skipped_all_correct += 1
            continue

        # prompt = apply_chat_template 的完整结果（包含 generation prompt）
        prompt_str = prompts[i]

        pairs.append({
            "prompt":   prompt_str,
            "chosen":   correct[0],
            "rejected": wrong[0],
        })

        if (i + 1) % 1000 == 0:
            print(f"  进度: {i+1}/{len(prompts)}, 已生成 {len(pairs)} 对")

    print(f"\n完成！")
    print(f"  生成对数: {len(pairs)}")
    print(f"  全对跳过: {skipped_all_correct}")
    print(f"  全错跳过: {skipped_all_wrong}")
    print(f"  yield 率: {len(pairs)/len(prompts):.1%}")

    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"已保存至 {output_path}")

    del llm
    shutil.rmtree(tmp_merge_dir, ignore_errors=True)
    torch.cuda.empty_cache()

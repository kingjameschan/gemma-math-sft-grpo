"""
用 base 模型跑指定错题集，zero-shot 和 8-shot 各一组对比。
用法：
  ~/vllm-env/bin/python scripts/eval_base_on_errors.py \
    --base_model models/Qwen3-1.7B-Base \
    --errors_jsonl outputs/sft_ckpt_evals/errors_common.jsonl \
    --output_dir outputs/sft_ckpt_evals \
    --num_samples 20
"""
import argparse, json, re, torch
from pathlib import Path
from transformers import AutoConfig

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

def extract_answer(text):
    m = re.search(r'####\s*(-?[\d,]*\d[\d,]*\.?\d*)', text)
    if m:
        return float(m.group(1).replace(',', ''))
    nums = re.findall(r'-?\d[\d,]*\.?\d*', text)
    return float(nums[-1].replace(',', '')) if nums else None

def build_8shot_prefix(train_path, exclude_questions, seed=42):
    import random
    random.seed(seed)
    train = [json.loads(l) for l in open(train_path)]
    exclude = set(exclude_questions)
    train = [x for x in train if x['question'] not in exclude]
    train_sorted = sorted(train, key=lambda x: len(x['answer']))
    step = len(train_sorted) // 8
    shots = [train_sorted[i * step] for i in range(8)]
    prefix = ""
    for ex in shots:
        prefix += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
    return prefix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",   type=str, required=True)
    parser.add_argument("--errors_jsonl", type=str, required=True)
    parser.add_argument("--output_dir",   type=str, default="outputs/sft_ckpt_evals")
    parser.add_argument("--num_samples",  type=int, default=20)
    parser.add_argument("--gpu_mem",      type=float, default=0.88)
    args = parser.parse_args()

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else project_root / p

    model_path = resolve(args.base_model)
    errors_path = resolve(args.errors_jsonl)
    output_dir  = resolve(args.output_dir)

    items = [json.loads(l) for l in open(errors_path)][:args.num_samples]
    print(f"取前 {len(items)} 道错题，分别做 zero-shot 和 8-shot 评测")

    train_path   = project_root / "data" / "gsm8k" / "train.jsonl"
    shot8_prefix = build_8shot_prefix(train_path, [x['question'] for x in items])

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    cfg = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    if getattr(cfg, "model_type", "") == "qwen3":
        tokenizer.chat_template = QWEN3_CHAT_TEMPLATE

    sampling = SamplingParams(temperature=0, max_tokens=512, stop=["<|im_end|>"])

    # zero-shot prompts
    prompts_0 = []
    for item in items:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": item["question"]},
        ]
        prompts_0.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    # 8-shot: 把示例放进 system prompt
    system_8shot = SYSTEM_PROMPT + "\n\nHere are some examples:\n\n" + shot8_prefix.strip()
    prompts_8 = []
    for item in items:
        msgs = [
            {"role": "system", "content": system_8shot},
            {"role": "user",   "content": item["question"]},
        ]
        prompts_8.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    out0 = llm.generate(prompts_0, sampling)
    out8 = llm.generate(prompts_8, sampling)

    results = []
    correct_0 = correct_8 = 0
    for item, o0, o8 in zip(items, out0, out8):
        gold = item['gold_val']
        r0   = o0.outputs[0].text
        r8   = o8.outputs[0].text
        p0   = extract_answer(r0)
        p8   = extract_answer(r8)
        ok0  = (p0 is not None and gold is not None and abs(p0 - gold) < 1e-6)
        ok8  = (p8 is not None and gold is not None and abs(p8 - gold) < 1e-6)
        if ok0: correct_0 += 1
        if ok8: correct_8 += 1
        results.append({
            "question":       item["question"],
            "gold_val":       gold,
            "zero_pred":      p0,
            "zero_correct":   ok0,
            "zero_response":  r0,
            "shot8_pred":     p8,
            "shot8_correct":  ok8,
            "shot8_response": r8,
        })

    n = len(items)
    print(f"\n{'='*50}")
    print(f"Base zero-shot 准确率: {correct_0/n:.2%} ({correct_0}/{n})")
    print(f"Base  8-shot  准确率: {correct_8/n:.2%} ({correct_8}/{n})")
    print(f"SFT best zero-shot  :  0.00% (0/{n})  ← 这些都是SFT错题")
    print(f"{'='*50}")

    out_path = output_dir / f"base_vs_sft_errors_{n}samples.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n详细结果已保存至 {out_path}")

    print("\n── 逐题对比 ──")
    for i, r in enumerate(results, 1):
        z = "✓" if r['zero_correct'] else "✗"
        s = "✓" if r['shot8_correct'] else "✗"
        print(f"{i:2}. gold={r['gold_val']}  0shot={z}({r['zero_pred']})  8shot={s}({r['shot8_pred']})")
        print(f"    Q: {r['question'][:75]}...")

    del llm
    torch.cuda.empty_cache()

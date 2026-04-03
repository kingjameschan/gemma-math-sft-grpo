"""
交互式测试 base 模型 zero-shot 推理，观察 EOS token 行为。

用法（WSL2）：
  ~/vllm-env/bin/python scripts/interactive_base.py
  ~/vllm-env/bin/python scripts/interactive_base.py --model models/Qwen3-1.7B-Base
"""
import argparse
from pathlib import Path

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoConfig

script_dir = Path(__file__).resolve().parent
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     type=str, default="models/Qwen3-1.7B-Base")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--gpu_mem",   type=float, default=0.85)
    args = parser.parse_args()

    model_path = args.model if Path(args.model).is_absolute() else project_root / args.model

    print(f"Loading {model_path} ...")
    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        trust_remote_code=True,
        max_model_len=4096,
    )
    tokenizer = llm.get_tokenizer()
    cfg = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    if getattr(cfg, "model_type", "") == "qwen3":
        tokenizer.chat_template = QWEN3_CHAT_TEMPLATE

    print(f"\nEOS token: {repr(tokenizer.eos_token)}  (id={tokenizer.eos_token_id})")
    print("模型加载完成。输入数学题，空行提交，输入 q 退出。\n")

    # 不设置 stop token，观察模型自然停止行为
    temperature = 0.1
    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=args.max_tokens,
        # 故意不加 stop=["<|im_end|>"]，看模型自己何时停
    )

    while True:
        print("=" * 60)
        lines = []
        while True:
            try:
                line = input("Q> " if not lines else "   ")
            except EOFError:
                return
            if line.strip().lower() == "q":
                return
            if line == "" and lines:
                break
            if line:
                lines.append(line)

        question = " ".join(lines)
        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": question},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        output = llm.generate([prompt], sampling)[0]
        result = output.outputs[0]

        print(f"\n--- 响应 ({result.token_ids.__len__()} tokens) ---")
        print(result.text)
        print(f"\n--- 停止原因: {result.finish_reason} ---temperature{temperature}")
        # finish_reason: "stop" = EOS 或 stop token；"length" = max_tokens 截断
        if result.finish_reason == "stop":
            print("✓ 模型自然停止（生成了 EOS token）")
        else:
            print("✗ 被 max_tokens 强制截断")
        print()


if __name__ == "__main__":
    main()

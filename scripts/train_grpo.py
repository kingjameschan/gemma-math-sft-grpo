"""
GRPO fine-tuning for math reasoning on GSM8K.
支持从 SFT adapter 续训（merge SFT → 加新 LoRA → GRPO 在线 RL）。

用法：
  # 从 SFT checkpoint 开始 GRPO
  ~/vllm-env/bin/python scripts/train_grpo.py \
    --base_model models/gemma-2-2b-it \
    --sft_adapter checkpoints/gemma2-2b-it-sft-lr1e5-r8-fa2/checkpoint-50 \
    --output_dir checkpoints/gemma2-2b-it-grpo \
    --lr 5e-6 --epochs 1

  # 直接在 IT 模型上 GRPO（不经 SFT）
  ~/vllm-env/bin/python scripts/train_grpo.py \
    --base_model models/gemma-2-2b-it \
    --output_dir checkpoints/gemma2-2b-it-grpo-nosft
"""
import argparse
import json
import os
import re
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Please solve the following math problem step by step "
    "and provide the final answer at the end preceded by ####."
)


# ── 答案提取（与所有评测脚本一致）──────────────────────────────────────────
def extract_answer(text: str):
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


# ── 奖励函数 ───────────────────────────────────────────────────────────────
def make_reward_fn(gold_answers: list[str]):
    """返回一个 reward_fn(completions, **kwargs) -> list[float]"""
    def reward_fn(completions, **kwargs):
        rewards = []
        for completion, gold in zip(completions, gold_answers[: len(completions)]):
            # completions 可能是字符串列表或 dict 列表
            text = completion if isinstance(completion, str) else completion.get("content", "")
            pred = extract_answer(text)
            try:
                correct = pred is not None and abs(float(pred) - float(gold)) < 1e-4
            except ValueError:
                correct = False
            rewards.append(1.0 if correct else 0.0)
        return rewards
    return reward_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",  default="models/gemma-2-2b-it")
    parser.add_argument("--sft_adapter", default=None, help="SFT adapter 路径，merge 后作为 GRPO 起点")
    parser.add_argument("--output_dir",  default="checkpoints/gemma2-2b-it-grpo")
    parser.add_argument("--lr",          type=float, default=5e-6)
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--lora_rank",   type=int,   default=8)
    parser.add_argument("--qlora",       action="store_true", help="4-bit QLoRA 量化")
    parser.add_argument("--fp8",         action="store_true", help="8-bit 量化（省显存给 vLLM）")
    parser.add_argument("--batch_size",  type=int,   default=4,  help="per_device train batch")
    parser.add_argument("--num_generations", type=int, default=4, help="每题生成几条候选（G）")
    parser.add_argument("--max_new_tokens",  type=int, default=512)
    parser.add_argument("--beta",        type=float, default=0.04, help="KL penalty coefficient")
    parser.add_argument("--use_vllm",    action="store_true", help="用 vLLM 加速生成")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.3, help="vLLM GPU memory 占比")
    parser.add_argument("--num_samples",   type=int, default=0, help="使用前 N 个训练样本，0=全量")
    args = parser.parse_args()

    model_id = os.path.join(project_root, args.base_model)
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ── 加载 tokenizer ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # GRPO 生成需要 left padding

    # ── 加载模型 ──────────────────────────────────────────────────────────
    if args.qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    elif args.fp8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # ── Merge SFT adapter（如果指定）────────────────────────────────────
    if args.sft_adapter:
        sft_path = args.sft_adapter if os.path.isabs(args.sft_adapter) \
            else os.path.join(project_root, args.sft_adapter)
        print(f"Merging SFT adapter: {sft_path}")
        model = PeftModel.from_pretrained(model, sft_path, is_trainable=False)
        model = model.merge_and_unload()
        # base weights = SFT model; fresh LoRA for GRPO added below
        # beta>0 时 ref = LoRA disabled = SFT merged（正确的 reference）

    # ── LoRA 配置 ─────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── 数据集 ────────────────────────────────────────────────────────────
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = getattr(cfg, "model_type", "")

    train_path = os.path.join(project_root, "data/gsm8k/train.jsonl")
    raw = [json.loads(l) for l in open(train_path)]

    def to_prompt(item):
        if model_type == "gemma2":
            # Gemma2 不支持 system role，折叠进 user message
            messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\nProblem: " + item["question"]}]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": item["question"]},
            ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # GRPOTrainer 期望 dataset 有 "prompt" 字段
    gold_answers = [extract_answer(item["answer"]) for item in raw]
    dataset = Dataset.from_dict({
        "prompt": [to_prompt(item) for item in raw],
        "gold":   gold_answers,
    })

    if args.num_samples > 0:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
        print(f"截断到前 {len(dataset)} 条（--num_samples={args.num_samples}）")

    # 采样日志
    completions_log_path = os.path.join(output_dir, "completions_log.jsonl")
    _reward_call_count = [0]  # mutable counter in closure

    # 奖励函数需要逐样本 gold，通过 dataset 的 "gold" 字段传入
    def reward_fn(completions, gold, **kwargs):
        _reward_call_count[0] += 1
        rewards = []
        log_entries = []
        for completion, g in zip(completions, gold):
            text = completion if isinstance(completion, str) else completion.get("content", "")

            # 格式检测：#### 后只跟数字（到行尾），更严格
            m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)\s*$", text, re.MULTILINE)
            has_clean_format = m is not None
            pred_str = m.group(1).replace(",", "") if m else None

            # fallback：最后一个数字
            if pred_str is None:
                nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
                pred_str = nums[-1].replace(",", "") if nums else None

            try:
                correct = pred_str is not None and abs(float(pred_str) - float(g)) < 1e-4
            except (ValueError, TypeError):
                correct = False

            # 长度检查：防止模型只输出 "#### 42" 不推理
            too_short = len(text.split()) < 30

            if too_short:
                r = -1.0
            elif correct and has_clean_format:
                r = 1.0
            elif correct:
                r = 0.2
            elif has_clean_format:
                r = 0.0
            else:
                r = -1.0
            rewards.append(r)
            log_entries.append({
                "gold": g, "pred": pred_str, "reward": r,
                "correct": correct, "clean": has_clean_format,
                "text": text[:800],
            })

        # 每 10 次调用记录一次（避免日志太大）
        if _reward_call_count[0] % 10 == 1:
            with open(completions_log_path, "a") as f:
                f.write(json.dumps({"call": _reward_call_count[0], "samples": log_entries}, ensure_ascii=False) + "\n")

        return rewards

    # ── GRPO 训练配置 ─────────────────────────────────────────────────────
    grpo_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        temperature=0.7,              # 生成时采样温度
        save_steps=50,
        save_total_limit=3,
        logging_steps=10,
        report_to="none",
        # GRPO 特有
        beta=args.beta,               # KL 惩罚系数
    )
    if args.use_vllm:
        grpo_kwargs.update(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=args.vllm_gpu_util,
        )
    training_args = GRPOConfig(**grpo_kwargs)

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        peft_config=peft_config,
    )

    import glob
    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    resume = max(ckpts, key=lambda x: int(x.split("-")[-1])) if ckpts else None
    print(f"GRPO training starting! (resume={resume})")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(os.path.join(output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_adapter"))
    print(f"\n训练完成，adapter 保存至: {output_dir}/final_adapter")


if __name__ == "__main__":
    main()

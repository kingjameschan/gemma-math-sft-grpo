"""
GRPO 全量训练脚本 — L4 优化版。
支持 vLLM colocate 采样加速 + bf16/fp8 训练。

用法（GCP L4 实例）：
  python3 gcp/train_grpo_l4.py \
    --base_model ~/models/gemma-2-2b-it \
    --sft_adapter checkpoints/gemma2-2b-it-sft-lr1e5-r8-fa2/checkpoint-50 \
    --output_dir checkpoints/gemma2-2b-it-grpo-l4 \
    --use_vllm --vllm_gpu_util 0.4 \
    --num_generations 8 --batch_size 4
"""
import argparse, json, os, re, sys, glob
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Please solve the following math problem step by step "
    "and provide the final answer at the end preceded by ####."
)


def extract_answer(text: str):
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",  default="models/gemma-2-2b-it")
    parser.add_argument("--sft_adapter", default=None)
    parser.add_argument("--output_dir",  default="checkpoints/gemma2-2b-it-grpo-l4")
    parser.add_argument("--lr",          type=float, default=5e-6)
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--lora_rank",   type=int,   default=8)
    parser.add_argument("--fp8",         action="store_true", help="8-bit quantization")
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_new_tokens",  type=int, default=384)
    parser.add_argument("--beta",        type=float, default=0.04)
    parser.add_argument("--use_vllm",    action="store_true")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.4)
    parser.add_argument("--num_samples", type=int, default=0, help="0=全量")
    parser.add_argument("--save_steps",  type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model_id = args.base_model if os.path.isabs(args.base_model) else os.path.join(project_root, args.base_model)
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Model
    if args.fp8:
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

    # Merge SFT adapter
    if args.sft_adapter:
        sft_path = args.sft_adapter if os.path.isabs(args.sft_adapter) \
            else os.path.join(project_root, args.sft_adapter)
        print(f"Merging SFT adapter: {sft_path}")
        model = PeftModel.from_pretrained(model, sft_path, is_trainable=False)
        model = model.merge_and_unload()

    # LoRA
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Dataset
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = getattr(cfg, "model_type", "")
    train_path = os.path.join(project_root, "data/gsm8k/train.jsonl")
    raw = [json.loads(l) for l in open(train_path)]

    def to_prompt(item):
        if model_type == "gemma2":
            messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\nProblem: " + item["question"]}]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": item["question"]},
            ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    gold_answers = [extract_answer(item["answer"]) for item in raw]
    dataset = Dataset.from_dict({
        "prompt": [to_prompt(item) for item in raw],
        "gold":   gold_answers,
    })

    if args.num_samples > 0:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
        print(f"Using {len(dataset)} samples (--num_samples={args.num_samples})")
    else:
        print(f"Using full dataset: {len(dataset)} samples")

    # Completions log
    completions_log_path = os.path.join(output_dir, "completions_log.jsonl")
    _reward_call_count = [0]

    def reward_fn(completions, gold, **kwargs):
        _reward_call_count[0] += 1
        rewards = []
        log_entries = []
        for completion, g in zip(completions, gold):
            text = completion if isinstance(completion, str) else completion.get("content", "")
            m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)\s*$", text, re.MULTILINE)
            has_clean_format = m is not None
            pred_str = m.group(1).replace(",", "") if m else None
            if pred_str is None:
                nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
                pred_str = nums[-1].replace(",", "") if nums else None
            try:
                correct = pred_str is not None and abs(float(pred_str) - float(g)) < 1e-4
            except (ValueError, TypeError):
                correct = False
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
        if _reward_call_count[0] % 10 == 1:
            with open(completions_log_path, "a") as f:
                f.write(json.dumps({"call": _reward_call_count[0], "samples": log_entries}, ensure_ascii=False) + "\n")
        return rewards

    # Training config
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
        temperature=args.temperature,
        save_steps=args.save_steps,
        save_total_limit=None,  # 保留所有 checkpoint，训练完再统一评测
        logging_steps=10,
        report_to="none",
        beta=args.beta,
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

    # Auto-resume
    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    resume = max(ckpts, key=lambda x: int(x.split("-")[-1])) if ckpts else None
    print(f"GRPO training starting! (resume={resume})")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(os.path.join(output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_adapter"))
    print(f"\nDone! Adapter saved to: {output_dir}/final_adapter")


if __name__ == "__main__":
    main()

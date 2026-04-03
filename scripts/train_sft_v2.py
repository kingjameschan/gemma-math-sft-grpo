import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Gemma2 chat template，注入 {% generation %} 标记（适用于 gemma-2-*-it）
# Gemma2 不支持 system role，跳过 system 消息（IT 模型已内化指令遵循能力）
GEMMA2_CHAT_TEMPLATE = (
    "{% set ns = namespace(system_content='') %}"
    "{% for message in messages %}"
    "{%- if message.role == 'system' %}"
    "{% set ns.system_content = message.content %}"
    "{%- elif message.role == 'user' %}"
    "{%- if ns.system_content %}"
    "{{- '<start_of_turn>user\\n' + ns.system_content + '\\n\\n' + message.content + '<end_of_turn>\\n' }}"
    "{% set ns.system_content = '' %}"
    "{%- else %}"
    "{{- '<start_of_turn>user\\n' + message.content + '<end_of_turn>\\n' }}"
    "{%- endif %}"
    "{%- elif message.role == 'assistant' %}"
    "{{- '<start_of_turn>model\\n' }}"
    "{% generation %}{{- message.content + '<end_of_turn>\\n' }}{% endgeneration %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<start_of_turn>model\\n' }}"
    "{%- endif %}"
)

# Qwen3 使用简化的 ChatML 模板（禁用 thinking），注入 {% generation %} 标记
QWEN3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{%- if message.role == 'system' %}"
    "{{- '<|im_start|>system\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'user' %}"
    "{{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{% generation %}{{- message.content }}{% endgeneration %}"
    "{{- '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",  type=str, default=None, help="基座模型路径（相对项目根或绝对路径），默认 models/Qwen2.5-7B")
    parser.add_argument("--output_dir",  type=str, default=None, help="checkpoint 输出目录，默认 checkpoints/qwen2.5-7b-sft-v2")
    parser.add_argument("--lr",          type=float, default=1e-4, help="学习率，默认 1e-4（小模型建议 2e-4）")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--lora_rank",   type=int,   default=16, help="LoRA rank，默认 16（搜索时可试 8/16/32）")
    parser.add_argument("--qlora",       action="store_true",    help="使用 QLoRA 4-bit 量化（大模型如 7B/8B 必须）")
    args = parser.parse_args()

    # --- 路径解析 ---
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    def resolve(p, default):
        if p is None:
            return os.path.join(project_root, default)
        return p if os.path.isabs(p) else os.path.join(project_root, p)

    model_id          = resolve(args.base_model, "models/Qwen2.5-7B")
    output_dir        = resolve(args.output_dir, "checkpoints/qwen2.5-7b-sft-v2")
    train_data_path   = os.path.join(project_root, "data/sft_formatted/train_sft.jsonl")

    print(f"基座模型:  {model_id}")
    print(f"输出目录:  {output_dir}")
    print(f"学习率:    {args.lr}")
    print(f"LoRA rank: {args.lora_rank}")

    # --- 加载分词器 ---
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 注入 {% generation %} 标记（TRL assistant_only_loss 需要）
    # Qwen3 模板复杂（含 thinking/tool blocks），直接替换为简化版
    # Qwen2.5 模板简单，做最小化 patch
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    if model_type == "qwen3":
        tokenizer.chat_template = QWEN3_CHAT_TEMPLATE
        print(f"Qwen3 检测到 (model_type={model_type})，使用自定义 chat template")
    elif model_type == "gemma2":
        tokenizer.chat_template = GEMMA2_CHAT_TEMPLATE
        print(f"Gemma2 检测到 (model_type={model_type})，使用 Gemma2 chat template")
    else:
        # Qwen2.5 原始模板最小化 patch
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "{%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}",
            "{%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" and not message.tool_calls %}\n        {{- '<|im_start|>' + message.role + '\\n' }}{% generation %}{{- message.content }}{% endgeneration %}{{- '<|im_end|>' + '\\n' }}"
        )
        print(f"Qwen2.5 检测到 (model_type={model_type})，使用 patch chat template")

    # --- 加载模型（bf16 或 QLoRA 4-bit）---
    if args.qlora:
        print("正在加载 QLoRA 4-bit 基座模型...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
    else:
        print("正在加载 bf16 基座模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        model.config.use_cache = False

    # --- 配置 LoRA ---
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- 加载数据集 ---
    print("正在加载数据集...")
    dataset = load_dataset("json", data_files=train_data_path, split="train")
    dataset = dataset.shuffle(seed=42)
    eval_dataset  = dataset.select(range(200))
    train_dataset = dataset.select(range(200, len(dataset)))

    # --- 训练参数 ---
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=1,
        learning_rate=args.lr,
        bf16=True,
        optim="paged_adamw_32bit" if args.qlora else "adamw_torch",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=512,
        packing=True,
        report_to="none",
        assistant_only_loss=True,
        gradient_checkpointing=True,
    )

    # --- 初始化训练器 ---
    print("初始化 SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    # --- 开始训练 ---
    print("SFT 训练开始！")
    # 有 checkpoint 则续跑，否则从头开始
    import glob
    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    resume = max(ckpts, key=lambda x: int(x.split("-")[-1])) if ckpts else None
    trainer.train(resume_from_checkpoint=resume)

    print(f"训练完成！最终权重已保存至 {output_dir}")
    trainer.model.save_pretrained(os.path.join(output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_adapter"))


if __name__ == "__main__":
    main()

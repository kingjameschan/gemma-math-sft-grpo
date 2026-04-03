"""
DPO 训练脚本（v2）。
Reference model = SFT-merged（通过禁用新 LoRA 实现，显存高效）
训练起点  = SFT-merged + 新 LoRA adapter

做法：先把 SFT LoRA merge 到基座权重，再加一个新 LoRA 作为 DPO 可训练层。
这样 ref_model=None 时，TRL 禁用 LoRA 得到的就是 SFT merged 模型（正确的 reference）。
原来的做法（ref_model=None + SFT LoRA 可训练）会让 reference 退化为裸基座模型。

用法示例：
  python scripts/train_dpo.py --num_pairs 100 --run_name dpo_100
  python scripts/train_dpo.py --num_pairs 0   --run_name dpo_all   # 0 = 全量
"""
import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

# --- CLI 参数 ---
parser = argparse.ArgumentParser()
parser.add_argument("--base_model",  type=str, default=None, help="基座模型路径（相对项目根或绝对路径），默认 models/Qwen2.5-7B")
parser.add_argument("--sft_adapter", type=str, default=None, help="SFT adapter 路径，默认 checkpoints/qwen2.5-7b-sft-v2/checkpoint-800")
parser.add_argument("--data_path",   type=str, default=None, help="DPO jsonl 路径，默认 data/dpo/dpo_train_full.jsonl")
parser.add_argument("--output_dir",  type=str, default=None, help="输出目录（相对项目根或绝对路径）")
parser.add_argument("--num_pairs",   type=int, default=0,    help="使用前 N 对数据，0 = 全量")
parser.add_argument("--run_name",    type=str, default="dpo_run", help="输出目录名，当 --output_dir 未指定时用于构造默认路径")
parser.add_argument("--qlora",       action="store_true", help="使用 4-bit QLoRA（默认 bf16 全精度）")
parser.add_argument("--lora_rank",   type=int, default=16)
parser.add_argument("--lr",          type=float, default=5e-6, help="学习率，默认 5e-6")
parser.add_argument("--beta",        type=float, default=0.3,  help="KL 约束强度，默认 0.3")
args = parser.parse_args()

# --- 路径配置 ---
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

def _resolve(p, default):
    if p is None:
        return os.path.join(project_root, default)
    return p if os.path.isabs(p) else os.path.join(project_root, p)

base_model_path  = _resolve(args.base_model,  "models/Qwen2.5-7B")
sft_adapter_path = _resolve(args.sft_adapter, "checkpoints/qwen2.5-7b-sft-v2/checkpoint-800")
dpo_data_path    = _resolve(args.data_path,   "data/dpo/dpo_train_full.jsonl")
if args.output_dir:
    output_dir = _resolve(args.output_dir, "")
else:
    output_dir = os.path.join(project_root, f"checkpoints/qwen2.5-7b-dpo/{args.run_name}")


def main():
    # --- 1. 加载分词器 ---
    print("🚀 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO 推荐 left padding

    # --- 2. 加载基座模型 ---
    if args.qlora:
        print("🚀 加载 4-bit QLoRA 基座模型...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
    else:
        print("🚀 加载 bf16 基座模型...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # --- 3. 挂载 SFT adapter 并 merge 进基座权重 ---
    # merge 后基座权重 = SFT 模型，之后加新 LoRA 做 DPO 训练。
    # ref_model=None 时 TRL 禁用新 LoRA → reference = SFT merged 模型（正确！）
    print(f"🔌 加载 SFT adapter 并 merge: {sft_adapter_path}")
    model = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=False)
    model = model.merge_and_unload()
    print("✅ SFT adapter 已 merge 到基座权重")

    # --- 4. 加新 LoRA 作为 DPO 可训练层 ---
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 5. Reference model = None（TRL 禁用新 LoRA → reference = SFT merged 模型）---
    ref_model = None

    # --- 6. 加载 DPO 数据集 ---
    print("📂 加载 DPO 数据集...")
    dataset = load_dataset("json", data_files=dpo_data_path, split="train")
    dataset = dataset.shuffle(seed=42)
    if args.num_pairs > 0:
        dataset = dataset.select(range(min(args.num_pairs, len(dataset))))
        print(f"   截断到前 {len(dataset)} 对（--num_pairs={args.num_pairs}）")
    eval_size = min(20, int(len(dataset) * 0.1))
    eval_dataset  = dataset.select(range(eval_size))
    train_dataset = dataset.select(range(eval_size, len(dataset)))
    print(f"   训练: {len(train_dataset)} 条，评估: {len(eval_dataset)} 条")

    # --- 7. DPO 训练配置 ---
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=3,             # 原来 1 epoch 步数太少，改为 3
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
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
        beta=args.beta,          # KL 约束强度
        max_length=512,
        padding_free=True,      # FA2 varlen，消除 padding，类似 packing
        report_to="none",
    )

    # --- 8. 初始化 DPOTrainer ---
    print("🚀 初始化 DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # --- 9. 训练 ---
    print("🔥 DPO 训练开始！")
    trainer.train()

    print(f"🎉 DPO 训练完成！权重保存至 {output_dir}")
    trainer.model.save_pretrained(os.path.join(output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_adapter"))


if __name__ == "__main__":
    main()

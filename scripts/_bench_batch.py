"""
快速测试不同 batch size 下的速度和显存占用。
用法：python scripts/_bench_batch.py
"""
import os, time, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

model_id        = os.path.join(project_root, "models/Qwen3-1.7B-Base")
train_data_path = os.path.join(project_root, "data/sft_formatted/train_sft.jsonl")

QWEN3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{%- if message.role == 'system' %}{{- '<|im_start|>system\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'user' %}{{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'assistant' %}{{- '<|im_start|>assistant\\n' }}{% generation %}{{- message.content }}{% endgeneration %}{{- '<|im_end|>\\n' }}"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)

BATCH_SIZES = [6, 7, 8, 9, 10, 11, 12]
STEPS = 20  # 每个 batch size 跑 20 步

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.chat_template = QWEN3_CHAT_TEMPLATE

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config,
    device_map="auto", trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)

dataset = load_dataset("json", data_files=train_data_path, split="train")
dataset = dataset.shuffle(seed=42).select(range(500))

print(f"\n{'='*55}")
print(f"{'batch':>6} {'steps':>6} {'s/step':>8} {'tok/s':>10} {'VRAM GB':>9}")
print(f"{'='*55}")

for bs in BATCH_SIZES:
    torch.cuda.reset_peak_memory_stats()
    try:
        args = SFTConfig(
            output_dir=os.path.join(project_root, f"checkpoints/_bench_bs{bs}"),
            num_train_epochs=99,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            bf16=True,
            optim="paged_adamw_32bit",
            logging_steps=STEPS,
            max_steps=STEPS,
            max_length=512,
            packing=True,
            report_to="none",
            assistant_only_loss=True,
            save_strategy="no",
            eval_strategy="no",
        )
        trainer = SFTTrainer(
            model=model, train_dataset=dataset,
            peft_config=peft_config, processing_class=tokenizer, args=args,
        )
        t0 = time.time()
        trainer.train()
        elapsed = time.time() - t0
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        s_per_step = elapsed / STEPS
        # tokens/sec: STEPS * bs * 512 (approx packed tokens)
        tps = STEPS * bs * 512 / elapsed
        print(f"{bs:>6} {STEPS:>6} {s_per_step:>8.2f} {tps:>10.0f} {vram_gb:>9.2f}")
        # 清理 trainer 释放显存
        del trainer
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"{bs:>6} {'OOM':>6}")
            torch.cuda.empty_cache()
        else:
            raise

import shutil, glob
for d in glob.glob(os.path.join(project_root, "checkpoints/_bench_bs*")):
    shutil.rmtree(d, ignore_errors=True)
print("\n测试完成，临时 checkpoint 已清理。")

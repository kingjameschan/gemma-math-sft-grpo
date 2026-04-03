"""快速找 DPO 最大 batch size（不实际训练，只做一次 forward+backward）"""
import os, sys, torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_model_path = os.path.join(project_root, "models/gemma-2-2b-it")
sft_adapter_path = os.path.join(project_root, "checkpoints/gemma2-2b-it-sft-lr1e5-r8-fa2/checkpoint-50")
data_path = os.path.join(project_root, "data/dpo/dpo_gemma2_2b.jsonl")

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model.config.use_cache = False
model.enable_input_require_grads()

model = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=False)
model = model.merge_and_unload()

lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files=data_path, split="train").select(range(64))

for bs in [1, 2, 4, 8]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        args = DPOConfig(
            output_dir="/tmp/dpo_test",
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=1,
            max_steps=2,
            bf16=True,
            gradient_checkpointing=True,
            padding_free=True,
            beta=0.1,
            max_length=512,
            logging_steps=1,
            report_to="none",
            save_strategy="no",
        )
        trainer = DPOTrainer(
            model=model, ref_model=None, args=args,
            train_dataset=dataset, processing_class=tokenizer,
        )
        trainer.train()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"batch_size={bs}: OK, peak VRAM={peak:.1f}GB")
        del trainer
    except torch.cuda.OutOfMemoryError:
        print(f"batch_size={bs}: OOM!")
        break
    except Exception as e:
        print(f"batch_size={bs}: Error: {e}")
        break

print("Done!")

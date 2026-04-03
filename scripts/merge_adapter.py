"""
通用 adapter merge 脚本。支持两种模式：

1. SFT merge（--mode sft）：
   base model → 挂载 SFT adapter → merge → 保存
   用法：
     python scripts/merge_adapter.py --mode sft \
       --base_model models/Qwen2.5-7B \
       --sft_adapter checkpoints/qwen2.5-7b-sft-v2/checkpoint-800 \
       --output_dir models/Qwen2.5-7B-SFT-merged

2. DPO merge（--mode dpo，默认）：
   SFT-merged model → 挂载 DPO adapter → merge → 保存
   用法：
     python scripts/merge_adapter.py --mode dpo \
       --base_model models/Qwen2.5-7B-SFT-merged \
       --dpo_adapter checkpoints/qwen2.5-7b-dpo/dpo_400_v2/final_adapter \
       --output_dir models/Qwen2.5-7B-DPO-400-v2-merged

所有路径支持相对路径（相对项目根目录）或绝对路径。
"""
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--mode",        type=str, default="dpo", choices=["sft", "dpo"], help="merge 模式")
parser.add_argument("--base_model",  type=str, required=True, help="基座/SFT-merged 模型路径")
parser.add_argument("--sft_adapter", type=str, default=None,  help="SFT adapter 路径（--mode sft 时使用）")
parser.add_argument("--dpo_adapter", type=str, default=None,  help="DPO adapter 路径（--mode dpo 时使用）")
parser.add_argument("--output_dir",  type=str, required=True, help="输出路径")
args = parser.parse_args()

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

def resolve(p):
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.join(project_root, p)

base_model_path = resolve(args.base_model)
output_dir      = resolve(args.output_dir)

if args.mode == "sft":
    adapter_path = resolve(args.sft_adapter)
    if adapter_path is None:
        raise ValueError("--mode sft 需要指定 --sft_adapter")
else:
    adapter_path = resolve(args.dpo_adapter)
    if adapter_path is None:
        raise ValueError("--mode dpo 需要指定 --dpo_adapter")

print(f"模式:      {args.mode}")
print(f"基座模型:  {base_model_path}")
print(f"Adapter:   {adapter_path}")
print(f"输出:      {output_dir}")

print("\n加载模型 (bf16, CPU)...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

print("挂载 adapter...")
model = PeftModel.from_pretrained(model, adapter_path)

print("Merge and unload...")
model = model.merge_and_unload()

os.makedirs(output_dir, exist_ok=True)
print(f"保存到 {output_dir} ...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("完成！")

# GCP Gemma2-2B GRPO 评测指南

你在一台 GCP T4 实例上。任务：评测 Gemma2-2B-IT GRPO checkpoint 的数学推理能力。

## Step 1: 环境搭建

```bash
pip install vllm peft transformers huggingface_hub
```

## Step 2: 下载模型和数据

```bash
# 下载 Gemma2-2B-IT base model
huggingface-cli download google/gemma-2-2b-it --local-dir ~/gemma-2-2b-it

# 下载 GSM8K 测试集（1319题）
mkdir -p ~/data
python3 -c "
from datasets import load_dataset
import json
ds = load_dataset('openai/gsm8k', 'main', split='test')
with open('$HOME/data/test.jsonl', 'w') as f:
    for item in ds:
        f.write(json.dumps({'question': item['question'], 'answer': item['answer']}) + '\n')
print(f'Saved {len(ds)} samples')
"
```

如果没有 datasets 库，用备选方案：
```bash
pip install datasets
```

## Step 3: 下载 SFT adapter（GRPO 评测需要两步 merge）

```bash
# 从用户本地传来（用户会执行 gcloud compute scp）
# SFT adapter 会放在 ~/sft-adapter/
# GRPO checkpoint 会放在 ~/grpo-checkpoints/checkpoint-XXX/
```

等用户传好后确认文件在位：
```bash
ls ~/sft-adapter/adapter_config.json
ls ~/grpo-checkpoints/
```

## Step 4: 评测

用下面的 Python 脚本评测。核心流程：base model + SFT merge + GRPO merge → vLLM 推理 → 三指标统计。

```bash
python3 ~/eval_grpo.py \
  --base_model ~/gemma-2-2b-it \
  --sft_adapter ~/sft-adapter \
  --checkpoint_dir ~/grpo-checkpoints \
  --output_dir ~/eval_results
```

## 三个核心指标

评测必须报告这三个指标（不是普通 accuracy）：

1. **####率**: 输出中包含 `####` 的比例
2. **数字正确**: 用 fallback（最后一个数字）判断答案是否正确
3. **严格正确**（主指标）: `#### 纯数字`（不带单位）且数值正确
   - 正则: `re.search(r"####\s*(-?\d[\d,]*\.?\d*)\s*$", text, re.MULTILINE)`
   - 这是最终优化目标

## 注意事项

- Gemma2 不支持 system role，system prompt 要折入 user message
- System prompt: "You are a mathematical reasoning assistant. Please solve the following math problem step by step and provide the final answer at the end preceded by ####."
- vLLM 推理用 temperature=0, max_tokens=512
- stop tokens: `["<end_of_turn>"]`
- T4 16GB 跑 bf16 Gemma2-2B 没问题，gpu_memory_utilization 用 0.85

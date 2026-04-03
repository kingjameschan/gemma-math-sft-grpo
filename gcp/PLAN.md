# GCP L4 GRPO 全量训练计划

## Step 1: 环境搭建

```bash
pip install vllm peft transformers huggingface_hub datasets trl
```

## Step 2: 下载模型和数据

```bash
# 下载 base model
huggingface-cli download google/gemma-2-2b-it --local-dir ~/models/gemma-2-2b-it

# 下载 GSM8K 数据集
python3 gcp/download_data.py
```

## Step 3: 确认文件就位

用户会通过 git clone 把以下文件带过来：
- `gcp/train_grpo_l4.py` — L4 优化版 GRPO 训练脚本
- `gcp/eval_grpo.py` — 三指标评测脚本
- `gcp/download_data.py` — 数据下载脚本
- `checkpoints/gemma2-2b-it-sft-lr1e5-r8-fa2/checkpoint-50/` — SFT adapter

如果 SFT adapter 不在 repo 里（太大），用户会用 `gcloud compute scp` 传过来。
确认：
```bash
ls ~/models/gemma-2-2b-it/config.json
ls checkpoints/gemma2-2b-it-sft-lr1e5-r8-fa2/checkpoint-50/adapter_config.json
ls data/gsm8k/train.jsonl
ls data/gsm8k/test.jsonl
```

## Step 4: GRPO 全量训练

```bash
python3 gcp/train_grpo_l4.py \
  --base_model ~/models/gemma-2-2b-it \
  --sft_adapter checkpoints/gemma2-2b-it-sft-lr1e5-r8-fa2/checkpoint-50 \
  --output_dir checkpoints/gemma2-2b-it-grpo-l4 \
  --lr 5e-6 --beta 0.04 --epochs 1 \
  --use_vllm --vllm_gpu_util 0.4 \
  --num_generations 8 --batch_size 4
```

预计 ~5-6 小时（全量 7473 样本，~1868 步）。

创建 5 分钟进度监控：每 5 分钟汇报当前 step 和 reward。

## Step 5: 评测所有 checkpoint

训练完成后，评测所有保存的 checkpoint：

```bash
python3 gcp/eval_grpo.py \
  --base_model ~/models/gemma-2-2b-it \
  --sft_adapter checkpoints/gemma2-2b-it-sft-lr1e5-r8-fa2/checkpoint-50 \
  --checkpoint_dir checkpoints/gemma2-2b-it-grpo-l4 \
  --output_dir eval_results
```

## Step 6: 结果回传

评测完成后，告知用户：
1. 汇总三指标表格
2. 最佳 checkpoint 是哪个
3. 用户需要执行的 scp 命令把结果拉回本地

## 本地试跑基线（供对比）

| Step | ####率 | 数字正确 | 严格正确 |
|------|--------|---------|---------|
| 200 | 97.12% | 68.01% | 61.71% |
| 400 | 97.95% | 68.08% | 62.47% |

目标：全量训练后严格正确超过 63%。

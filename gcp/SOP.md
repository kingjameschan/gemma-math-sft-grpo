# GCP 实例使用 SOP

## 1. 开机

```bash
# 本地 WSL
gcloud compute instances start <实例名> --zone=<zone>
```

## 2. SSH 进入

```bash
gcloud compute ssh <实例名> --zone=<zone> --tunnel-through-iap
```

## 3. 拉代码（首次）

```bash
git clone https://github.com/KingjamesChan/gemma-math-sft-grpo.git
cd gemma-math-sft-grpo
git lfs pull   # 拉 LFS 大文件（SFT adapter）
```

如果已经 clone 过：
```bash
cd gemma-math-sft-grpo && git pull && git lfs pull
```

## 4. 环境配置（首次）

```bash
pip install vllm peft transformers huggingface_hub datasets trl
huggingface-cli login   # 粘贴 HF token（Gemma 需要授权）
```

## 5. 下载模型（首次）

```bash
huggingface-cli download google/gemma-2-2b-it --local-dir ~/models/gemma-2-2b-it
```

## 6. 下载数据（首次）

```bash
python3 gcp/download_data.py
```

## 7. 执行任务

训练：
```bash
python3 gcp/train_grpo_l4.py \
  --base_model ~/models/gemma-2-2b-it \
  --sft_adapter checkpoints/gemma2-2b-it-sft-lr1e5-r8-fa2/checkpoint-50 \
  --output_dir checkpoints/gemma2-2b-it-grpo-l4 \
  --use_vllm --vllm_gpu_util 0.4 \
  --num_generations 8 --batch_size 4
```

评测：
```bash
python3 gcp/eval_grpo.py \
  --base_model ~/models/gemma-2-2b-it \
  --sft_adapter checkpoints/gemma2-2b-it-sft-lr1e5-r8-fa2/checkpoint-50 \
  --checkpoint_dir checkpoints/gemma2-2b-it-grpo-l4 \
  --output_dir eval_results
```

## 8. 回收结果

```bash
# 实例上：把结果 push 回 repo
git add eval_results/ checkpoints/gemma2-2b-it-grpo-l4/
git commit -m "L4 GRPO results"
git push
```

或者 scp 回本地：
```bash
# 本地 WSL
gcloud compute scp --recurse <实例名>:~/gemma-math-sft-grpo/eval_results ./ --zone=<zone> --tunnel-through-iap
```

## 9. 关机（省钱！）

```bash
# 实例上
sudo shutdown -h now
```

或本地：
```bash
gcloud compute instances stop <实例名> --zone=<zone>
```

## 注意事项

- **用完必须关机**，T4 ~$0.35/h，L4 ~$0.70/h，开着不用也计费
- 模型和 pip 包装在实例磁盘上，关机不丢，下次开机还在
- 只有磁盘数据在开关机之间持久化，内存和 /tmp 会清空
- 训练跑长任务用 `nohup` 或 `tmux`，防止 SSH 断开后进程被杀

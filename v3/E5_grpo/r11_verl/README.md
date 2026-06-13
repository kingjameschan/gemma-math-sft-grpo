# R11 — verl DAPO 完整 6/6

R10 在 TRL 0.29 跑的 DAPO **4/6** 件套（缺 Dynamic Sampling + Soft Overlong + μ=1 让 Clip-Higher 死 flag）。R11 迁到 verl 框架补全 6/6 + 修 μ。

## 文件

| 文件 | 作用 |
|---|---|
| `reward_judge.py` | Gemini judge reward (port from R10), async signature for verl |
| `data_prep.py` | GSM8K jsonl → verl parquet schema |
| `run_dapo_r11.sh` | AWS 启动脚本, hydra overrides 全部超参 |

## R10 → R11 配置 diff

| | R10 (TRL) | R11 (verl) |
|---|---|---|
| Token-level loss | ✓ `loss_type=dapo` | ✓ `loss_agg_mode=token-mean` |
| Clip-Higher | ⚠ 配了但 μ=1 不触发 | ✓ μ=2 (`ppo_epochs=2`) 真触发 |
| Overlong Filter | ✓ `mask_truncated_completions` | ✓ `data.truncation='error'` + cap |
| **Dynamic Sampling** | ✗ TRL 不支持 | ✓ `filter_groups.enable=True` |
| **Soft Overlong** | ✗ `--no_length_penalty` | ✓ `overlong_buffer.enable=True` |
| KL β | 0 | 0 (一致) |
| lr | 1e-5 | 1e-5 |
| G | 8 | 8 |
| per_device | 16 | 16 |
| accum | 24 | 24 (mini-batch=384) |
| max_steps | 240 | 240 |
| save_freq | 4 | 4 |
| max_completion | 384 | 384 |

## AWS 部署步骤

### 1. clone verl + 装依赖
```bash
ssh aws
cd /home/ubuntu/v3/refs
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
pip install ray vllm pandas pyarrow google-genai
```

### 2. 准备数据 (本地一次)
```bash
cd /mnt/d/fine-tuning/v3/E5_grpo/r11_verl
python data_prep.py
# 输出: data/{train,dev,test}.parquet
# scp 上传:
aws s3 sync data s3://kingjameschan-fine-tuning-v3/r11_data/
# AWS 侧拉:
ssh aws "aws s3 sync s3://kingjameschan-fine-tuning-v3/r11_data /home/ubuntu/v3/E5_grpo/r11_verl/data"
```

### 3. 上传 reward + launcher
```bash
aws s3 cp /mnt/d/fine-tuning/v3/E5_grpo/r11_verl/reward_judge.py s3://kingjameschan-fine-tuning-v3/scripts/r11_reward_judge.py
aws s3 cp /mnt/d/fine-tuning/v3/E5_grpo/r11_verl/run_dapo_r11.sh s3://kingjameschan-fine-tuning-v3/scripts/run_dapo_r11.sh
# AWS 拉到本地, chmod +x, 跑
```

### 4. 启动
```bash
# 类似 R10, 用 SSM 传 GOOGLE_API_KEY 启动
aws ssm send-command ... (见 R10 的 ssm_launch_*.json 模板)
```

## 待验证 (TODO before launch)

★ verl 当前最新版的精确 hydra 路径，因为 verl API 改动较快：
- [ ] `actor_rollout_ref.model.lora_rank` 是否当前路径 (老版本可能是 `model.lora.rank`)
- [ ] `algorithm.filter_groups.metric` 选项 (`acc` / `score` / `seq_reward`)
- [ ] `reward_model.overlong_buffer` 是否还在 `reward_model` 命名空间下
- [ ] `actor_rollout_ref.actor.ppo_epochs` 还是 `num_ppo_epochs`
- [ ] vLLM colocate 在 verl 中的精确 mode 名 (可能是 `colocate` / `hybrid` / `inplace`)
- [ ] LoRA 在 vLLM rollout 时的同步机制 (`actor_rollout_ref.rollout.load_format=lora`)

跑前先在 AWS 上 dry-run:
```bash
python3 -m recipe.dapo.main_dapo --help 2>&1 | head -100   # 看精确 arg list
python3 -m recipe.dapo.main_dapo --cfg job 2>&1 | grep -E "filter_groups|overlong"  # 看默认 config
```

## 与 R10 直接对比

```
R10 (TRL):     base 61.71% → ?% (跑到 step 18, 还在噪声内)
R11 (verl):    base 61.71% → ?%

期望 delta:
  + Dynamic Sampling:    +1-3 pp (我们 frac_zero_std=0.30 中等水平)
  + Soft Overlong:       +0-1 pp (mean_len 195 远未达 384, 可能不激活)
  + μ=2 Clip-Higher:     +0-1 pp (ratio 真有 spread 时才有意义)
合计预期           +1-5 pp test pass@1 over R10

代价:
  judge API +30-50% (DS 过采样)
  GPU step time +30-50% (加上 DS over-sampling)
  实际 wall-clock 13h × 1.4 ≈ 18h
```

## verl 参考

- [DAPO recipe doc](https://verl.readthedocs.io/en/latest/algo/dapo.html)
- [run_dapo_qwen2.5_32b.sh template](https://github.com/verl-project/verl/blob/main/recipe/dapo/run_dapo_qwen2.5_32b.sh)
- [Custom reward function doc](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
- [LoRA in verl](https://verl.readthedocs.io/en/latest/perf/lora.html)

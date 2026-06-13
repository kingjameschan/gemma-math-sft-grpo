# R11 verl 部署 — 接力 handover

## 最新状态 (2026-05-09 凌晨)

R11 verl 整体 12 次 smoke 迭代后**仍未跑通**:
- v1-v6: hydra 路径 / TRL 版本 / TP / mode / V1 env / memory pool 各种 fix
- v7: 跑过 model 加载, 死在 verl bucketed_weight_transfer IPC IndexError
- v8: 把 verl 源码 patch 后, 跑过 IPC, 但 vLLM HTTP server init 挂死 25min
- v9-v10: enforce_eager 反而更糟, hang 在 multiproc spawn 之前
- v11+: 试图升 vllm 0.10.1.dev → 0.10.2 (verl/TRL 都明确支持的版本),
        docker commit 35GB 镜像在 SSM 上反复静默失败 (image 时间戳不变)

**根本原因**: 我们 docker 里的 vllm 0.10.1.dev (一个 dev 版) 与 verl + ray
              + torch 2.10 组合不稳。verl 源码假设的 IPC tensor 格式 / vLLM
              async server 协议 与 0.10.1.dev 不匹配 (我们打了 1 个 patch 修
              IPC, 但还有更深的 hang).

**继续推 R11 的可行路径**:
1. 用 Dockerfile 重 build 一个干净 image: torch 2.5 + vllm 0.8.4 (verl 期望)
2. 或: torch 2.10 保留, 升 vllm 到 0.11.x (TRL/verl 较新支持) — 注意
   docker commit 必须 SSH 直接做,不走 SSM (commit 需要长时间)
3. 重新 git checkout verl 到 reproduction commit (4f80e465c2ec...) +
   配套 vllm 0.8.4

**ROI 判断**:
R11 verl 实际跑通需要 build 干净 image (~30min) + 重 smoke 几轮 + 真跑 18-22h
= 总投入 ~24h.  增益预期 +1-5pp test pass@1 over R10.
对面试 portfolio 价值: 非常高 (展示框架对比 + 算法理解)
对实际数字提升: 中等 (我们 base 噪声 ±1pp, 5pp 增益勉强能区分)

更高 ROI 的 fallback: TRL 0.29 直接写 DAPO 3 件套 subclass (R12),
完全在 R10 镜像跑, 不踩 verl 的依赖坑. 见 README.md "R10 缺少哪些 DAPO 组件".

## 之前状态

R11 准备 90% 完成, smoke test 走到 model 加载阶段, 还差 1-3 个配置 fix。

```
✓  vllm-trl:v3-r11-verl 镜像已 build (verl + ray + trl==0.18 + pandas + pyarrow)
✓  parquet 数据 (6973/500/1319) 在 S3 + AWS
✓  launcher (run_dapo_r11.sh) 已 6 轮迭代修过:
   1. logger=[\"console\"] → [console]               (Hydra parser)
   2. reward_model.reward_kwargs → reward.reward_kwargs (top-key 错)
   3. + log_prob_micro_batch_size_per_gpu (rollout + ref)
   4. + tensor_model_parallel_size=1                (单 GPU, 默认 2)
   5. -v recipe → /opt/verl_pkg/recipe (运行时挂载, submodule)
   6. trl 0.29 → 0.18 修 AutoModelForCausalLMWithValueHead 缺失
✓  R11 watcher / convert_verl_ckpt_to_hf / excel skel
✓  GOOGLE_API_KEY 在 SSM Parameter Store (/v3/grpo/google_api_key)
✓  smoke wrapper r11_smoke_wrapper.sh 在 S3 + AWS

✗  下一步: smoke test 重跑
   命令: aws ssm send-command ... file:///tmp/ssm_smoke_v2.json ... 
        (file 在 /tmp/ssm_smoke_v2.json,本地有,可重新上传)
```

## 已知潜在 blocker (按概率排)

### 1. vLLM colocate 在 verl 中可能不 work-out-of-box
verl 默认 rollout.mode 是 `async` 不是 `colocate`。我们 launcher 设了 colocate 但 verl 的 colocate 可能要额外 config (e.g. `actor_rollout_ref.rollout.engine_kwargs`)。

如果 smoke 报 mode 相关错, 改 mode=async 试。

### 2. LoRA + vLLM 同步
verl 的 LoRA + vLLM rollout 同步机制 (`load_format=auto/lora/dummy`) 在不同版本不一致。
关键 flag: `actor_rollout_ref.rollout.load_format` (默认 `auto`, 可能要改 `safetensors`).

### 3. custom_reward_function 异步注入 GOOGLE_API_KEY
verl reward worker 是独立的 Ray actor, 环境变量从 ray runtime_env 传, 不直接继承 docker env。
我们的 reward_judge.py 用 `os.environ["GOOGLE_API_KEY"]`, ray actor 里可能取不到。

修法: 在 launcher 加 `+actor_rollout_ref.rollout.engine_kwargs.env={GOOGLE_API_KEY:$GOOGLE_API_KEY}` 或类似。

### NEW (2026-05-09 后续 smoke): hang in vLLM HTTP server init
smoke v8 用了 patched 镜像 (rebuild_ipc 加 len > 6 guard), 跑过 IPC 那层
但 vLLM Engine 启动后挂死 25+ min 无 log 输出, GPU 21 GB / 14% util.
怀疑 Ray actor 间 deadlock 或 sleep_mode wake 协议错配.

调试入口:
- `cat /home/ubuntu/r11_smoke_master.log | tail -100` 看最后输出
- `docker logs <container_id>` 看 docker stderr
- ssh 到 ray dashboard http://172.17.0.2:8265 看 actor state

修法选择:
- (a) 关 enable_sleep_mode (Ray actor 唤醒可能挂)
- (b) 关 vLLM V1 用 V0 (设 VLLM_USE_V1=0, 但 verl 又要 V1)
- (c) 升级 vllm 到 0.11+ (verl 期望区间)
- (d) 改 verl 的 launcher.py async wait timeout

### 之前: verl bucketed_weight_transfer IndexError ✓ 已修
smoke v7 走到 vLLM V1 engine init 后的 weight sync 时炸:
```
verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py:51
    list_args[6] = device_id
    IndexError: list assignment index out of range
```
原因: verl 假设 IPC tensor reduce arg list 至少 7 元素, 但我们的
torch 2.10 + vllm V1 实际只 < 7 元素。verl 与底层版本不匹配。

修法 (按风险递增):
- (a) 把 docker 内 vllm 升到 verl 期望区间 (0.10.2-0.12.0, 我们当前 0.10.1.dev)
- (b) checkout verl 到更老 commit  (`cd /home/ubuntu/v3/refs/verl && git checkout v0.5.x`)
- (c) patch verl 源码 line 51 为 `list_args.extend([None] * (7 - len(list_args))); list_args[6] = device_id`
- (d) 换 sglang 后端

### 5. ckpt 保存格式
verl 默认 FSDP shard 保存, **不是** HF LoRA 格式。我们 eval 脚本 (01_grpo_dev_eval.py) 期待 HF。

修法: 在 watcher 用 `convert_verl_ckpt_to_hf.py` 转格式 (已写, 但 SOURCE_LAYOUTS 可能要看实际 verl 输出补)。
或: 加 `actor_rollout_ref.actor.fsdp_config.save_in_hf_format=True` 之类 flag (需查 verl doc)。

## 关键文件位置

```
本地:
  /mnt/d/fine-tuning/v3/E5_grpo/r11_verl/
    ├── reward_judge.py             ← Gemini judge (verl async signature)
    ├── data_prep.py                ← jsonl → parquet (已跑过)
    ├── run_dapo_r11.sh             ← launcher (最新含 6 处 fix)
    ├── README.md                   ← 设计文档
    └── HANDOVER.md                 ← 本文件
  /mnt/d/fine-tuning/scripts_local/
    ├── local_watcher_dapo_r11.sh   ← S3 sync + eval (含 convert)
    ├── convert_verl_ckpt_to_hf.py  ← verl ckpt → HF LoRA
    └── update_excel_dapo_r11.py    ← excel summary
  /tmp/
    ├── ssm_smoke_v2.json           ← smoke launcher
    ├── ssm_check_r11.json          ← health check
    ├── r11_smoke_wrapper.sh        ← env wrapper for smoke
    └── build_verl_inside.sh        ← image build script

AWS (i-02cb096896c1aada8, IP 100.27.33.70 SSH 端口 22 防火墙超时, 用 SSM):
  /home/ubuntu/
    ├── run_dapo_r11.sh             ← 最新版从 S3 拉
    ├── r11_smoke_wrapper.sh
    ├── build_verl_inside.sh
    └── v3/refs/verl/               ← clone, recipe submodule 已 init
  docker images:
    vllm-trl:v3-r11-verl 35.5GB     ← verl + ray + trl 0.18

S3 prefix: s3://kingjameschan-fine-tuning-v3/
  scripts/run_dapo_r11.sh
  scripts/r11_reward_judge.py
  scripts/r11_smoke_wrapper.sh
  scripts/build_verl_inside.sh
  r11_data/{train,dev,test}.parquet
```

## 重启 smoke 命令 (粘贴即跑)

```bash
# 上传最新 launcher (本地若改过)
aws s3 cp /mnt/d/fine-tuning/v3/E5_grpo/r11_verl/run_dapo_r11.sh \
  s3://kingjameschan-fine-tuning-v3/scripts/run_dapo_r11.sh --quiet

# 触发 smoke
aws ssm send-command --instance-ids i-02cb096896c1aada8 \
  --document-name "AWS-RunShellScript" \
  --parameters file:///tmp/ssm_smoke_v2.json --region us-east-1 \
  --timeout-seconds 60

# ~50s 后查状态
aws ssm send-command --instance-ids i-02cb096896c1aada8 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader", "tail -50 /home/ubuntu/r11_smoke_master.log"]' \
  --region us-east-1
```

## 完成标准 (smoke pass criteria)

```
✓ master.log 无 Traceback / OutOfMemory / AssertionError
✓ 出现 "step 1/" 或 verl 等价的 metric log
✓ ckpt 出现在 /home/ubuntu/v3/E5_grpo/checkpoints/baseit_r11_verl_dapo_full_15ep/
✓ GPU peak < 44 GB (单 GPU 47 GB total, 留 3 GB margin)
```

## 切换 fallback 策略

如果 verl 反复踩坑:
- **方案 A**: 留下 R11 verl 代码作为面试 portfolio, 不实际跑 (代码本身展示理解)
- **方案 B**: 把 dynamic sampling 在 TRL 0.29 里手写 subclass (见 README "为啥 R10 不能加 DS")
- **方案 C**: 跑 R10 finished training 的剩余 ckpt (5/60 已存) 看趋势, 不 R11

#!/usr/bin/env bash
# v3 E5 R11 — DAPO 6/6 件套 via verl framework (vs R10 TRL 4/6)
#
# Diff vs R10:
#   ✓ Dynamic Sampling          (algorithm.filter_groups.enable=True)   ← 新增
#   ✓ Soft Overlong Punishment  (reward_model.overlong_buffer.enable=True) ← 新增
#   ✓ Token-level Loss          (loss_agg_mode=token-mean)
#   ✓ Clip-Higher real          (μ=2 让 ratio≠1 触发)
#   ✓ Overlong Filtering        (max_response_length cap)
#   ✓ Naive GRPO base
set -uo pipefail

# ============ Paths ============
V3_HOST=/home/ubuntu/v3
MODELS_HOST=/home/ubuntu/models
RUN_TAG=baseit_r11_verl_dapo_full_15ep
CKPT_BASE=$V3_HOST/E5_grpo/checkpoints/$RUN_TAG
LOG_DIR=$V3_HOST/E5_grpo/outputs/${RUN_TAG}_logs
mkdir -p "$CKPT_BASE" "$LOG_DIR"

DATA_DIR=/workspace/v3/E5_grpo/r11_verl/data
TRAIN_FILE=$DATA_DIR/train.parquet
TEST_FILE=$DATA_DIR/test.parquet
REWARD_FN=/workspace/v3/E5_grpo/r11_verl/reward_judge.py

MODEL_PATH=/workspace/models/gemma-2-2b-it
S3_PREFIX=s3://kingjameschan-fine-tuning-v3/$RUN_TAG

# ============ Hyperparams (same as R10 for fair ablation) ============
LR=1e-5
KL_COEF=0.0                # β=0 like R10
CLIP_LOW=0.2
CLIP_HIGH=0.28
GROUP_SIZE=8                # G
NUM_ITER=1                  # ★ DAPO paper 默认 μ=1 (verl async vLLM 自带 IS drift, clip 已能触发)
PER_DEVICE=8                # ★ R10 是 16, 但 verl + vllm colocate 单卡内存紧, 降到 8
ACCUM=48                    # 保 mini-batch=384 (8 × 48)
MAX_PROMPT_LEN=256
MAX_RESPONSE_LEN=384        # 与 R10 一致
WARMUP=7
TOTAL_STEPS=${TOTAL_STEPS:-240}     # full 跑 (smoke override 用 TOTAL_STEPS=5)
SAVE_FREQ=${SAVE_FREQ:-4}           # 每 4 step 存一次, 60 ckpts

# Dynamic sampling (★ R10 没有, R11 启用)
ENABLE_FILTER_GROUPS=True
FILTER_METRIC=acc           # 按 reward 准确率筛, 0/1 二元
MAX_NUM_GEN_BATCHES=10      # over-sample 最多 10× 直到 batch 填满

# Soft Overlong Punishment (★ R10 没有, R11 启用)
ENABLE_OVERLONG_BUFFER=True
OVERLONG_BUFFER_LEN=64      # 在 max-64 = 320 token 开始 soft penalty
OVERLONG_PENALTY=1.0        # 每超 1 token 扣 0.01 (verl 默认 1.0/buffer_len)

t_start=$(date +%s)
echo "============================================================="
echo "R11: verl DAPO 6/6 — vs R10 TRL 4/6"
echo "  + Dynamic Sampling     (filter_groups.enable=True, max=${MAX_NUM_GEN_BATCHES}×)"
echo "  + Soft Overlong        (buffer_len=${OVERLONG_BUFFER_LEN}, penalty=${OVERLONG_PENALTY})"
echo "  + μ=2 让 Clip-Higher 真触发 (R10 μ=1 死 flag)"
echo "  per_device=$PER_DEVICE × accum=$ACCUM × G=$GROUP_SIZE"
echo "  loss_type=token-mean  init=base IT  lr=$LR β=0"
echo "============================================================="

# ============ GPU peak monitor ============
pkill -9 -f gpu_peak_mon 2>/dev/null || true
cat > /tmp/gpu_peak_mon.sh << 'EOF'
#!/bin/bash
PEAK=0; PEAK_TIME=''; LOG=/tmp/gpu_peak_mon.log
echo "start $(date +%T)" > $LOG
while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    [ -z "$used" ] && continue
    if [ "$used" -gt "$PEAK" ]; then
        PEAK=$used; PEAK_TIME=$(date +%T)
        echo "NEW_PEAK $PEAK_TIME used=${used}MiB" >> $LOG
    fi
    if [ $((SECONDS % 5)) -eq 0 ]; then
        echo "STATUS $(date +%T) cur=${used}MiB peak=${PEAK}MiB(${PEAK_TIME})" >> $LOG
    fi
    sleep 0.2
done
EOF
chmod +x /tmp/gpu_peak_mon.sh
nohup bash /tmp/gpu_peak_mon.sh > /dev/null 2>&1 < /dev/null &
disown 2>/dev/null || true
echo "[bg] gpu peak monitor restarted"

# ============ S3 sync ============
nohup bash -c "
START=\$(date +%s)
while [ \$((\$(date +%s) - START)) -lt 86400 ]; do
    aws s3 sync $CKPT_BASE $S3_PREFIX/checkpoints/ --quiet 2>/dev/null
    aws s3 sync $LOG_DIR $S3_PREFIX/logs/ --quiet 2>/dev/null
    aws s3 cp /home/ubuntu/${RUN_TAG}_master.log $S3_PREFIX/logs/master.log --quiet 2>/dev/null || true
    sleep 60
done
" > /home/ubuntu/incr_sync_${RUN_TAG}.log 2>&1 < /dev/null &
INCR_SYNC_PID=$!
echo "[bg] incr sync pid=$INCR_SYNC_PID"

# ============ verl DAPO recipe (inside docker) ============
# verl 装在 vllm-trl:v3-r11-verl 镜像里，invoke 走 docker run.
# Ray 由 verl 内部以 driver 模式启动，无需手动 ray start.
DOCKER_IMAGE=vllm-trl:v3-r11-verl-vllm102

docker run --rm --gpus all --shm-size=16g \
    -e "GOOGLE_API_KEY=$GOOGLE_API_KEY" \
    -e "PYTHONUNBUFFERED=1" \
    -e "RAY_DISABLE_DOCKER_CPU_WARNING=1" \
    -e "VLLM_USE_V1=1" \
    -e "VLLM_LOGGING_LEVEL=INFO" \
    -e "RAY_DEDUP_LOGS=0" \
    -v "$V3_HOST":/workspace/v3 \
    -v "$MODELS_HOST":/workspace/models \
    -v "$CKPT_BASE":/workspace/ckpt_out \
    -v "$V3_HOST/refs/verl/recipe":/opt/verl_pkg/recipe:ro \
    --workdir /opt/verl_pkg \
    --entrypoint python3 $DOCKER_IMAGE \
    -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=$ENABLE_FILTER_GROUPS \
    algorithm.filter_groups.metric=$FILTER_METRIC \
    algorithm.filter_groups.max_num_gen_batches=$MAX_NUM_GEN_BATCHES \
    \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.train_batch_size=$((PER_DEVICE * ACCUM)) \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$WARMUP \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_HIGH \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((PER_DEVICE * ACCUM)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PER_DEVICE \
    actor_rollout_ref.actor.ppo_epochs=$NUM_ITER \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.rollout.max_model_len=640 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PER_DEVICE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PER_DEVICE \
    \
    reward_model.reward_manager=naive \
    reward.reward_kwargs.overlong_buffer_cfg.enable=$ENABLE_OVERLONG_BUFFER \
    reward.reward_kwargs.overlong_buffer_cfg.len=$OVERLONG_BUFFER_LEN \
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=$OVERLONG_PENALTY \
    reward.reward_kwargs.max_resp_len=$MAX_RESPONSE_LEN \
    \
    custom_reward_function.path=$REWARD_FN \
    custom_reward_function.name=compute_score \
    \
    trainer.total_epochs=2 \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=-1 \
    trainer.default_local_dir=/workspace/ckpt_out \
    trainer.project_name=v3_grpo_r11 \
    trainer.experiment_name=$RUN_TAG \
    trainer.logger=[console] \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    2>&1 | tee -a "$LOG_DIR/train.log"
TRAIN_EXIT=${PIPESTATUS[0]}

t_train=$(($(date +%s) - t_start))
ray stop --force 2>/dev/null || true
kill $INCR_SYNC_PID 2>/dev/null || true
aws s3 sync $CKPT_BASE $S3_PREFIX/checkpoints/ --quiet 2>&1 | tail -3
aws s3 sync $LOG_DIR $S3_PREFIX/logs/ --quiet 2>&1 | tail -3

[ "$TRAIN_EXIT" -ne 0 ] && { echo "[fail] verl exit=$TRAIN_EXIT after ${t_train}s"; exit $TRAIN_EXIT; }
echo "${RUN_TAG}_DONE in $((t_train / 60)) min"

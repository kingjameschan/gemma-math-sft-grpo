#!/usr/bin/env bash
# v3 E5 R15 — = R14 + 3 changes:
#   1. ppo_mini_batch_size: 384 → 192     (★ gradient 频率 ×2, 同份 rollout 做 2 mini × μ=3 = 6 update/step)
#   2. max_response_length: 384 → 512     (★ 覆盖 R14 P95 ~409 token, overlong 12.8% → ~3-4%)
#   3. lr_warmup_steps:     3   → 10      (★ lr ramp 同步缩放, 防 lr=2e-5 + 6 update/step 早期 collapse)
#
# 不变: train_batch=384, G=8, μ=3, lr=2e-5, ε_low=0.2, ε_high=0.4,
#       per_device=12, gpu_mem=0.3, total_epochs=50, total_steps=240, save_freq=1
#       data=train_gsm8k_math_numerical.parquet (11875), reward=R15 frac-aware
set -uo pipefail

V3_HOST=/home/ubuntu/v3
MODELS_HOST=/home/ubuntu/models
RUN_TAG=baseit_r15_verl_dapo_full_15ep
CKPT_BASE=$V3_HOST/E5_grpo/checkpoints/$RUN_TAG
LOG_DIR=$V3_HOST/E5_grpo/outputs/${RUN_TAG}_logs
mkdir -p "$CKPT_BASE" "$LOG_DIR"

DATA_DIR=/workspace/v3/E5_grpo/r11_verl/data
TRAIN_FILE=$DATA_DIR/train_gsm8k_math_numerical.parquet
TEST_FILE=$DATA_DIR/test.parquet
REWARD_FN=/workspace/v3/E5_grpo/r15_dapo/reward_judge_r15.py
MODEL_PATH=/workspace/models/gemma-2-2b-it
S3_PREFIX=s3://kingjameschan-fine-tuning-v3/$RUN_TAG

: > /home/ubuntu/r15_reward_log.jsonl

LR=2e-5
KL_COEF=0.0
CLIP_LOW=0.2
CLIP_HIGH=0.40
GROUP_SIZE=8
NUM_ITER=3                          # μ
PER_DEVICE=12
ACCUM=32                            # = train_batch / per_device = 384/12
TRAIN_BATCH=$((PER_DEVICE * ACCUM)) # 384
PPO_MINI_BATCH=192                  # ★ R15: 384 → 192
MAX_PROMPT_LEN=192
MAX_RESPONSE_LEN=512                # ★ R15: 384 → 512
MAX_MODEL_LEN=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))  # 704
WARMUP=10                           # ★ R15: 3 → 10
WEIGHT_DECAY=0.0
TOTAL_STEPS=${TOTAL_STEPS:-240}
SAVE_FREQ=${SAVE_FREQ:-1}
TOTAL_EPOCHS=50

ENABLE_FILTER_GROUPS=True
FILTER_METRIC=acc
MAX_NUM_GEN_BATCHES=5

ENABLE_OVERLONG_BUFFER=True
OVERLONG_BUFFER_LEN=64
OVERLONG_PENALTY=1.0

t_start=$(date +%s)
echo "============================================================="
echo "R15: R14 + 3 changes (mini 384→192, max_resp 384→512, warmup 3→10)"
echo "  train_batch=$TRAIN_BATCH  mini=$PPO_MINI_BATCH  G=$GROUP_SIZE  μ=$NUM_ITER"
echo "  → updates/step = (train/mini)×μ = (384/192)×3 = 6  ← R14 是 3"
echo "  lr=$LR  warmup=$WARMUP  ε_low=$CLIP_LOW  ε_high=$CLIP_HIGH"
echo "  max_prompt=$MAX_PROMPT_LEN max_resp=$MAX_RESPONSE_LEN max_model=$MAX_MODEL_LEN"
echo "  data=$TRAIN_FILE  reward=$REWARD_FN"
echo "============================================================="

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

nohup bash -c "
START=\$(date +%s)
while [ \$((\$(date +%s) - START)) -lt 86400 ]; do
    aws s3 sync $CKPT_BASE $S3_PREFIX/checkpoints/ --quiet 2>/dev/null
    aws s3 sync $LOG_DIR $S3_PREFIX/logs/ --quiet 2>/dev/null
    aws s3 cp /home/ubuntu/r15_full_master.log $S3_PREFIX/logs/master.log --quiet 2>/dev/null || true
    aws s3 cp /home/ubuntu/r15_reward_log.jsonl $S3_PREFIX/logs/reward_log.jsonl --quiet 2>/dev/null || true
    sleep 60
done
" > /home/ubuntu/incr_sync_${RUN_TAG}.log 2>&1 < /dev/null &
INCR_SYNC_PID=$!

DOCKER_IMAGE=vllm-trl:v3-r11-verl-vllm102

docker run --rm --gpus all --shm-size=16g \
    -e "PYTHONUNBUFFERED=1" \
    -e "RAY_DISABLE_DOCKER_CPU_WARNING=1" \
    -e "VLLM_USE_V1=1" \
    -e "VLLM_LOGGING_LEVEL=INFO" \
    -e "RAY_DEDUP_LOGS=0" \
    -e "V3_SHARED=/workspace/v3/shared" \
    -e "R15_REWARD_LOG=/workspace/r15_reward_log.jsonl" \
    -v "$V3_HOST":/workspace/v3 \
    -v "$MODELS_HOST":/workspace/models \
    -v "$CKPT_BASE":/workspace/ckpt_out \
    -v "/home/ubuntu/r15_reward_log.jsonl":/workspace/r15_reward_log.jsonl \
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
    data.train_batch_size=$TRAIN_BATCH \
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
    actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_HIGH \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PER_DEVICE \
    actor_rollout_ref.actor.ppo_epochs=$NUM_ITER \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
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
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.default_local_dir=/workspace/ckpt_out \
    trainer.project_name=v3_grpo_r15 \
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
aws s3 cp /home/ubuntu/r15_reward_log.jsonl $S3_PREFIX/logs/reward_log.jsonl --quiet || true

[ "$TRAIN_EXIT" -ne 0 ] && { echo "[fail] verl exit=$TRAIN_EXIT after ${t_train}s"; exit $TRAIN_EXIT; }
echo "${RUN_TAG}_DONE in $((t_train / 60)) min"

#!/usr/bin/env bash
# v3 E5 R14 — = R13 retry-C config + 5 changes:
#   1. lr: 1e-5 → 2e-5  (Schulman LoRA-RL: 5-15× FullFT lr; 2× is conservative bump)
#   2. data: GSM8K (6973) → GSM8K + MATH-numerical (11875), DAPO DS auto-filters Lv4-5
#   3. reward: math_equal_numerical → math_equal_v2 (R14 = R13 + \frac/dfrac/simple-frac canonical)
#   4. ppo_epochs μ: 1 → 3   (R13 全程 clipfrac=0% = PPO 没工作 → 增 μ 让 policy 漂移)
#   5. clip_high ε⁺: 0.28 → 0.40 (DAPO Clip-Higher, μ=3 配套, 防止过度 clip)
#   + total_epochs: 2 → 50 (R13 epoch cap incident fix)
#
# Other RL config IDENTICAL to R13:
#   per_dev=12, accum=32, mini_batch=384, G=8
#   gpu_mem=0.3, max_prompt=192, max_resp=384
#   warmup=3, β=0
#   clip_low ε⁻=0.2, loss_agg=token-mean
#   DS enable, save_freq=1, val_before_train=False, wd=0
set -uo pipefail

V3_HOST=/home/ubuntu/v3
MODELS_HOST=/home/ubuntu/models
RUN_TAG=baseit_r14_verl_dapo_full_15ep
CKPT_BASE=$V3_HOST/E5_grpo/checkpoints/$RUN_TAG
LOG_DIR=$V3_HOST/E5_grpo/outputs/${RUN_TAG}_logs
mkdir -p "$CKPT_BASE" "$LOG_DIR"

DATA_DIR=/workspace/v3/E5_grpo/r11_verl/data
TRAIN_FILE=$DATA_DIR/train_gsm8k_math_numerical.parquet  # ★ R14: GSM8K+MATH 11875
TEST_FILE=$DATA_DIR/test.parquet                          # 仍 GSM8K test
REWARD_FN=/workspace/v3/E5_grpo/r14_dapo/reward_judge_r14.py  # ★ R14 reward
MODEL_PATH=/workspace/models/gemma-2-2b-it
S3_PREFIX=s3://kingjameschan-fine-tuning-v3/$RUN_TAG

# Reset reward log (fresh per launch)
: > /home/ubuntu/r14_reward_log.jsonl

LR=2e-5                # ★ R14: 1e-5 → 2e-5
KL_COEF=0.0
CLIP_LOW=0.2
CLIP_HIGH=0.40         # ★ R14: 0.28 → 0.40 (Clip-Higher, 配套 μ=3)
GROUP_SIZE=8
NUM_ITER=3             # ★ R14: μ=1 → μ=3 (R13 clipfrac=0% 修复)
PER_DEVICE=12
ACCUM=32
MAX_PROMPT_LEN=192
MAX_RESPONSE_LEN=384
WARMUP=3
WEIGHT_DECAY=0.0
TOTAL_STEPS=${TOTAL_STEPS:-240}
SAVE_FREQ=${SAVE_FREQ:-1}
TOTAL_EPOCHS=50        # ★ R14: 2 → 50 (R13 epoch cap fix)

ENABLE_FILTER_GROUPS=True
FILTER_METRIC=acc
MAX_NUM_GEN_BATCHES=5

ENABLE_OVERLONG_BUFFER=True
OVERLONG_BUFFER_LEN=64
OVERLONG_PENALTY=1.0

t_start=$(date +%s)
echo "============================================================="
echo "R14: R13 + 5 changes (lr 2e-5, +MATH, frac-aware reward, μ=3, ε_high=0.4)"
echo "  per_device=$PER_DEVICE × accum=$ACCUM × G=$GROUP_SIZE   mini=$((PER_DEVICE * ACCUM))"
echo "  lr=$LR  warmup=$WARMUP  total_epochs=$TOTAL_EPOCHS  save_freq=$SAVE_FREQ"
echo "  μ=$NUM_ITER  ε_low=$CLIP_LOW  ε_high=$CLIP_HIGH (Clip-Higher)"
echo "  data=$TRAIN_FILE"
echo "  reward=$REWARD_FN"
echo "  log per-call → /home/ubuntu/r14_reward_log.jsonl"
echo "============================================================="

# GPU peak monitor
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

# S3 sync
nohup bash -c "
START=\$(date +%s)
while [ \$((\$(date +%s) - START)) -lt 86400 ]; do
    aws s3 sync $CKPT_BASE $S3_PREFIX/checkpoints/ --quiet 2>/dev/null
    aws s3 sync $LOG_DIR $S3_PREFIX/logs/ --quiet 2>/dev/null
    aws s3 cp /home/ubuntu/r14_full_master.log $S3_PREFIX/logs/master.log --quiet 2>/dev/null || true
    aws s3 cp /home/ubuntu/r14_reward_log.jsonl $S3_PREFIX/logs/reward_log.jsonl --quiet 2>/dev/null || true
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
    -e "R14_REWARD_LOG=/workspace/r14_reward_log.jsonl" \
    -v "$V3_HOST":/workspace/v3 \
    -v "$MODELS_HOST":/workspace/models \
    -v "$CKPT_BASE":/workspace/ckpt_out \
    -v "/home/ubuntu/r14_reward_log.jsonl":/workspace/r14_reward_log.jsonl \
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
    actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.rollout.max_model_len=576 \
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
    trainer.project_name=v3_grpo_r14 \
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
aws s3 cp /home/ubuntu/r14_reward_log.jsonl $S3_PREFIX/logs/reward_log.jsonl --quiet || true

[ "$TRAIN_EXIT" -ne 0 ] && { echo "[fail] verl exit=$TRAIN_EXIT after ${t_train}s"; exit $TRAIN_EXIT; }
echo "${RUN_TAG}_DONE in $((t_train / 60)) min"

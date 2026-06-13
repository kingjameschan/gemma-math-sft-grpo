#!/usr/bin/env bash
# v3 E5 R17 — DAPO ~10-effective-epoch rerun (R15 复刻 + 跑长)
# 4 项 DAPO additions 全开 (复刻 R15); 算法/超参完全等同 R15.
# 仅改 3 处:
#   · TOTAL_STEPS 240 → 100 (R15 实际只跑 23 步因 spot 抢占)
#   · Pruner: 非 latest 2 fullstate 剥 LoRA-only
#   · Infra: AWS L40S → Aliyun L20 (脚本基底用 R16 Aliyun 模板)
set -uo pipefail

V3_HOST=/root/v3
MODELS_HOST=/root/models
RUN_TAG=baseit_r17_dapo_100step
CKPT_BASE=$V3_HOST/E5_grpo/checkpoints/$RUN_TAG
LOG_DIR=$V3_HOST/E5_grpo/outputs/${RUN_TAG}_logs
mkdir -p "$CKPT_BASE" "$LOG_DIR"

DATA_DIR=/workspace/v3/E5_grpo/r11_verl/data
TRAIN_FILE=$DATA_DIR/train_gsm8k_math_numerical.parquet
TEST_FILE=$DATA_DIR/dev.parquet
REWARD_FN=/workspace/v3/E5_grpo/r17_dapo_100step/reward_judge.py
MODEL_PATH=/workspace/models/gemma-2-2b-it

# Hyperparams — IDENTICAL to R15
LR=2e-5
KL_COEF=0.0
CLIP_LOW=0.20
CLIP_HIGH=0.40                        # ★ DAPO #2 Clip-Higher ON (= R15)
GROUP_SIZE=8
NUM_ITER=3                            # μ (PPO inner epochs)
PER_DEVICE=12
ACCUM=32
TRAIN_BATCH=$((PER_DEVICE * ACCUM))   # 384
PPO_MINI_BATCH=192
MAX_PROMPT_LEN=192
MAX_RESPONSE_LEN=512
MAX_MODEL_LEN=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))  # 704
WARMUP=10
WEIGHT_DECAY=0.0

ENABLE_FILTER_GROUPS=True             # ★ DAPO #1 Dynamic Sampling ON (= R15)
FILTER_METRIC=acc
MAX_NUM_GEN_BATCHES=5

ENABLE_OVERLONG_BUFFER=True           # ★ DAPO #3 Overlong Buffer ON (= R15)
OVERLONG_BUFFER_LEN=64
OVERLONG_PENALTY=1.0

TOTAL_STEPS=${TOTAL_STEPS:-100}       # ★ R15 cap 240 (实跑 23), R17 目标 100
SAVE_FREQ=${SAVE_FREQ:-1}             # 每步存 (= R15); pruner 守护非 latest 2 只留 LoRA
TOTAL_EPOCHS=50                       # ceiling, 100 step 先 hit
TEST_FREQ=${TEST_FREQ:--1}            # 训练时不做 val (=R15, 训完 sweep)
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.45}    # L20 KV cache
RESUME_MODE=${RESUME_MODE:-auto}

DOCKER_IMAGE=${DOCKER_IMAGE:-vllm-trl:v3-r16-grpo}

t_start=$(date +%s)
echo "================================================================"
echo "R17 DAPO ~10-effective-epoch rerun — 复刻 R15 + 跑长到 100 step"
echo "================================================================"
echo "  algo:         GRPO (adv_estimator=grpo) with 4 DAPO additions ON"
echo "  clip ε_low:   $CLIP_LOW"
echo "  clip ε_high:  $CLIP_HIGH         ★ DAPO #2 Clip-Higher"
echo "  filter_groups: $ENABLE_FILTER_GROUPS  ★ DAPO #1 Dynamic Sampling"
echo "  overlong_buf: $ENABLE_OVERLONG_BUFFER  ★ DAPO #3 Overlong Buffer"
echo "  loss_agg:     token-mean  ★ DAPO #4"
echo ""
echo "  train_batch:  $TRAIN_BATCH (mini=$PPO_MINI_BATCH × μ=$NUM_ITER × G=$GROUP_SIZE)"
echo "  per_device:   $PER_DEVICE × accum $ACCUM"
echo "  lr:           $LR  warmup=$WARMUP"
echo "  max_prompt:   $MAX_PROMPT_LEN  max_resp: $MAX_RESPONSE_LEN  max_model: $MAX_MODEL_LEN"
echo "  data:         $TRAIN_FILE"
echo "  dev:          $TEST_FILE"
echo "  reward:       $REWARD_FN"
echo "  model:        $MODEL_PATH (Gemma2-2B-IT)"
echo "  LoRA:         r=64, alpha=32, all-linear, dropout=0"
echo "  total_epochs: $TOTAL_EPOCHS ceiling, max_steps: $TOTAL_STEPS (= ~10 effective epoch)"
echo "  save_freq:    every $SAVE_FREQ steps (~$TOTAL_STEPS ckpts, pruner 守护 latest 2 fullstate)"
echo "  docker:       $DOCKER_IMAGE"
echo "================================================================"

docker run --rm --gpus all --shm-size=16g \
    -e "PYTHONUNBUFFERED=1" \
    -e "RAY_DISABLE_DOCKER_CPU_WARNING=1" \
    -e "VLLM_USE_V1=1" \
    -e "VLLM_LOGGING_LEVEL=INFO" \
    -e "RAY_DEDUP_LOGS=0" \
    -v "$V3_HOST":/workspace/v3 \
    -v "$MODELS_HOST":/workspace/models \
    -v "$CKPT_BASE":/workspace/ckpt_out \
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
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
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
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=$ENABLE_OVERLONG_BUFFER \
    reward_model.overlong_buffer.len=$OVERLONG_BUFFER_LEN \
    reward_model.overlong_buffer.penalty_factor=$OVERLONG_PENALTY \
    \
    custom_reward_function.path=$REWARD_FN \
    custom_reward_function.name=compute_score \
    \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.resume_mode=$RESUME_MODE \
    trainer.default_local_dir=/workspace/ckpt_out \
    trainer.project_name=v3_dapo_r17 \
    trainer.experiment_name=$RUN_TAG \
    trainer.logger=[console] \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    2>&1 | tee -a "$LOG_DIR/train.log"
TRAIN_EXIT=${PIPESTATUS[0]}

t_train=$(($(date +%s) - t_start))
echo ""
echo "================================================================"
echo "R17 DAPO finished. exit=$TRAIN_EXIT  duration=$(($t_train / 60)) min"
echo "================================================================"
[ "$TRAIN_EXIT" -ne 0 ] && exit $TRAIN_EXIT

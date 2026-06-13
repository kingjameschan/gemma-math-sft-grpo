#!/usr/bin/env bash
# v3 E5 R16 — Clean GRPO ablation = R15 DAPO − 4 项 DAPO additions
#
# 4 项 DAPO additions disabled (vs R15):
#   1. Clip-Higher:     CLIP_HIGH=0.40 → 0.20  (symmetric clip)
#   2. Dynamic Sampling: filter_groups=True → False
#   3. Overlong Buffer:  overlong_buffer=True → False
#   4. Token-level Loss: loss_agg_mode=token-mean → seq-mean-token-mean
#
# Everything else identical to R15:
#   train_batch=384, mini=192, μ=3, G=8, lr=2e-5, warmup=10
#   max_prompt=192, max_resp=512, lora r=64/alpha=32
#   adv_estimator=grpo, KL_coef=0, use_kl_loss=False
#
# B option (per discussion 5/13): NO KL (跟 R15 一致)
# 想测 vanilla GRPO+KL 的话另开 R17.
set -uo pipefail

V3_HOST=/root/v3
MODELS_HOST=/root/models
RUN_TAG=baseit_r16_clean_grpo
CKPT_BASE=$V3_HOST/E5_grpo/checkpoints/$RUN_TAG
LOG_DIR=$V3_HOST/E5_grpo/outputs/${RUN_TAG}_logs
mkdir -p "$CKPT_BASE" "$LOG_DIR"

DATA_DIR=/workspace/v3/E5_grpo/r11_verl/data
TRAIN_FILE=$DATA_DIR/train_gsm8k_math_numerical.parquet
TEST_FILE=$DATA_DIR/dev.parquet
REWARD_FN=/workspace/v3/E5_grpo/r16_grpo_clean/reward_judge.py
MODEL_PATH=/workspace/models/gemma-2-2b-it

# Hyperparams (= R15 baseline)
LR=2e-5
KL_COEF=0.0
CLIP_LOW=0.20
CLIP_HIGH=0.20                       # ★ DAPO #2 OFF (R15: 0.40)
GROUP_SIZE=8
NUM_ITER=3                            # μ (PPO inner epochs, R15 same)
PER_DEVICE=12
ACCUM=32
TRAIN_BATCH=$((PER_DEVICE * ACCUM))   # 384
PPO_MINI_BATCH=192                    # R15 same
MAX_PROMPT_LEN=192
MAX_RESPONSE_LEN=512                  # R15 same
MAX_MODEL_LEN=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))  # 704
WARMUP=10
WEIGHT_DECAY=0.0

ENABLE_FILTER_GROUPS=False            # ★ DAPO #1 OFF (R15: True)
FILTER_METRIC=acc
MAX_NUM_GEN_BATCHES=5

ENABLE_OVERLONG_BUFFER=False          # ★ DAPO #3 OFF (R15: True)
OVERLONG_BUFFER_LEN=64
OVERLONG_PENALTY=1.0

TOTAL_STEPS=${TOTAL_STEPS:-240}
SAVE_FREQ=${SAVE_FREQ:-2}             # ★ disk 限制: 60 step / 2 = 30 ckpts. pruner 守护非 latest 只留 LoRA
TOTAL_EPOCHS=2                        # bottleneck (filter_groups=False, 60 step total)
TEST_FREQ=${TEST_FREQ:--1}            # disable in-train val (跟 R15 一致)
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.45}    # vLLM KV cache (>R15 0.30, infra knob 不破 ablation)
RESUME_MODE=${RESUME_MODE:-auto}      # auto-resume from latest ckpt in default_local_dir

DOCKER_IMAGE=${DOCKER_IMAGE:-vllm-trl:v3-r16-grpo}

t_start=$(date +%s)
echo "================================================================"
echo "R16 Clean GRPO Ablation — 4 项 DAPO additions OFF, else = R15"
echo "================================================================"
echo "  algo:         GRPO (adv_estimator=grpo)"
echo "  clip ε_low:   $CLIP_LOW"
echo "  clip ε_high:  $CLIP_HIGH         ★ DAPO #2 OFF"
echo "  filter_groups: $ENABLE_FILTER_GROUPS  ★ DAPO #1 OFF"
echo "  overlong_buf: $ENABLE_OVERLONG_BUFFER  ★ DAPO #3 OFF"
echo "  loss_agg:     seq-mean-token-mean  ★ DAPO #4 OFF"
echo ""
echo "  train_batch:  $TRAIN_BATCH (mini=$PPO_MINI_BATCH × μ=$NUM_ITER × G=$GROUP_SIZE)"
echo "  per_device:   $PER_DEVICE × accum $ACCUM"
echo "  lr:           $LR  warmup=$WARMUP"
echo "  max_prompt:   $MAX_PROMPT_LEN  max_resp: $MAX_RESPONSE_LEN  max_model: $MAX_MODEL_LEN"
echo "  data:         $TRAIN_FILE"
echo "  dev:          $TEST_FILE (500 GSM8K, dev_eval)"
echo "  reward:       $REWARD_FN"
echo "  model:        $MODEL_PATH (Gemma2-2B-IT)"
echo "  LoRA:         r=64, alpha=32, all-linear, dropout=0"
echo "  total_epochs: $TOTAL_EPOCHS (bottleneck), max_steps: $TOTAL_STEPS"
echo "  save_freq:    every $SAVE_FREQ steps (~$(($TOTAL_STEPS / $SAVE_FREQ)) ckpts)"
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
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
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
    trainer.project_name=v3_grpo_r16 \
    trainer.experiment_name=$RUN_TAG \
    trainer.logger=[console] \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    2>&1 | tee -a "$LOG_DIR/train.log"
TRAIN_EXIT=${PIPESTATUS[0]}

t_train=$(($(date +%s) - t_start))
echo ""
echo "================================================================"
echo "R16 GRPO finished. exit=$TRAIN_EXIT  duration=$(($t_train / 60)) min"
echo "================================================================"
[ "$TRAIN_EXIT" -ne 0 ] && exit $TRAIN_EXIT

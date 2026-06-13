#!/bin/bash
# R15 host LoRA hook = R14 hook with RUN_TAG bumped to R15
set -uo pipefail
RUN_TAG=baseit_r15_verl_dapo_full_15ep
CKPT_HOST=/home/ubuntu/v3/E5_grpo/checkpoints/$RUN_TAG
LORA_HOST=/home/ubuntu/v3/E5_grpo/checkpoints/${RUN_TAG}_lora
S3_BASE=s3://kingjameschan-fine-tuning-v3/$RUN_TAG
S3_LORA=$S3_BASE/lora_only
DOCKER_IMG=vllm-trl:v3-r11-verl-vllm102
LOG=/home/ubuntu/r15_lora_hook.log
CONVERTER=/home/ubuntu/convert_verl_to_peft.py
mkdir -p $LORA_HOST
echo "[$(date -u)] hook started (R15)" >> $LOG
while true; do
    steps=$(ls -d $CKPT_HOST/global_step_* 2>/dev/null | grep -oE '[0-9]+$' | sort -n)
    [ -z "$steps" ] && { sleep 60; continue; }
    latest=$(echo "$steps" | tail -1)
    for s in $steps; do
        SRC=$CKPT_HOST/global_step_$s; DST=$LORA_HOST/global_step_$s
        SRC_PT=$SRC/actor/model_world_size_1_rank_0.pt
        if [ -f "$DST/adapter_model.safetensors" ]; then
            if [ "$s" != "$latest" ] && [ -f "$SRC_PT" ]; then
                echo "[$(date -u)] cleanup heavy step $s" >> $LOG
                rm -f $SRC/actor/model_world_size_*.pt $SRC/actor/optim_world_size_*.pt $SRC/actor/extra_state_world_size_*.pt
                aws s3 rm $S3_BASE/checkpoints/global_step_$s/actor/model_world_size_1_rank_0.pt --quiet 2>/dev/null
                aws s3 rm $S3_BASE/checkpoints/global_step_$s/actor/optim_world_size_1_rank_0.pt --quiet 2>/dev/null
                aws s3 rm $S3_BASE/checkpoints/global_step_$s/actor/extra_state_world_size_1_rank_0.pt --quiet 2>/dev/null
            fi
            continue
        fi
        [ ! -f "$SRC_PT" ] && continue
        ago=$(($(date +%s) - $(stat -c %Y $SRC_PT)))
        [ "$ago" -lt 60 ] && continue
        echo "[$(date -u)] extracting LoRA step $s" >> $LOG
        docker run --rm --shm-size=2g \
          -v /home/ubuntu/v3:/workspace/v3 \
          -v /home/ubuntu/models:/workspace/models \
          -v $CKPT_HOST:/workspace/ckpt_in \
          -v $LORA_HOST:/workspace/ckpt_out \
          -v $CONVERTER:/workspace/convert.py:ro \
          --workdir /workspace --entrypoint python3 $DOCKER_IMG \
          /workspace/convert.py --ckpt /workspace/ckpt_in/global_step_$s --out /workspace/ckpt_out/global_step_$s --base-model /workspace/models/gemma-2-2b-it >> $LOG 2>&1
        if [ -f "$DST/adapter_model.safetensors" ]; then
            aws s3 sync $DST $S3_LORA/global_step_$s/ --quiet
            if [ "$s" != "$latest" ]; then
                rm -f $SRC/actor/model_world_size_*.pt $SRC/actor/optim_world_size_*.pt $SRC/actor/extra_state_world_size_*.pt
                aws s3 rm $S3_BASE/checkpoints/global_step_$s/actor/model_world_size_1_rank_0.pt --quiet 2>/dev/null
                aws s3 rm $S3_BASE/checkpoints/global_step_$s/actor/optim_world_size_1_rank_0.pt --quiet 2>/dev/null
                aws s3 rm $S3_BASE/checkpoints/global_step_$s/actor/extra_state_world_size_1_rank_0.pt --quiet 2>/dev/null
                echo "[$(date -u)] step $s done (LoRA up + heavy purged)" >> $LOG
            else
                echo "[$(date -u)] step $s done (LoRA up, heavy KEPT)" >> $LOG
            fi
        else
            echo "[$(date -u)] CONVERT FAILED step $s" >> $LOG
        fi
    done
    sleep 60
done

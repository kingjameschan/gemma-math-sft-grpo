#!/usr/bin/env bash
# R17 ckpt pruner: keep latest 2 fullstate, strip rest to LoRA-only.
# Runs every 60s as nohup background. Reads CKPT_DIR from env or defaults.
CKPT_DIR=${CKPT_DIR:-/root/v3/E5_grpo/checkpoints/baseit_r17_dapo_100step}
KEEP_LATEST=${KEEP_LATEST:-2}
INTERVAL=${INTERVAL:-60}

while true; do
  if [ -d "$CKPT_DIR" ]; then
    # global_step_N dirs sorted by N descending; latest KEEP_LATEST kept full
    mapfile -t steps < <(ls -d $CKPT_DIR/global_step_* 2>/dev/null | \
                         awk -F'global_step_' '{print $2}' | sort -rn)
    if [ ${#steps[@]} -gt $KEEP_LATEST ]; then
      keep=()
      for i in $(seq 0 $((KEEP_LATEST - 1))); do
        keep+=("${steps[$i]}")
      done
      for s in "${steps[@]:$KEEP_LATEST}"; do
        d=$CKPT_DIR/global_step_$s
        if [ -e "$d/actor/optim_world_size_1_rank_0.pt" ]; then
          rm -rf "$d/actor/optim"*.pt "$d/actor/extra_state"*.pt 2>/dev/null
          rm -f "$d/actor/model_world_size_"*.pt 2>/dev/null
          find "$d/actor" -type f -name 'rng_state*' -delete 2>/dev/null
          echo "$(date +%T) pruned step_$s (kept LoRA only)" >> /root/ckpt_pruner.log
        fi
      done
    fi
  fi
  sleep $INTERVAL
done

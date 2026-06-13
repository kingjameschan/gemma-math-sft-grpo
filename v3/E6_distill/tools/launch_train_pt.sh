#!/bin/bash
# Single source of truth for E7 pretrain-base SFT launch.
# Pass --resume to continue from the latest checkpoint-* (IDENTICAL hyperparams —
# never change batch/accum/lr on resume). Logs are APPENDED so resume keeps history.
cd /mnt/d/fine-tuning || exit 1
RESUME=""
[ "$1" = "--resume" ] && RESUME="--resume"
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /mnt/d/miniconda/envs/siren/python.exe v3/E6_distill/tools/local_train.py \
  --model 'D:\fine-tuning\models\gemma-2-2b' \
  --train_file 'D:\fine-tuning\v3\E6_distill\data\distill_train_100k.jsonl' \
  --output_dir 'D:\fine-tuning\v3\E6_distill\checkpoints_pt' \
  --ep 2 --batch_size 2 --accum 8 --max_length 768 --opt adamw_8bit \
  --lr 5e-4 --save_steps 400 --logging_steps 20 $RESUME \
  >> v3/E6_distill/logs/train_pt_100k.log 2>&1 &
echo "launched train pid=$! resume=${RESUME:-no}"

#!/bin/bash
# E6 distill eval chain: 2 test sets × 2 seeds K=64, sequential (GPU serial).
CKPT=/mnt/d/fine-tuning/v3/E6_distill/checkpoints/final
OUT=/mnt/d/fine-tuning/v3/E6_distill/outputs
EVAL=/mnt/d/fine-tuning/v3/E1_baseline/eval/03_eval_pass_at_k.py
PY=/home/kingjames/vllm-env/bin/python
LOG=/mnt/d/fine-tuning/v3/E6_distill/logs

run() {
  name=$1; tf=$2; seed=$3
  echo "===== EVAL $name seed=$seed ====="
  VLLM_USE_V1=0 $PY $EVAL --ckpt $CKPT --test_file "$tf" \
    --k 64 --chunk_size 50 --max_lora_rank 64 --max_model_len 1408 --seed $seed \
    --output_dir $OUT --task_tag distill_${name}_s${seed} \
    > $LOG/eval_${name}_s${seed}.log 2>&1
  echo "  done $name seed=$seed exit=$?"
}

run gsm8k      /mnt/d/fine-tuning/data/gsm8k/test.jsonl                       42
run gsm8k      /mnt/d/fine-tuning/data/gsm8k/test.jsonl                       43
run math500aug /mnt/d/fine-tuning/data/math500_aug/math500_aug_numeric.jsonl  42
run math500aug /mnt/d/fine-tuning/data/math500_aug/math500_aug_numeric.jsonl  43
echo "EVAL_CHAIN_DONE"

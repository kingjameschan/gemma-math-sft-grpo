#!/bin/bash
# E7 RESUME: only the 4 BEFORE (ptbase = raw pretrain base, no LoRA) eval runs.
# The 4 ptsft (after) runs are already done & preserved — do NOT re-run them.
PT_BASE=/mnt/d/fine-tuning/models/gemma-2-2b
OUT=/mnt/d/fine-tuning/v3/E6_distill/outputs
EVAL=/mnt/d/fine-tuning/v3/E1_baseline/eval/03_eval_pass_at_k.py
PY=/home/kingjames/vllm-env/bin/python
LOG=/mnt/d/fine-tuning/v3/E6_distill/logs
GSM=/mnt/d/fine-tuning/data/gsm8k/test.jsonl
MATH=/mnt/d/fine-tuning/data/math500_aug/math500_aug_numeric.jsonl

run() {  # tag  testfile  seed   (ckpt=base → no LoRA, eval PT_BASE directly)
  echo "===== EVAL $1 seed=$3 ====="
  VLLM_USE_V1=0 $PY $EVAL --base_model "$PT_BASE" --ckpt base --test_file "$2" \
    --k 64 --chunk_size 50 --max_lora_rank 64 --max_model_len 1408 --seed $3 \
    --output_dir $OUT --task_tag $1_s$3 \
    > $LOG/eval_$1_s$3.log 2>&1
  echo "  done $1 s$3 exit=$?"
}

run ptbase_gsm8k      "$GSM"  42
run ptbase_gsm8k      "$GSM"  43
run ptbase_math500aug "$MATH" 42
run ptbase_math500aug "$MATH" 43
echo "PT_BASE_EVAL_DONE"

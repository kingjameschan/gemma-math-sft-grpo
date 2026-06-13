#!/bin/bash
# E7 pretrain-base eval: BOTH (1) pretrain base alone [BEFORE] and
# (2) pretrain+distill-SFT LoRA [AFTER], on GSM8K + math500aug, seed42+43 K=64.
# Pretrain base (gemma-2-2b) has chat_template injected to match it eval protocol.
PT_BASE=/mnt/d/fine-tuning/models/gemma-2-2b
SFT=/mnt/d/fine-tuning/v3/E6_distill/checkpoints_pt/final
OUT=/mnt/d/fine-tuning/v3/E6_distill/outputs
EVAL=/mnt/d/fine-tuning/v3/E1_baseline/eval/03_eval_pass_at_k.py
PY=/home/kingjames/vllm-env/bin/python
LOG=/mnt/d/fine-tuning/v3/E6_distill/logs
GSM=/mnt/d/fine-tuning/data/gsm8k/test.jsonl
MATH=/mnt/d/fine-tuning/data/math500_aug/math500_aug_numeric.jsonl

run() {  # tag  ckpt  testfile  seed
  echo "===== EVAL $1 seed=$4 ====="
  VLLM_USE_V1=0 $PY $EVAL --base_model "$PT_BASE" --ckpt "$2" --test_file "$3" \
    --k 64 --chunk_size 50 --max_lora_rank 64 --max_model_len 1408 --seed $4 \
    --output_dir $OUT --task_tag $1_s$4 \
    > $LOG/eval_$1_s$4.log 2>&1
  echo "  done $1 s$4 exit=$?"
}

# AFTER: pretrain + distill-SFT LoRA
run ptsft_gsm8k      "$SFT"   "$GSM"  42
run ptsft_gsm8k      "$SFT"   "$GSM"  43
run ptsft_math500aug "$SFT"   "$MATH" 42
run ptsft_math500aug "$SFT"   "$MATH" 43
# BEFORE: pretrain base alone (ckpt=base → no LoRA, eval PT_BASE directly)
run ptbase_gsm8k      base    "$GSM"  42
run ptbase_gsm8k      base    "$GSM"  43
run ptbase_math500aug base    "$MATH" 42
run ptbase_math500aug base    "$MATH" 43
echo "PT_EVAL_CHAIN_DONE"

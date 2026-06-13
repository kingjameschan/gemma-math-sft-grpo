# E6 Distillation — LOCAL E2E (Goal: 缩到100k本地E2E)

## DONE
- Phase 0: clean_all.parquet 10.28M (decontaminated)
- Phase 1: 100K subset → v3/E6_distill/data/distill_train_100k.jsonl (source: aug_math 58K/aug_gsm 20K/math 17K/gsm 4.5K) + distill_dev.jsonl(500)
- OOM fixed: batch=2 accum=8 len=768 8bit-adam expandable_segments. batch=4 SLOWER (20s/it) → batch=2 (10.5s/step) optimal.

## RUNNING (Phase 2 training)
- local_train.py PID 67782, log: v3/E6_distill/logs/train_100k.log
- config: 100K × 2ep, batch=2 accum=8 len=768, r=64 a=32 lr=5e-4 (= SFT-ck130 config, only data source differs)
- ~5700 steps @ 10.5s = ~17h. ckpt save_steps=1400 (4 ckpts + final).
- env: siren python.exe (D:\ paths), PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

## NEXT (after train done — cron 66e8b36f handles)
- Phase 3 eval: K=128 (seed42+43 merge) on GSM8K test 1319 + math500_aug 500
  VLLM_USE_V1=0 ~/vllm-env/bin/python v3/E1_baseline/eval/03_eval_pass_at_k.py --ckpt v3/E6_distill/checkpoints/final --test_file <set> --k 64 --chunk_size 50 --max_lora_rank 64 --max_model_len 1408 --seed 42/43 --output_dir v3/E6_distill/outputs --task_tag distill_*
  ⚠ GPU-bound: only AFTER training done (no concurrent vLLM).
- Phase 4: 9-row combined plot (mirror E2_sft/_plot_step130_combined.py), base vs distill. → v3/E6_distill/outputs/distill_combined.png
- PushNotification on full completion.

## base K=128 ref data (for plot)
- GSM: v3/E5_grpo/outputs/k128_merged/base_k128_gsm8k_verbose.json
- MATH: v3/E5_grpo/outputs/k128_merged_math500_aug_slice/base_k128_math500_aug_verbose.json

#!/usr/bin/env bash
# v3 E5 GRPO Stage 1 fastgrid sweep:
#   4 LR × 3 beta = 12 configs
#   max_steps=20, save_steps=5 (4 ckpts/run = 48 ckpts total)
#   G=8, T=0.7, top_p=1.0, scheduler=constant_with_warmup
#
# After training, runs D_dev pass@1 eval on all ckpts via the SFT
# fastgrid eval script (single vLLM session, LoRA hot-swap).
#
# Usage:
#   bash v3/E5_grpo/tools/run_stage1.sh
#   bash v3/E5_grpo/tools/run_stage1.sh --dry_run
#   bash v3/E5_grpo/tools/run_stage1.sh --skip_train  # only re-run eval
#   bash v3/E5_grpo/tools/run_stage1.sh --skip_eval   # only train
set -euo pipefail

TRAIN_PYTHON="${TRAIN_PYTHON:-$HOME/train-env/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-$HOME/vllm-env/bin/python}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

TRAIN_SCRIPT="$ROOT/v3/E5_grpo/train/01_grpo.py"
EVAL_SCRIPT="$ROOT/v3/E2_sft/eval/05_fastgrid_eval.py"
DEV_FILE="$ROOT/v3/shared/data/sft/dev.jsonl"

CKPT_BASE="$ROOT/v3/E5_grpo/checkpoints/fastgrid/stage1"
EVAL_OUT="$ROOT/v3/E5_grpo/outputs/fastgrid/stage1_eval"
LOG_DIR="$ROOT/v3/E5_grpo/outputs/fastgrid/stage1_logs"

DRY_RUN=0
SKIP_TRAIN=0
SKIP_EVAL=0
for arg in "$@"; do
    case "$arg" in
        --dry_run|--dry-run) DRY_RUN=1 ;;
        --skip_train) SKIP_TRAIN=1 ;;
        --skip_eval) SKIP_EVAL=1 ;;
    esac
done

mkdir -p "$CKPT_BASE" "$EVAL_OUT" "$LOG_DIR"

LRS=("1e-6" "5e-6" "1e-5" "5e-5")
BETAS=("0.01" "0.04" "0.1")

total=$((${#LRS[@]} * ${#BETAS[@]}))
echo "=========================================="
echo "v3 E5 GRPO Stage 1 fastgrid sweep"
echo "  configs : $total  (4 LR × 3 beta)"
echo "  LRs     : ${LRS[*]}"
echo "  betas   : ${BETAS[*]}"
echo "  G=8, max_steps=20, save_steps=5, T=0.7"
echo "  ckpts   : $CKPT_BASE"
echo "  eval    : $EVAL_OUT"
echo "  logs    : $LOG_DIR"
echo "=========================================="

t_start=$(date +%s)

# ============================================================
# Phase 1: Training (12 configs sequential)
# ============================================================
if [[ $SKIP_TRAIN -eq 0 ]]; then
    i=0
    n_run=0
    n_skip=0
    for lr in "${LRS[@]}"; do
        for beta in "${BETAS[@]}"; do
            i=$((i+1))
            out_dir="$CKPT_BASE/lr${lr}_b${beta}"
            log_file="$LOG_DIR/lr${lr}_b${beta}.log"

            if [[ -d "$out_dir/checkpoint-20" ]]; then
                echo "[$i/$total] [skip] lr=$lr beta=$beta (checkpoint-20 exists)"
                n_skip=$((n_skip+1))
                continue
            fi

            echo ""
            echo "[$i/$total] lr=$lr  beta=$beta"
            echo "  out: $out_dir"
            echo "  log: $log_file"

            if [[ $DRY_RUN -eq 1 ]]; then
                echo "  [DRY] would run: $TRAIN_PYTHON $TRAIN_SCRIPT --lr $lr --beta $beta"
                continue
            fi

            # Clean partial outputs (no checkpoint-20 means previous run failed mid-way)
            if [[ -d "$out_dir" ]]; then
                rm -rf "$out_dir"
            fi

            $TRAIN_PYTHON $TRAIN_SCRIPT \
                --lr "$lr" \
                --beta "$beta" \
                --output_dir "$out_dir" \
                2>&1 | tee "$log_file"
            n_run=$((n_run+1))
        done
    done
    t_train=$(($(date +%s) - t_start))
    echo ""
    echo "=========================================="
    echo "Training phase done in $((t_train / 60)) min"
    echo "  ran     : $n_run"
    echo "  skipped : $n_skip"
    echo "=========================================="
fi

# ============================================================
# Phase 2: D_dev evaluation (single vLLM session, all ckpts)
# ============================================================
if [[ $SKIP_EVAL -eq 0 ]]; then
    echo ""
    echo "=========================================="
    echo "Eval phase: D_dev pass@1 on all ckpts"
    echo "=========================================="

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY] would run: $VLLM_PYTHON $EVAL_SCRIPT --ckpt_root $CKPT_BASE --out_dir $EVAL_OUT --dev_file $DEV_FILE"
    else
        $VLLM_PYTHON $EVAL_SCRIPT \
            --ckpt_root "$CKPT_BASE" \
            --out_dir "$EVAL_OUT" \
            --dev_file "$DEV_FILE" \
            2>&1 | tee "$LOG_DIR/_eval.log"
    fi
fi

t_total=$(($(date +%s) - t_start))
echo ""
echo "=========================================="
echo "Stage 1 ALL DONE in $((t_total / 60)) min"
echo "  ckpts   : $CKPT_BASE"
echo "  eval    : $EVAL_OUT"
echo "  logs    : $LOG_DIR"
echo "=========================================="

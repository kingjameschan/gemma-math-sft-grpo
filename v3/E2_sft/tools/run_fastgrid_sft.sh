#!/usr/bin/env bash
# Fastgrid SFT sweep: 8 LRs × 4 ranks = 32 configs, max_steps=9, save every 3.
#
# Goal: prove SFT pass@1 collapses fast (within ~10% of train data) regardless
# of LR / LoRA rank. Output: 32×3 = 96 ckpts → eval to 8×4 heatmap.
#
# Cost: ~80 sec/run × 32 ≈ 45 min train.
#
# Usage:
#   bash v3/tools/run_fastgrid_sft.sh
#   bash v3/tools/run_fastgrid_sft.sh --dry_run
set -euo pipefail

PYTHON="${TRAIN_PYTHON:-$HOME/train-env/bin/python}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TRAIN_SCRIPT="$ROOT/v3/train/01_sft.py"
CKPT_BASE="$ROOT/v3/checkpoints/fastgrid"
LOG_DIR="$ROOT/v3/outputs/fastgrid_train_logs"

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry_run|--dry-run) DRY_RUN=1 ;;
    esac
done

mkdir -p "$CKPT_BASE" "$LOG_DIR"

LRS=("1e-5" "5e-5" "1e-4" "2.5e-4" "5e-4" "7.5e-4" "1e-3" "2.5e-3")
RANKS=(8 16 32 64)

total=$((${#LRS[@]} * ${#RANKS[@]}))
i=0
n_run=0
n_skip=0

t_start=$(date +%s)
echo "=========================================="
echo "Fastgrid SFT sweep — $total configs"
echo "  LRs   : ${LRS[*]}"
echo "  ranks : ${RANKS[*]}"
echo "  steps : 9 (save 3/6/9), warmup=6 (matches main run)"
echo "  out   : $CKPT_BASE"
echo "=========================================="

for lr in "${LRS[@]}"; do
    for r in "${RANKS[@]}"; do
        i=$((i+1))
        out_dir="$CKPT_BASE/sft_lr${lr}_r${r}"
        log_file="$LOG_DIR/sft_lr${lr}_r${r}.log"

        # Skip if all 3 ckpts present
        if [[ -d "$out_dir/checkpoint-3" && -d "$out_dir/checkpoint-6" && -d "$out_dir/checkpoint-9" ]]; then
            echo "[$i/$total] [skip] sft_lr${lr}_r${r} (3 ckpts present)"
            n_skip=$((n_skip+1))
            continue
        fi

        # Clean partial output
        if [[ -d "$out_dir" ]]; then
            echo "[$i/$total] [clean] removing partial $out_dir"
            rm -rf "$out_dir"
        fi

        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "[$i/$total] [dry] sft_lr${lr}_r${r}"
            continue
        fi

        echo "[$i/$total] [train] sft_lr${lr}_r${r} → $log_file"
        t0=$(date +%s)
        "$PYTHON" "$TRAIN_SCRIPT" \
            --lr "$lr" \
            --lora_r "$r" \
            --max_steps 9 \
            --save_steps 3 \
            --warmup_steps 6 \
            --logging_steps 1 \
            --output_dir "$out_dir" \
            > "$log_file" 2>&1 \
            && status="ok" || status="FAIL"
        dt=$(( $(date +%s) - t0 ))

        if [[ -d "$out_dir/checkpoint-9" ]]; then
            echo "[$i/$total] [done] sft_lr${lr}_r${r} (${dt}s)"
            n_run=$((n_run+1))
        else
            echo "[$i/$total] [FAIL] sft_lr${lr}_r${r} after ${dt}s — see $log_file"
            tail -20 "$log_file" | sed 's/^/    /'
        fi
    done
done

elapsed=$(( $(date +%s) - t_start ))
echo
echo "=========================================="
echo "Fastgrid train done in $((elapsed/60)) min"
echo "  ran  : $n_run"
echo "  skip : $n_skip"
echo "=========================================="
ls "$CKPT_BASE/" 2>/dev/null | wc -l | xargs echo "config dirs:"

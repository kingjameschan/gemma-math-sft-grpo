#!/usr/bin/env bash
# Batch D_dev eval over all ckpts (Phase 2).
#
# Iterates:
#   - base IT (one-time anchor)
#   - all sft_lr*_r64/checkpoint-N for the 3 LR runs
#
# Each call → v3/outputs/dev_eval/<tag>.json
#
# Usage:
#   bash v3/eval/_run_dev_eval_batch.sh
#   bash v3/eval/_run_dev_eval_batch.sh --skip_existing  # don't re-eval if json exists
set -euo pipefail

PYTHON="${VLLM_PYTHON:-$HOME/vllm-env/bin/python}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EVAL_SCRIPT="$ROOT/v3/eval/04_dev_metrics.py"
OUT_DIR="$ROOT/v3/outputs/dev_eval"
CKPT_DIR="$ROOT/v3/checkpoints"

SKIP_EXISTING=0
for arg in "$@"; do
    case "$arg" in
        --skip_existing) SKIP_EXISTING=1 ;;
    esac
done

mkdir -p "$OUT_DIR"

run_eval() {
    local ckpt_arg="$1"
    local tag="$2"
    local out_file="$OUT_DIR/${tag}.json"
    if [[ "$SKIP_EXISTING" == "1" && -f "$out_file" ]]; then
        echo "  [skip] $tag (already exists)"
        return
    fi
    echo "  [eval] $tag"
    "$PYTHON" "$EVAL_SCRIPT" --ckpt "$ckpt_arg" 2>&1 | tail -10
}

# 1. Base IT (must run first to get base_pass8 for ckpt selection)
echo "=== base IT ==="
run_eval "base" "base_gemma-2-2b-it"

# 2. Each LR run's ckpts
for lr_dir in "$CKPT_DIR"/sft_lr*_r64; do
    [[ -d "$lr_dir" ]] || continue
    lr_name="$(basename "$lr_dir")"
    echo "=== $lr_name ==="
    for ckpt in "$lr_dir"/checkpoint-*; do
        [[ -d "$ckpt" ]] || continue
        ckpt_name="$(basename "$ckpt")"
        tag="${lr_name}_${ckpt_name}"
        run_eval "$ckpt" "$tag"
    done
done

echo
echo "=== Done. ==="
ls "$OUT_DIR" | wc -l | xargs echo "Total result files:"

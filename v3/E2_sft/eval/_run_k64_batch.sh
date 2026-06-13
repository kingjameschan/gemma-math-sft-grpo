#!/usr/bin/env bash
# Batch K=64 deep analysis over the 9 selected ckpts (Phase 3).
#
# Reads selected_ckpts.json, runs v3/eval/03_eval_pass_at_k.py with K=64
# on each of the 9 (early, best, final) × 3 LR ckpts.
#
# Usage:
#   bash v3/eval/_run_k64_batch.sh
set -euo pipefail

PYTHON="${VLLM_PYTHON:-$HOME/vllm-env/bin/python}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EVAL_SCRIPT="$ROOT/v3/eval/03_eval_pass_at_k.py"
SELECTED_JSON="$ROOT/v3/outputs/dev_eval/selected_ckpts.json"

if [[ ! -f "$SELECTED_JSON" ]]; then
    echo "ERR: $SELECTED_JSON not found. Run v3/tools/_select_sft_ckpts.py first."
    exit 1
fi

# Extract 9 ckpt paths from selected_ckpts.json via Python (jq might not be installed)
mapfile -t CKPT_PATHS < <(
    "$PYTHON" - <<EOF
import json
d = json.load(open("$SELECTED_JSON"))
for lr_str, sel in d["selections"].items():
    for label in ("early", "best", "final"):
        c = sel[label]
        print(c["ckpt_path"])
EOF
)

echo "Will eval ${#CKPT_PATHS[@]} ckpts with K=64:"
for p in "${CKPT_PATHS[@]}"; do
    echo "  $p"
done

for ckpt in "${CKPT_PATHS[@]}"; do
    echo
    echo "=== K=64 on $ckpt ==="
    "$PYTHON" "$EVAL_SCRIPT" --ckpt "$ckpt" --k 64 --max_lora_rank 64
done

echo
echo "=== K=64 batch done ==="

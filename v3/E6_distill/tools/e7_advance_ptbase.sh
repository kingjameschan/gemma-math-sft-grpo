#!/bin/bash
# E7 RESUME driver — runs ONLY the 4 ptbase (before) evals, then merge+plot.
# Preserves the 4 already-done ptsft runs (never wipes them). Idempotent.
set +e
B=/mnt/d/fine-tuning/v3/E6_distill
LOG=$B/logs
OUT=$B/outputs
PLOT=$OUT/distill_pretrain_combined.png
PY=/home/kingjames/vllm-env/bin/python
TOOLS=$B/tools
mkdir -p "$LOG" "$OUT"

# single-instance lock
exec 9>"$LOG/.e7_ptbase.lock" 2>/dev/null || exec 9>/tmp/.e7_ptbase.lock
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then echo "STATE=LOCKED"; exit 0; fi

# 0) fully done
if [ -f "$PLOT" ]; then echo "STATE=ALL_DONE plot=$PLOT"; exit 0; fi

# count completed ptbase runs (dirs with a *_k64.json)
done_n=0
for tag in ptbase_gsm8k ptbase_math500aug; do for s in 42 43; do
  ls $OUT/pass_at_k_${tag}_s${s}_*/*_k64.json >/dev/null 2>&1 && done_n=$((done_n+1))
done; done

# 1) eval running?
if pgrep -f 03_eval_pass_at_k >/dev/null 2>&1; then
  echo "STATE=EVAL_RUNNING ptbase_done=$done_n/4"; exit 0
fi

# 2) all 4 ptbase done -> merge + plot
if [ "$done_n" -ge 4 ]; then
  echo "STATE=EVAL_DONE -> merge+plot"
  $PY "$TOOLS/merge_pt_k128.py" > "$LOG/merge_pt.log" 2>&1
  $PY "$TOOLS/plot_pt_combined.py" > "$LOG/plot_pt.log" 2>&1
  if [ -f "$PLOT" ]; then echo "STATE=PLOT_DONE plot=$PLOT"; tail -6 "$LOG/merge_pt.log"
  else echo "STATE=PLOT_FAILED"; tail -15 "$LOG/plot_pt.log"; fi
  exit 0
fi

# 3) not running, not all done -> clean only INCOMPLETE ptbase dirs, relaunch ptbase eval
for d in $OUT/pass_at_k_ptbase_*; do
  [ -d "$d" ] && ! ls "$d"/*_k64.json >/dev/null 2>&1 && rm -rf "$d"
done
pkill -9 -f vllm 2>/dev/null; true
nohup bash "$TOOLS/run_eval_ptbase_only.sh" > "$LOG/eval_ptbase.log" 2>&1 9>&- &
echo "STATE=EVAL_LAUNCHED pid=$! ptbase_done=$done_n/4"
exit 0

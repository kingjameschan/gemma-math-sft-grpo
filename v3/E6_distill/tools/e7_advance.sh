#!/bin/bash
# E7 chain state machine — idempotent, safe to call repeatedly (by cron).
# Advances: TRAIN -> EVAL(8 runs) -> MERGE(K128) -> PLOT. Prints STATE=... .
set +e
B=/mnt/d/fine-tuning/v3/E6_distill
LOG=$B/logs
OUT=$B/outputs
FINAL=$B/checkpoints_pt/final
PLOT=$OUT/distill_pretrain_combined.png
PY=/home/kingjames/vllm-env/bin/python
TOOLS=$B/tools
mkdir -p "$LOG" "$OUT"

# concurrency guard: cron + monitor may call this simultaneously. Only one
# instance may run the launch logic, else two eval chains could start (GPU OOM).
exec 9>"$LOG/.e7_advance.lock" 2>/dev/null || exec 9>/tmp/.e7_advance.lock
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
  echo "STATE=LOCKED (another driver instance running)"; exit 0
fi

# 0) fully done
if [ -f "$PLOT" ]; then echo "STATE=ALL_DONE plot=$PLOT"; exit 0; fi

# 1) training still running
if pgrep -f local_train.py >/dev/null 2>&1; then
  S=$(tr '\r' '\n' < "$LOG/train_pt_100k.log" 2>/dev/null | grep -aoE "[0-9]+/6274" | tail -1)
  echo "STATE=TRAIN_RUNNING step=${S:-?}"; exit 0
fi

# 2) training not running, no final dir -> crashed. Auto-resume from latest
#    checkpoint (identical hyperparams). Counter resets when training has
#    progressed since last resume; stop only on a genuine crash-loop (3 no-progress).
if [ ! -d "$FINAL" ]; then
  HAVE=$(ls -d $B/checkpoints_pt/checkpoint-* 2>/dev/null | wc -l)
  CUR=$(ls -d $B/checkpoints_pt/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1); CUR=${CUR:-0}
  read RC LAST < <(cat $LOG/resume_count 2>/dev/null || echo "0 0"); RC=${RC:-0}; LAST=${LAST:-0}
  if [ "$HAVE" -ge 1 ]; then
    [ "$CUR" -gt "$LAST" ] && RC=0          # progressed since last resume -> reset
    if [ "$RC" -lt 3 ]; then
      echo "$((RC+1)) $CUR" > $LOG/resume_count
      pkill -9 -f local_train.py 2>/dev/null; true
      echo "STATE=TRAIN_RESUMED from=checkpoint-$CUR attempt=$((RC+1))/3"
      bash "$TOOLS/launch_train_pt.sh" --resume 9>&-   # close lock FD so child doesn't hold it
      exit 0
    fi
  fi
  echo "STATE=TRAIN_DEAD_NO_FINAL ckpts=$HAVE last_resume_step=$LAST attempts=$RC (check $LOG/train_pt_100k.log)"; exit 2
fi

# 3) final exists. eval running?
if pgrep -f 03_eval_pass_at_k >/dev/null 2>&1; then
  N=$(ls -d $OUT/pass_at_k_pt*_s*/ 2>/dev/null | wc -l)
  echo "STATE=EVAL_RUNNING dirs=$N/8"; exit 0
fi

# 4) eval chain finished -> merge + plot
if grep -q PT_EVAL_CHAIN_DONE "$LOG/eval_chain_pt.log" 2>/dev/null; then
  echo "STATE=EVAL_DONE -> merge+plot"
  $PY "$TOOLS/merge_pt_k128.py" > "$LOG/merge_pt.log" 2>&1
  $PY "$TOOLS/plot_pt_combined.py" > "$LOG/plot_pt.log" 2>&1
  if [ -f "$PLOT" ]; then
    echo "STATE=PLOT_DONE plot=$PLOT"
    tail -6 "$LOG/merge_pt.log"
  else
    echo "STATE=PLOT_FAILED"; tail -15 "$LOG/plot_pt.log"
  fi
  exit 0
fi

# 5) final exists, eval not done. First launch (no log) -> launch.
#    Relaunch (log exists, proc dead, no DONE) -> clean all pt eval dirs + relaunch.
if [ -f "$LOG/eval_chain_pt.log" ]; then
  echo "STATE=EVAL_DIED -> clean partials + relaunch"
  rm -rf $OUT/pass_at_k_ptbase_* $OUT/pass_at_k_ptsft_* 2>/dev/null
fi
pkill -9 -f vllm 2>/dev/null; true
nohup bash "$TOOLS/run_eval_pt_chain.sh" > "$LOG/eval_chain_pt.log" 2>&1 9>&- &   # 9>&- so child doesn't hold lock
echo "STATE=EVAL_LAUNCHED pid=$!"
exit 0

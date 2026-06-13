#!/bin/bash
# R13 launch wrapper (no GOOGLE_API_KEY needed, rule-based reward)
MASTER_LOG=/home/ubuntu/r13_full_master.log
: > $MASTER_LOG
exec /home/ubuntu/run_dapo_r13.sh >> $MASTER_LOG 2>&1

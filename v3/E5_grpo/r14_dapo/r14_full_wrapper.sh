#!/bin/bash
# R14 launch wrapper (rule-based reward, GSM8K+MATH, lr 2e-5)
MASTER_LOG=/home/ubuntu/r14_full_master.log
: > $MASTER_LOG
exec /home/ubuntu/run_dapo_r14.sh >> $MASTER_LOG 2>&1

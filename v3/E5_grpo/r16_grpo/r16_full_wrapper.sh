#!/bin/bash
MASTER_LOG=/home/ubuntu/r16_full_master.log
: > $MASTER_LOG
exec /home/ubuntu/run_dapo_r16.sh >> $MASTER_LOG 2>&1

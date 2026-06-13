#!/bin/bash
# R12 verl DAPO FULL launch — 240 steps (optimized config)
MASTER_LOG=/home/ubuntu/r12_full_master.log
: > $MASTER_LOG
export GOOGLE_API_KEY=$(aws ssm get-parameter --name /v3/grpo/google_api_key --with-decryption --region us-east-1 --query Parameter.Value --output text)
exec /home/ubuntu/run_dapo_r12.sh >> $MASTER_LOG 2>&1

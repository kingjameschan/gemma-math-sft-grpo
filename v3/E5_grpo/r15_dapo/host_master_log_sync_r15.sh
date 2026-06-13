#!/bin/bash
HOST_LOG=/home/ubuntu/r15_full_master.log
HOST_REWARD=/home/ubuntu/r15_reward_log.jsonl
S3_LOG=s3://kingjameschan-fine-tuning-v3/baseit_r15_verl_dapo_full_15ep/logs/master.log
S3_REWARD=s3://kingjameschan-fine-tuning-v3/baseit_r15_verl_dapo_full_15ep/logs/reward_log.jsonl
while true; do
    aws s3 cp $HOST_LOG $S3_LOG --quiet 2>/dev/null
    aws s3 cp $HOST_REWARD $S3_REWARD --quiet 2>/dev/null
    sleep 60
done

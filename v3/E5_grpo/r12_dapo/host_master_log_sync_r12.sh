#!/bin/bash
# Sync R12 host master log to S3 every 60s
HOST_LOG=/home/ubuntu/r12_full_master.log
S3_LOG=s3://kingjameschan-fine-tuning-v3/baseit_r12_verl_dapo_full_15ep/logs/master.log
while true; do
    aws s3 cp $HOST_LOG $S3_LOG --quiet 2>/dev/null
    sleep 60
done

#!/bin/bash
# Try A10 spot across regions/zones until one has stock. Print INSTANCE_ID + REGION on success.
ALI=~/.local/bin/aliyun
KEYNAME=cc-distill-hz
IMG_HZ=ubuntu_22_04_x64_100G_with_gpu_driver_and_cuda_alibase_20260405.vhd

# (region, zone, instance_type, sg, vsw) — only hangzhou has prebuilt net; others need keypair too
# Strategy: hammer hangzhou zones j/k with A10, fall back to gn7i-c16 (also A10, 1 GPU)
REGION=cn-hangzhou
SG=sg-bp15d1dyabdvr62d2kn2
declare -A VSW=( [cn-hangzhou-k]=vsw-bp1pqr5v4vllg18u2e4ax )
TYPES="ecs.gn7i-c8g1.2xlarge ecs.gn7i-c16g1.4xlarge"

for round in $(seq 1 30); do
  for zone in cn-hangzhou-k; do
    vsw=${VSW[$zone]}
    for t in $TYPES; do
      OUT=$($ALI ecs RunInstances --RegionId $REGION --ZoneId $zone \
        --InstanceType $t --ImageId $IMG_HZ \
        --SecurityGroupId $SG --VSwitchId $vsw --KeyPairName $KEYNAME \
        --InstanceName cc-distill-a10 \
        --SystemDisk.Category cloud_essd --SystemDisk.Size 100 \
        --InternetMaxBandwidthOut 100 --SpotStrategy SpotAsPriceGo --Amount 1 2>&1)
      if echo "$OUT" | grep -q InstanceIdSet; then
        IID=$(echo "$OUT" | ~/vllm-env/bin/python -c "import json,sys; print(json.load(sys.stdin)['InstanceIdSets']['InstanceIdSet'][0])" 2>/dev/null)
        echo "LAUNCHED $IID region=$REGION zone=$zone type=$t round=$round"
        echo "$IID" > /tmp/launched_iid.txt
        echo "$REGION" > /tmp/launched_region.txt
        exit 0
      elif echo "$OUT" | grep -q NoStock; then
        echo "[r$round $zone $t] NoStock"
      else
        echo "[r$round $zone $t] ERR: $(echo "$OUT" | grep -i message | head -1)"
      fi
    done
  done
  sleep 20
done
echo "EXHAUSTED — no stock after 30 rounds"
exit 1

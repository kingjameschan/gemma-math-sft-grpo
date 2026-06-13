#!/bin/bash
# Scan GPU instance-type stock across regions. Report WithStock zones.
ALI=~/.local/bin/aliyun
PY=~/vllm-env/bin/python
# instance types: A10 (gn7i), L20 (gn8is), T4 (gn6i), A10*2/4, V100(gn6v), A100(gn7)
TYPES="ecs.gn7i-c8g1.2xlarge ecs.gn8is-2x.8xlarge ecs.gn8is.4xlarge ecs.gn8is-2x.4xlarge ecs.gn6i-c4g1.xlarge ecs.gn6v-c8g1.2xlarge"
REGIONS="cn-hangzhou cn-shanghai cn-beijing cn-shenzhen cn-wulanchabu cn-heyuan cn-chengdu ap-southeast-1"
for r in $REGIONS; do
  # get zones for region
  ZONES=$($ALI ecs DescribeZones --RegionId $r 2>/dev/null | $PY -c "import json,sys; print(' '.join(z['ZoneId'] for z in json.load(sys.stdin).get('Zones',{}).get('Zone',[])))" 2>/dev/null)
  for z in $ZONES; do
    for t in $TYPES; do
      ST=$($ALI ecs DescribeAvailableResource --RegionId $r --ZoneId $z \
        --DestinationResource InstanceType --InstanceType $t \
        --InstanceChargeType PostPaid --SpotStrategy SpotAsPriceGo 2>/dev/null | \
        $PY -c "import json,sys
try:
  d=json.load(sys.stdin); zs=d.get('AvailableZones',{}).get('AvailableZone',[])
  for zz in zs:
    print(zz.get('StatusCategory'))
except: pass" 2>/dev/null)
      if echo "$ST" | grep -q WithStock; then
        echo "STOCK $r $z $t"
      fi
    done
  done
done
echo "SCAN_DONE"

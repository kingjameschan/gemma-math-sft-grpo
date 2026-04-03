#!/bin/bash
# GCP 实例开机自动初始化脚本
# 用法：gcloud compute instances create <name> \
#   --metadata-from-file startup-script=gcp/startup.sh
#
# 只在首次开机时安装，后续开机跳过（通过标记文件判断）

MARKER="/home/${SUDO_USER:-$(logname)}/.setup_done"
USER_HOME="/home/${SUDO_USER:-$(logname)}"
LOG="$USER_HOME/startup.log"

if [ -f "$MARKER" ]; then
    echo "$(date): Setup already done, skipping." >> "$LOG"
    exit 0
fi

echo "$(date): Starting setup..." >> "$LOG"

# 1. 系统依赖
apt-get update -qq && apt-get install -y -qq git-lfs >> "$LOG" 2>&1
git lfs install >> "$LOG" 2>&1

# 2. Python 依赖
pip install vllm peft transformers huggingface_hub datasets trl >> "$LOG" 2>&1

# 3. 配置 git
sudo -u $(logname) git config --global user.name "KingjamesChan"
sudo -u $(logname) git config --global user.email "1925716170cyk@gmail.com"

# 4. 拉代码
sudo -u $(logname) bash -c "
cd $USER_HOME
git clone https://github.com/KingjamesChan/gemma-math-sft-grpo.git >> $LOG 2>&1
cd gemma-math-sft-grpo
git lfs pull >> $LOG 2>&1
"

# 5. 下载数据
sudo -u $(logname) bash -c "
cd $USER_HOME/gemma-math-sft-grpo
python3 gcp/download_data.py >> $LOG 2>&1
"

# 6. 提示手动步骤
cat >> "$USER_HOME/NEXT_STEPS.txt" << 'NEXT'
=== 手动步骤 ===
1. 配置 HuggingFace token:
   huggingface-cli login

2. 下载模型:
   huggingface-cli download google/gemma-2-2b-it --local-dir ~/models/gemma-2-2b-it

3. 开始训练或评测:
   cd ~/gemma-math-sft-grpo
   python3 gcp/train_grpo_l4.py --help
NEXT

touch "$MARKER"
echo "$(date): Setup complete! See ~/NEXT_STEPS.txt" >> "$LOG"

#!/bin/bash
# GCP 实例开机自动初始化脚本
# 用法：gcloud compute instances create <name> \
#   --metadata-from-file startup-script=gcp/startup.sh
#
# 只在首次开机时安装，后续开机跳过（通过标记文件判断）

set -e

# 找到实际登录用户（startup script 以 root 运行）
REAL_USER=$(getent passwd 1000 | cut -d: -f1)
USER_HOME="/home/$REAL_USER"
MARKER="$USER_HOME/.setup_done"
LOG="$USER_HOME/startup.log"

if [ -f "$MARKER" ]; then
    echo "$(date): Setup already done, skipping." >> "$LOG"
    exit 0
fi

echo "$(date): Starting setup..." > "$LOG"

# 1. 系统依赖
echo "$(date): Installing system packages..." >> "$LOG"
apt-get update -qq >> "$LOG" 2>&1
apt-get install -y -qq git-lfs python3.12-venv >> "$LOG" 2>&1
sudo -u $REAL_USER git lfs install >> "$LOG" 2>&1

# 2. Node.js + Claude Code
echo "$(date): Installing Node.js and Claude Code..." >> "$LOG"
curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >> "$LOG" 2>&1
apt-get install -y -qq nodejs >> "$LOG" 2>&1
npm install -g @anthropic-ai/claude-code >> "$LOG" 2>&1

# 3. Python venv + 依赖
echo "$(date): Creating venv and installing Python packages..." >> "$LOG"
sudo -u $REAL_USER python3 -m venv $USER_HOME/venv >> "$LOG" 2>&1
sudo -u $REAL_USER bash -c "
source $USER_HOME/venv/bin/activate
pip install vllm peft transformers huggingface_hub datasets trl
" >> "$LOG" 2>&1

# 4. 全局 shell 配置（对所有用户生效）
cat > /etc/profile.d/claude-env.sh << 'PROFILE'
alias claude="claude --dangerously-skip-permissions"
# 自动激活 venv（如果存在）
[ -f ~/venv/bin/activate ] && source ~/venv/bin/activate
PROFILE
chmod 644 /etc/profile.d/claude-env.sh

# git 配置给默认用户
sudo -u $REAL_USER bash -c "
git config --global user.name 'KingjamesChan'
git config --global user.email '1925716170cyk@gmail.com'
"

# 5. 拉代码（公开 clone，私有 repo 需要 token）
echo "$(date): Cloning repo..." >> "$LOG"
sudo -u $REAL_USER bash -c "
cd $USER_HOME
git clone https://github.com/KingjamesChan/gemma-math-sft-grpo.git >> $LOG 2>&1 || echo 'Clone failed - repo may be private, clone manually after login' >> $LOG
if [ -d gemma-math-sft-grpo ]; then
    cd gemma-math-sft-grpo
    git lfs pull >> $LOG 2>&1
    source $USER_HOME/venv/bin/activate
    python3 gcp/download_data.py >> $LOG 2>&1
fi
"

# 6. 提示手动步骤
cat > "$USER_HOME/NEXT_STEPS.txt" << 'NEXT'
=== 环境已就绪，还需手动执行 ===

1. 如果 repo 没 clone 成功（私有 repo）：
   git clone https://<你的GitHub_Token>@github.com/KingjamesChan/gemma-math-sft-grpo.git
   cd gemma-math-sft-grpo && git lfs pull
   python3 gcp/download_data.py

2. 配置 HuggingFace token：
   huggingface-cli login

3. 下载模型：
   huggingface-cli download google/gemma-2-2b-it --local-dir ~/models/gemma-2-2b-it

4. 开始：
   cd ~/gemma-math-sft-grpo
   claude
NEXT
chown $REAL_USER:$REAL_USER "$USER_HOME/NEXT_STEPS.txt"

touch "$MARKER"
chown $REAL_USER:$REAL_USER "$MARKER" "$LOG"
echo "$(date): Setup complete! See ~/NEXT_STEPS.txt" >> "$LOG"

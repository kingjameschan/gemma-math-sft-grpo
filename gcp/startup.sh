#!/bin/bash
# GCP 实例开机自动初始化脚本
#
# 两阶段：
# 1. 开机时（root）：装全局包
# 2. 首次登录时：后台装用户级环境，不阻塞 SSH

MARKER="/etc/.gcp_startup_done"
LOG="/var/log/startup-script.log"

if [ -f "$MARKER" ]; then
    echo "$(date): Setup already done, skipping." >> "$LOG"
    exit 0
fi

echo "$(date): Starting setup..." > "$LOG"

# 1. 系统依赖（全局）
echo "$(date): Installing system packages..." >> "$LOG"
apt-get update -qq >> "$LOG" 2>&1
apt-get install -y -qq git-lfs python3.12-venv >> "$LOG" 2>&1

# 2. Node.js + Claude Code（全局）
echo "$(date): Installing Node.js and Claude Code..." >> "$LOG"
curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >> "$LOG" 2>&1
apt-get install -y -qq nodejs >> "$LOG" 2>&1
npm install -g @anthropic-ai/claude-code >> "$LOG" 2>&1

# 3. 首次登录：后台安装用户环境（不阻塞 SSH）
cat > /etc/profile.d/gcp-user-setup.sh << 'PROFILE'
if [ ! -f "$HOME/.gcp_user_setup_done" ]; then
    USER_LOG="$HOME/setup.log"

    # 轻量配置（立即完成）
    git lfs install 2>/dev/null
    git config --global user.name 'KingjamesChan'
    git config --global user.email '1925716170cyk@gmail.com'

    # 重量级安装放后台
    echo "=== 环境正在后台安装，查看进度：tail -f ~/setup.log ==="
    nohup bash -c "
        echo \"\$(date): Starting user setup...\" > $USER_LOG

        # venv + pip
        if [ ! -d \"$HOME/venv\" ]; then
            echo \"\$(date): Creating venv...\" >> $USER_LOG
            python3 -m venv \"$HOME/venv\" >> $USER_LOG 2>&1
            source \"$HOME/venv/bin/activate\"
            echo \"\$(date): Installing pip packages...\" >> $USER_LOG
            pip install -q vllm peft transformers huggingface_hub datasets trl >> $USER_LOG 2>&1
            echo \"\$(date): Installing flash-attn (MAX_JOBS=2)...\" >> $USER_LOG
            MAX_JOBS=2 pip install -q flash-attn --no-build-isolation >> $USER_LOG 2>&1
        fi

        # clone repo
        if [ ! -d \"$HOME/gemma-math-sft-grpo\" ]; then
            echo \"\$(date): Cloning repo...\" >> $USER_LOG
            git clone https://github.com/KingjamesChan/gemma-math-sft-grpo.git \"$HOME/gemma-math-sft-grpo\" >> $USER_LOG 2>&1
            if [ -d \"$HOME/gemma-math-sft-grpo\" ]; then
                cd \"$HOME/gemma-math-sft-grpo\" && git lfs pull >> $USER_LOG 2>&1
                source \"$HOME/venv/bin/activate\"
                python3 gcp/download_data.py >> $USER_LOG 2>&1
            fi
        fi

        echo \"\$(date): Setup complete!\" >> $USER_LOG
        echo \"\" >> $USER_LOG
        echo \"=== 还需手动执行 ===\" >> $USER_LOG
        echo \"1. huggingface-cli login\" >> $USER_LOG
        echo \"2. huggingface-cli download google/gemma-2-2b-it --local-dir ~/models/gemma-2-2b-it\" >> $USER_LOG
        echo \"3. cd ~/gemma-math-sft-grpo && claude\" >> $USER_LOG
    " >> /dev/null 2>&1 &

    touch "$HOME/.gcp_user_setup_done"
fi

# 每次登录都执行
alias claude="claude --dangerously-skip-permissions"
[ -f ~/venv/bin/activate ] && source ~/venv/bin/activate
PROFILE
chmod 644 /etc/profile.d/gcp-user-setup.sh

touch "$MARKER"
echo "$(date): Global setup complete." >> "$LOG"

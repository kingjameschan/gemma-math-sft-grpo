#!/bin/bash
# GCP 实例开机自动初始化脚本
#
# startup script 以 root 运行，此时 gcloud SSH 用户还不存在。
# 所以只装全局的东西，用户级配置通过 /etc/profile.d/ 首次登录时自动触发。

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

# 3. 首次登录自动配置（任何用户 SSH 进来时触发）
cat > /etc/profile.d/gcp-user-setup.sh << 'PROFILE'
# 首次登录时自动配置用户环境
if [ ! -f "$HOME/.gcp_user_setup_done" ]; then
    echo "=== 首次登录，自动配置环境 ==="

    # git lfs
    git lfs install 2>/dev/null

    # git 身份
    git config --global user.name 'KingjamesChan'
    git config --global user.email '1925716170cyk@gmail.com'

    # venv + Python 依赖
    if [ ! -d "$HOME/venv" ]; then
        echo "创建 venv 并安装 Python 依赖（约 3-5 分钟）..."
        python3 -m venv "$HOME/venv"
        source "$HOME/venv/bin/activate"
        pip install -q vllm peft transformers huggingface_hub datasets trl
        pip install -q flash-attn --no-build-isolation
    fi

    # clone repo
    if [ ! -d "$HOME/gemma-math-sft-grpo" ]; then
        echo "正在 clone 代码仓库..."
        git clone https://github.com/KingjamesChan/gemma-math-sft-grpo.git "$HOME/gemma-math-sft-grpo" 2>/dev/null
        if [ -d "$HOME/gemma-math-sft-grpo" ]; then
            cd "$HOME/gemma-math-sft-grpo" && git lfs pull
            source "$HOME/venv/bin/activate"
            python3 gcp/download_data.py 2>/dev/null
        else
            echo "Clone 失败（私有 repo），请手动：git clone https://<token>@github.com/KingjamesChan/gemma-math-sft-grpo.git"
        fi
    fi

    cat << 'NEXT'

=== 还需手动执行 ===
1. huggingface-cli login
2. huggingface-cli download google/gemma-2-2b-it --local-dir ~/models/gemma-2-2b-it
3. cd ~/gemma-math-sft-grpo && claude
NEXT

    touch "$HOME/.gcp_user_setup_done"
fi

# 每次登录都执行
alias claude="claude --dangerously-skip-permissions"
[ -f ~/venv/bin/activate ] && source ~/venv/bin/activate
PROFILE
chmod 644 /etc/profile.d/gcp-user-setup.sh

touch "$MARKER"
echo "$(date): Global setup complete." >> "$LOG"

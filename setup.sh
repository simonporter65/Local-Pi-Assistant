#!/bin/bash
# setup.sh — Full setup for Pi 5 Personal Assistant
set -e

AGENT_HOME=/mnt/nvme/agent
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Pi 5 Personal Assistant — Setup"
echo "  Private • Local • Always yours"
echo "═══════════════════════════════════════════════════"

# ── NVMe ─────────────────────────────────────────────
if [ ! -d "/mnt/nvme" ]; then
    echo "Setting up NVMe..."
    NVME=$(lsblk -d -o NAME,TYPE | grep disk | grep nvme | awk '{print $1}' | head -1)
    [ -z "$NVME" ] && echo "ERROR: No NVMe found" && exit 1
    sudo parted /dev/$NVME --script mklabel gpt
    sudo parted /dev/$NVME --script mkpart primary ext4 0% 100%
    sudo mkfs.ext4 /dev/${NVME}p1
    sudo mkdir -p /mnt/nvme
    sudo mount /dev/${NVME}p1 /mnt/nvme
    echo "/dev/${NVME}p1 /mnt/nvme ext4 defaults,noatime 0 2" | sudo tee -a /etc/fstab
fi
echo "✓ NVMe at /mnt/nvme"

# ── Swap ─────────────────────────────────────────────
if [ ! -f /mnt/nvme/swapfile ]; then
    sudo fallocate -l 8G /mnt/nvme/swapfile
    sudo chmod 600 /mnt/nvme/swapfile && sudo mkswap /mnt/nvme/swapfile && sudo swapon /mnt/nvme/swapfile
    echo "/mnt/nvme/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab
fi
cat <<EOF | sudo tee /etc/sysctl.d/99-agent-swap.conf > /dev/null
vm.swappiness=80
vm.vfs_cache_pressure=50
vm.dirty_ratio=20
vm.dirty_background_ratio=5
EOF
sudo sysctl -p /etc/sysctl.d/99-agent-swap.conf > /dev/null
echo "✓ 8GB swap on NVMe"

# ── System packages ───────────────────────────────────
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip python3-venv \
    scrot imagemagick \
    curl git sqlite3 \
    xvfb x11-utils \
    chromium-browser \
    ffmpeg htop ncdu \
    build-essential python3-dev \
    avahi-daemon  # for raspberrypi.local mDNS
echo "✓ System packages"

# ── Directories ───────────────────────────────────────
sudo mkdir -p $AGENT_HOME/{skills,memory,logs,screenshots,workspace,ui,proactive,core}
sudo chown -R $USER:$USER $AGENT_HOME
echo "✓ Directories at $AGENT_HOME"

# ── Files ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp -r "$SCRIPT_DIR"/* $AGENT_HOME/
echo "✓ Agent files copied"

# ── Python env ────────────────────────────────────────
python3 -m venv $AGENT_HOME/venv
source $AGENT_HOME/venv/bin/activate
pip install --upgrade pip -q
pip install -r $AGENT_HOME/requirements.txt -q
playwright install chromium
echo "✓ Python environment ready"

# ── Ollama ────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.ai/install.sh | sh
fi
sudo mkdir -p /mnt/nvme/ollama/models
sudo mkdir -p /etc/systemd/system/ollama.service.d
cat <<EOF | sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null
[Service]
Environment="OLLAMA_MODELS=/mnt/nvme/ollama/models"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_FLASH_ATTENTION=1"
EOF
sudo systemctl daemon-reload && sudo systemctl enable ollama && sudo systemctl restart ollama
sleep 2
echo "✓ Ollama (models at /mnt/nvme/ollama/models)"

# ── Environment ───────────────────────────────────────
cat <<EOF >> ~/.bashrc

# Pi Assistant
export AGENT_HOME=/mnt/nvme/agent
export AGENT_WORKSPACE=/mnt/nvme/agent/workspace
export AGENT_SCREENSHOTS=/mnt/nvme/agent/screenshots
export AGENT_DB=/mnt/nvme/agent/memory/agent.db
export OLLAMA_MODELS=/mnt/nvme/ollama/models
export DISPLAY=:99
EOF

# ── Systemd services ──────────────────────────────────
cat <<EOF | sudo tee /etc/systemd/system/assistant-xvfb.service > /dev/null
[Unit]
Description=Virtual Display for Assistant
After=network.target

[Service]
ExecStart=/usr/bin/Xvfb :99 -screen 0 1920x1080x24
Restart=always
User=$USER

[Install]
WantedBy=multi-user.target
EOF

cat <<EOF | sudo tee /etc/systemd/system/assistant.service > /dev/null
[Unit]
Description=Pi 5 Personal Assistant
After=network.target ollama.service assistant-xvfb.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$AGENT_HOME
EnvironmentFile=-/etc/agent-env
Environment="AGENT_HOME=$AGENT_HOME"
Environment="AGENT_WORKSPACE=$AGENT_HOME/workspace"
Environment="AGENT_SCREENSHOTS=$AGENT_HOME/screenshots"
Environment="AGENT_DB=$AGENT_HOME/memory/agent.db"
Environment="OLLAMA_MODELS=/mnt/nvme/ollama/models"
Environment="DISPLAY=:99"
ExecStart=$AGENT_HOME/venv/bin/python $AGENT_HOME/server.py
Restart=on-failure
RestartSec=5
StandardOutput=append:$AGENT_HOME/logs/assistant.log
StandardError=append:$AGENT_HOME/logs/assistant.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable assistant-xvfb assistant
echo "✓ Systemd services installed"

# ── Network access info ───────────────────────────────
LOCAL_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Next: pull models"
echo "  $ tmux new -s models && bash $AGENT_HOME/pull_models.sh"
echo ""
echo "  Then start:"
echo "  $ sudo systemctl start assistant"
echo ""
echo "  Access from any device on your network:"
echo "  http://$LOCAL_IP:8765"
echo "  http://raspberrypi.local:8765"
echo "═══════════════════════════════════════════════════"

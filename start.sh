#!/bin/bash
# start.sh — Start the assistant (server + Xvfb)
# Can be run manually or via systemd

AGENT_HOME="${AGENT_HOME:-/mnt/nvme/agent}"
VENV="$AGENT_HOME/venv/bin/python"

# Start virtual framebuffer if not running (needed for screenshots/browser)
if ! pgrep -x Xvfb > /dev/null; then
    Xvfb :99 -screen 0 1920x1080x24 &
    echo "Started Xvfb"
fi

export DISPLAY=:99
export AGENT_HOME=$AGENT_HOME
export AGENT_WORKSPACE=$AGENT_HOME/workspace
export AGENT_SCREENSHOTS=$AGENT_HOME/screenshots
export AGENT_DB=$AGENT_HOME/memory/agent.db
export OLLAMA_MODELS=/mnt/nvme/ollama/models

echo "Starting assistant at http://localhost:8765"
echo "Open in browser or phone on local network"
exec $VENV $AGENT_HOME/server.py

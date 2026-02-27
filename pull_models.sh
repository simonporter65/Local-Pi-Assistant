#!/bin/bash
# pull_models.sh — Pull all models to NVMe
# Run in tmux: tmux new -s models

set -e

pull() {
    echo ""
    echo "▶ Pulling $1..."
    ollama pull "$1"
    echo "✓ $1 done"
}

echo "═══════════════════════════════════════════"
echo "  Pulling all models (~55GB total)"
echo "  Use tmux to keep this running if SSH disconnects"
echo "═══════════════════════════════════════════"

echo ""
echo "── TIER 0: Classifier (400MB) ──"
pull qwen2.5:0.5b

echo ""
echo "── TIER 1: Fast responders (~8GB) ──"
pull llama3.2:1b
pull llama3.2:3b
pull qwen2.5:3b
pull phi4-mini:3.8b

echo ""
echo "── TIER 2: Primary 7-8B models (~30GB) ──"
pull qwen2.5-coder:7b
pull mistral:7b
pull deepseek-r1:7b
pull llava:7b
pull qwen2.5:7b
pull llama3.1:8b
pull llama3.2-vision:11b

echo ""
echo "── TIER 3: High-quality 14B (agentic) (~45GB total inc above) ──"
pull qwen2.5-coder:14b
pull deepseek-r1:14b
pull phi4:14b
pull qwen2.5:14b
pull mistral-nemo:12b
pull llava:13b

echo ""
echo "── EMBEDDINGS (1GB) ──"
pull nomic-embed-text
pull mxbai-embed-large

echo ""
echo "═══════════════════════════════════════════"
echo "  All models pulled!"
ollama list
echo "═══════════════════════════════════════════"

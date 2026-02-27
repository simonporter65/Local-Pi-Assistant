# Local Pi Assistant

> A private, autonomous AI personal assistant that runs entirely on your Raspberry Pi 5. No cloud. No subscriptions. No data leaving your hardware.

![Python](https://img.shields.io/badge/python-3.11+-blue) ![Ollama](https://img.shields.io/badge/ollama-0.2+-green) ![Pi5](https://img.shields.io/badge/hardware-Raspberry%20Pi%205-red) ![License](https://img.shields.io/badge/license-MIT-purple)

---

## What it is

A fully autonomous personal assistant with a WhatsApp-style chat UI, accessible from any device on your local network. It learns about you over time, works in the background when you're not talking to it, and runs 14B parameter models locally using NVMe swap.

Built for the Raspberry Pi 5 (8GB) with a 2TB NVMe — eventually targeting the Compute Module 5 as a dedicated private AI hardware product.

---

## Features

- **Messaging UI** — WhatsApp-style chat served at `http://raspberrypi.local:8765`, works from phone/tablet/laptop
- **Personality system** — TARS-style flavor dials (Humor, Warmth, Sass, Verbosity, Chaos) with name generation on first run
- **Learns about you** — builds a persistent user profile from every conversation, injects context into every response
- **Autonomous background loop** — heartbeat fires every 5 minutes, works through a self-managed task queue when you're not chatting
- **Self-improving** — writes its own new skills when it identifies capability gaps
- **Proactive** — morning briefings, end-of-day summaries, context-aware suggestions
- **14B models on 8GB** — runs `qwen2.5:14b`, `deepseek-r1:14b`, `phi4:14b` via NVMe swap
- **Fully private** — zero telemetry, zero cloud calls, SQLite database never leaves your device

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Browser / Phone                │
│         http://raspberrypi.local:8765           │
└───────────────────┬─────────────────────────────┘
                    │ SSE streaming
┌───────────────────▼─────────────────────────────┐
│              FastAPI Server (server.py)          │
│  /chat  /events  /setup  /tasks  /personality   │
└──┬──────────────┬──────────────┬────────────────┘
   │              │              │
┌──▼────┐  ┌─────▼──────┐  ┌───▼──────────────┐
│Pre-   │  │Personality  │  │Heartbeat Loop     │
│pipe   │  │Config       │  │(autonomous tasks) │
│(0.5b) │  │(user prefs) │  │every 5 minutes    │
└──┬────┘  └─────────────┘  └───────────────────┘
   │
┌──▼─────────────────────────────────────────────┐
│              Model Manager                      │
│  8B-first routing → escalate to 14B if needed  │
│  Keepalive pings keep warm models hot           │
└──┬─────────────────────────────────────────────┘
   │
┌──▼─────────────────────────────────────────────┐
│                  Ollama                         │
│  0.5b router · 3b chat · 8b normal · 14b deep  │
└──┬─────────────────────────────────────────────┘
   │
┌──▼─────────────────────────────────────────────┐
│              Skills (tools)                     │
│  web_search · bash_exec · browser · workspace  │
│  screenshot · skill_writer · python_repl · ...  │
└─────────────────────────────────────────────────┘
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Board | Raspberry Pi 5 4GB | Raspberry Pi 5 **8GB** |
| Storage | 256GB NVMe | **2TB NVMe** (models + swap) |
| OS | Raspberry Pi OS Bookworm (64-bit) | same |
| Power | 5V 5A USB-C | Official Pi 5 PSU |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/simonporter65/Local-Pi-Assistant.git
cd Local-Pi-Assistant

# 2. Run setup (handles NVMe, swap, packages, Ollama, systemd)
bash setup.sh

# 3. Pull models overnight (~55GB)
tmux new -s models
bash pull_models.sh

# 4. Start
sudo systemctl start assistant

# 5. Open on any device on your network
open http://raspberrypi.local:8765
```

---

## Model Stack (~55GB total)

| Tier | Model | Size | Used for |
|------|-------|------|----------|
| 0 — Router | `qwen2.5:0.5b` | 400MB | Classify + rewrite every message (stays hot) |
| 1 — Fast | `llama3.2:3b` | 2GB | General chat, background tasks |
| 2 — Normal | `llama3.1:8b` | 4.9GB | Web search, analysis, planning |
| 2 — Vision | `llama3.2-vision:11b` | 7.9GB | Screenshots, image tasks |
| 3 — Deep | `qwen2.5-coder:14b` | 9GB | Coding, debugging, skill writing |
| 3 — Reasoning | `deepseek-r1:14b` | 9GB | Math, complex reasoning |
| 3 — Planning | `phi4:14b` | 9.1GB | Long-horizon planning |
| Embeddings | `nomic-embed-text` | 270MB | Semantic memory search |

---

## File Structure

```
├── server.py               # FastAPI server, SSE streaming, all routes
├── setup.sh                # Full automated setup script
├── pull_models.sh          # Pull all Ollama models
├── start.sh                # Manual start script
├── requirements.txt
│
├── core/
│   ├── pipeline_pre.py     # Merged classify+rewrite+extract (single 0.5b call)
│   ├── model_manager.py    # 8B-first routing, keepalive, history compression
│   ├── classifier.py       # Intent classification
│   ├── router.py           # Model routing table
│   ├── executor.py         # Tool-use loop
│   └── validator.py        # Result validation
│
├── memory/
│   ├── store.py            # SQLite + semantic search
│   ├── user_model.py       # User profile — learns from every conversation
│   ├── personality.py      # Personality config (name + flavor settings)
│   └── embed_cache.py      # LRU cache for embeddings
│
├── autonomous/
│   ├── task_queue.py       # SQLite-backed self-managed task queue
│   └── heartbeat.py        # Background autonomous loop
│
├── proactive/
│   └── engine.py           # Proactive suggestions + push messages
│
├── skills/                 # Agent tools (agent writes new ones itself)
│   ├── registry.py         # Hot-reload skill loader
│   ├── web_search.py       # DuckDuckGo (no API key)
│   ├── web_fetch.py        # Full page extraction
│   ├── bash_exec.py        # Shell (with safety blocks)
│   ├── workspace.py        # File I/O
│   ├── browser.py          # Playwright headless
│   ├── skill_writer.py     # Self-improvement: writes new skills
│   ├── screenshot.py       # Screen capture
│   ├── python_repl.py      # Python execution
│   ├── memory_search.py    # Semantic memory search
│   └── system_info.py      # Pi hardware introspection
│
└── ui/
    ├── index.html          # WhatsApp-style chat UI
    ├── personality.html    # First-run personality setup
    └── heartbeat.js        # Background task status in UI
```

---

## Performance (Pi 5, 8GB RAM, NVMe swap)

| Path | Time to first token |
|------|-------------------|
| General chat (3B warm) | ~2–4s |
| Web search / analysis (8B) | ~6–10s |
| Coding / agentic (14B warm) | ~4–8s |
| Coding / agentic (14B cold) | ~25–45s |

---

## Roadmap

- [ ] CalDAV / calendar integration
- [ ] Desktop/push notifications
- [ ] Voice input (whisper.cpp)
- [ ] CM5 custom hardware enclosure
- [ ] Multi-user support
- [ ] iOS/Android companion app

---

## License

MIT — do whatever you want with it.

---

*Built conversation by conversation with Claude.*

"""
core/router.py
Routes tasks to the best available model.

Model stack (all Qwen thinking models, think=False as top-level kwarg):
- qwen3.5:0.8b  fastest — direct-mode chat and utility (1 GB)
- qwen3.5:2b    fast — medium complexity, direct or light thinking (2.7 GB)
- qwen3:4b      medium — planning/research, direct mode (2.5 GB)
- qwen3:8b      thorough — accuracy-critical, thinking mode (5 GB)
- llava:7b      vision tasks (4.7 GB)

think=False must be passed as a top-level kwarg to ollama.chat(), NOT inside options.
For ollama.generate() calls use /no_think prefix in the prompt instead.
route_to_model() returns a "thinking" bool — caller is responsible for passing it.

If scripts/lora_train.py has been run and created arc-personal, that model is
used for general_chat and conversational categories automatically.
"""
import json
import os
import subprocess
import time
from pathlib import Path

# Cache installed models — refresh every 5 minutes
_model_cache = {"models": [], "updated": 0.0}

# Cache the personal model check — refresh every 10 minutes
_personal_model_cache = {"model": None, "updated": 0.0}


def _get_personal_model() -> str | None:
    """Return arc-personal model name if it has been built, else None."""
    now = time.time()
    if now - _personal_model_cache["updated"] < 600:
        return _personal_model_cache["model"]
    _personal_model_cache["updated"] = now
    marker = Path(__file__).parent.parent / "memory" / "personal_model.json"
    try:
        if marker.exists():
            info = json.loads(marker.read_text())
            model = info.get("model")
            installed = get_installed_models()
            if model and model in installed:
                _personal_model_cache["model"] = model
                return model
    except Exception:
        pass
    _personal_model_cache["model"] = None
    return None


def get_installed_models() -> list:
    if time.time() - _model_cache["updated"] > 300:
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')[1:]
            _model_cache["models"] = [line.split()[0] for line in lines if line.strip()]
            _model_cache["updated"] = time.time()
        except Exception:
            pass
    return _model_cache["models"]


# Category → model, latency, and whether to use thinking mode
# thinking=False  → pass think=False to ollama.chat() (direct/fast mode)
# thinking=True   → omit think kwarg (model reasons before answering)
MODEL_MAP = {
    # ── Tier 1: qwen3.5:0.8b — instant conversational ────────────────────────
    "general_chat":           {"model": "qwen3.5:0.8b", "latency": "fast",   "thinking": False},
    "summarization":          {"model": "qwen3.5:0.8b", "latency": "fast",   "thinking": False},
    "translation":            {"model": "qwen3.5:0.8b", "latency": "fast",   "thinking": False},
    "task_management":        {"model": "qwen3.5:0.8b", "latency": "fast",   "thinking": False},
    "intent_classification":  {"model": "qwen3.5:0.8b", "latency": "fast",   "thinking": False},
    "sentiment_analysis":     {"model": "qwen3.5:0.8b", "latency": "fast",   "thinking": False},
    "web_search":             {"model": "qwen3.5:0.8b", "latency": "fast",   "thinking": False},

    # ── Tier 2: qwen3:4b — medium complexity, direct mode ────────────────────
    "planning":               {"model": "qwen3:4b",     "latency": "normal", "thinking": False},
    "research":               {"model": "qwen3:4b",     "latency": "normal", "thinking": False},
    "creative_writing":       {"model": "qwen3:4b",     "latency": "normal", "thinking": False},
    "agentic_task":           {"model": "qwen3:4b",     "latency": "normal", "thinking": False},

    # ── Tier 3: qwen3:8b — accuracy-critical, thinking mode ──────────────────
    "coding":                 {"model": "qwen3:8b",     "latency": "slow",   "thinking": True},
    "debugging":              {"model": "qwen3:8b",     "latency": "slow",   "thinking": True},
    "shell_command":          {"model": "qwen3:8b",     "latency": "slow",   "thinking": True},
    "web_browsing":           {"model": "qwen3:8b",     "latency": "slow",   "thinking": True},
    "skill_writing":          {"model": "qwen3:8b",     "latency": "slow",   "thinking": True},
    "structured_output":      {"model": "qwen3:8b",     "latency": "slow",   "thinking": True},
    "file_management":        {"model": "qwen3:8b",     "latency": "slow",   "thinking": True},
    "data_analysis":          {"model": "qwen3:8b",     "latency": "slow",   "thinking": True},
    "math":                   {"model": "qwen3:8b",     "latency": "slow",   "thinking": True},
    "reasoning":              {"model": "qwen3:8b",     "latency": "slow",   "thinking": True},
    "error_recovery":         {"model": "qwen3:8b",     "latency": "slow",   "thinking": True},

    # ── Vision ────────────────────────────────────────────────────────────────
    "image_description":      {"model": "llava:7b",     "latency": "slow",   "thinking": False},
    "screenshot_analysis":    {"model": "llava:7b",     "latency": "slow",   "thinking": False},
}

# Fallback chains: if primary not installed, try these in order
FALLBACK_CHAINS = {
    "qwen3.5:0.8b": ["qwen3.5:2b", "qwen3:4b", "qwen3:1.7b", "qwen3:0.6b"],
    "qwen3.5:2b":   ["qwen3:4b", "qwen3.5:0.8b", "qwen3:1.7b"],
    "qwen3:4b":     ["qwen3.5:2b", "qwen3:8b", "qwen3:1.7b"],
    "qwen3:8b":     ["qwen3:4b", "qwen3.5:2b"],
    "llava:7b":     ["qwen3.5:2b"],
}

DEFAULT = {"model": "qwen3.5:0.8b", "latency": "fast", "thinking": False}


def route_to_model(intent: dict) -> dict:
    category = intent.get("category", "general_chat")
    route = MODEL_MAP.get(category, DEFAULT).copy()

    # If arc-personal is built, use it for conversational direct-mode categories
    if not route["thinking"] and category in {
        "general_chat", "summarization", "translation",
        "task_management", "creative_writing", "planning",
    }:
        personal = _get_personal_model()
        if personal:
            route["model"] = personal

    # Check if preferred model is installed, fall back if not
    installed = get_installed_models()
    if installed and route["model"] not in installed:
        for fallback in FALLBACK_CHAINS.get(route["model"], []):
            if fallback in installed:
                route["model"] = fallback
                break
        else:
            if installed:
                route["model"] = installed[0]

    return route


def get_fallback(model: str) -> list:
    installed = get_installed_models()
    chain = FALLBACK_CHAINS.get(model, ["qwen3.5:0.8b"])
    return [m for m in chain if m in installed] or list(chain)

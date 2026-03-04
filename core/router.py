"""
core/router.py
Routes tasks to the best available model.
Uses MODEL_MAP for explicit category→model mapping.
Falls back gracefully if preferred model not installed.

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

# Category → preferred model + latency
# NOTE: qwen3.5 and qwen3 models are Qwen3 thinking models — in Ollama 0.17.4,
# all tokens go to the `thinking` field; `response`/`content` is always empty.
# Use llama3.2:3b for all conversational tasks until Ollama supports think=False.
MODEL_MAP = {
    # Fast: simple chat
    "intent_classification":  {"model": "llama3.2:3b",  "latency": "fast"},
    "sentiment_analysis":     {"model": "llama3.2:3b",  "latency": "fast"},
    "general_chat":           {"model": "llama3.2:3b",  "latency": "fast"},
    "summarization":          {"model": "llama3.2:3b",  "latency": "fast"},
    "task_management":        {"model": "llama3.2:3b",  "latency": "fast"},
    "translation":            {"model": "llama3.2:3b",  "latency": "fast"},

    # Normal: capable 8B
    "web_search":             {"model": "llama3.1:8b",         "latency": "normal"},
    "research":               {"model": "llama3.1:8b",         "latency": "normal"},
    "planning":               {"model": "llama3.1:8b",         "latency": "normal"},
    "creative_writing":       {"model": "llama3.1:8b",         "latency": "normal"},
    "image_description":      {"model": "llama3.2-vision:11b", "latency": "slow"},

    # Slow: specialist models — browser automation needs coder for reliable JSON/selectors
    "web_browsing":           {"model": "qwen2.5-coder:7b",    "latency": "slow"},
    "coding":                 {"model": "qwen2.5-coder:7b",    "latency": "slow"},
    "debugging":              {"model": "qwen2.5-coder:7b",    "latency": "slow"},
    "shell_command":          {"model": "qwen2.5-coder:7b",    "latency": "slow"},
    "skill_writing":          {"model": "qwen2.5-coder:7b",    "latency": "slow"},
    "structured_output":      {"model": "qwen2.5-coder:7b",    "latency": "slow"},
    "file_management":        {"model": "qwen2.5-coder:7b",    "latency": "slow"},
    "data_analysis":          {"model": "qwen2.5-coder:7b",    "latency": "slow"},
    "math":                   {"model": "deepseek-r1:7b",      "latency": "slow"},
    "reasoning":              {"model": "deepseek-r1:7b",      "latency": "slow"},
    "error_recovery":         {"model": "deepseek-r1:7b",      "latency": "slow"},
    "agentic_task":           {"model": "llama3.1:8b",         "latency": "slow"},
    "screenshot_analysis":    {"model": "llama3.2-vision:11b", "latency": "slow"},
}

# Fallback chains: if primary not installed or fails, try these in order
FALLBACK_CHAINS = {
    "llama3.2:3b":             ["llama3.2:1b",     "mistral:7b"],
    "qwen2.5-coder:7b":        ["llama3.1:8b",     "mistral:7b",   "llama3.2:3b"],
    "deepseek-r1:7b":          ["llama3.1:8b",     "mistral:7b",   "llama3.2:3b"],
    "llama3.1:8b":             ["mistral:7b",       "llama3.2:3b"],
    "llama3.2-vision:11b":     ["llava:13b",        "llava:7b",     "llama3.2:3b"],
    "llava:7b":                ["llama3.2:3b"],
}

DEFAULT = {"model": "llama3.2:3b", "latency": "fast"}


def route_to_model(intent: dict) -> dict:
    category = intent.get("category", "general_chat")
    route = MODEL_MAP.get(category, DEFAULT).copy()

    # If arc-personal is built, use it for conversational categories
    if category in {"general_chat", "summarization", "translation",
                    "task_management", "creative_writing", "planning"}:
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
            # Last resort — first installed model
            if installed:
                route["model"] = installed[0]

    return route


def get_fallback(model: str) -> list:
    installed = get_installed_models()
    chain = FALLBACK_CHAINS.get(model, ["llama3.1:8b", "llama3.2:3b"])
    return [m for m in chain if m in installed] or [m for m in chain]

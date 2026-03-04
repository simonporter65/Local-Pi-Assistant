"""
core/router.py
Routes tasks to the best available model.
Uses MODEL_MAP for explicit category→model mapping.
Falls back gracefully if preferred model not installed.
"""
import subprocess
import time

# Cache installed models — refresh every 5 minutes
_model_cache = {"models": [], "updated": 0.0}

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
MODEL_MAP = {
    # Instant: classifier pipeline
    "intent_classification":  {"model": "qwen3.5:0.8b",           "latency": "instant"},
    "sentiment_analysis":     {"model": "qwen3.5:0.8b",           "latency": "instant"},

    # Fast: simple chat — qwen3.5:0.8b is 15x faster than llama3.2:3b on Pi5
    "general_chat":           {"model": "qwen3.5:0.8b",          "latency": "fast"},
    "summarization":          {"model": "qwen3.5:0.8b",          "latency": "fast"},
    "task_management":        {"model": "qwen3.5:0.8b",          "latency": "fast"},
    "translation":            {"model": "qwen3.5:0.8b",          "latency": "fast"},

    # Normal: capable 8B
    "web_search":             {"model": "llama3.1:8b",         "latency": "normal"},
    "research":               {"model": "llama3.1:8b",         "latency": "normal"},
    "planning":               {"model": "llama3.1:8b",         "latency": "normal"},
    "creative_writing":       {"model": "llama3.1:8b",         "latency": "normal"},
    "image_description":      {"model": "llava:7b",            "latency": "normal"},

    # Slow: specialist models
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
    "screenshot_analysis":    {"model": "llava:7b",            "latency": "slow"},
}

# Fallback chains: if primary not installed or fails, try these in order
FALLBACK_CHAINS = {
    "qwen3.5:0.8b":          ["qwen3.5:2b",     "qwen3:1.7b",   "llama3.2:3b"],
    "qwen2.5-coder:7b":    ["llama3.1:8b",    "mistral:7b",   "llama3.2:3b"],
    "deepseek-r1:7b":      ["llama3.1:8b",    "mistral:7b",   "llama3.2:3b"],
    "llama3.1:8b":         ["mistral:7b",     "llama3.2:3b"],
    "llava:7b":            ["llama3.2:3b"],
}

DEFAULT = {"model": "llama3.2:3b", "latency": "fast"}


def route_to_model(intent: dict) -> dict:
    category = intent.get("category", "general_chat")
    route = MODEL_MAP.get(category, DEFAULT).copy()

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

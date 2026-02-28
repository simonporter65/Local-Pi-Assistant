"""
core/router.py — Dynamic routing based on installed models
"""
import subprocess
import json

def get_installed_models() -> list:
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')[1:]  # skip header
        return [line.split()[0] for line in lines if line.strip()]
    except:
        return []

# Priority order — first installed model in each tier wins
TIER_PREFERENCES = {
    "fast":   ["llama3.2:3b", "llama3.2:1b", "qwen2.5:3b", "qwen2.5:0.5b"],
    "normal": ["llama3.1:8b", "mistral:7b", "qwen2.5:7b", "llama3.2:3b"],
    "coding": ["qwen2.5-coder:7b", "qwen2.5-coder:14b", "llama3.1:8b", "llama3.2:3b"],
    "slow":   ["qwen2.5:14b", "phi4:14b", "deepseek-r1:14b", "llama3.1:8b", "llama3.2:3b"],
}

CATEGORY_TIER = {
    "general_chat":       "fast",
    "summarization":      "fast",
    "translation":        "fast",
    "sentiment_analysis": "fast",
    "web_search":         "normal",
    "research":           "normal",
    "planning":           "normal",
    "data_analysis":      "normal",
    "file_management":    "normal",
    "task_management":    "normal",
    "creative_writing":   "normal",
    "coding":             "coding",
    "debugging":          "coding",
    "shell_command":      "coding",
    "math":               "coding",
    "skill_writing":      "slow",
    "agentic_task":       "slow",
    "reasoning":          "slow",
}

def route_to_model(intent: dict) -> dict:
    category = intent.get("category", "general_chat")
    tier = CATEGORY_TIER.get(category, "fast")
    installed = get_installed_models()

    for model in TIER_PREFERENCES.get(tier, TIER_PREFERENCES["fast"]):
        if model in installed:
            return {"model": model, "tier": tier, "latency": tier}

    # Last resort — whatever is installed
    if installed:
        return {"model": installed[0], "tier": "fast", "latency": "fast"}

    return {"model": "llama3.2:3b", "tier": "fast", "latency": "fast"}

def get_fallback(model: str) -> list:
    installed = get_installed_models()
    return [m for m in installed if m != model]

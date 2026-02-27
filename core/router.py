"""
core/router.py
Routes tasks to the best available model.
14B models are used freely for agentic/slow tasks.
"""

# Full routing table
# latency: instant (<3s), fast (<15s), normal (<60s), slow (ok for agentic)
MODEL_MAP = {
    # ── Instant: classifier pipeline (0.5b stays hot in RAM) ──
    "intent_classification":  {"model": "qwen2.5:0.5b",          "latency": "instant"},
    "prompt_rewriting":       {"model": "qwen2.5:0.5b",          "latency": "instant"},
    "sentiment_analysis":     {"model": "qwen2.5:0.5b",          "latency": "instant"},
    "safety_check":           {"model": "qwen2.5:0.5b",          "latency": "instant"},

    # ── Fast: simple, well-scoped tasks ──
    "general_chat":           {"model": "llama3.2:3b",           "latency": "fast"},
    "summarization":          {"model": "llama3.2:3b",           "latency": "fast"},
    "task_management":        {"model": "llama3.2:3b",           "latency": "fast"},
    "translation":            {"model": "qwen2.5:3b",            "latency": "fast"},

    # ── Normal: capable 8B ──
    "web_search":             {"model": "llama3.1:8b",           "latency": "normal"},
    "image_description":      {"model": "llava:7b",              "latency": "normal"},

    # ── Slow / Agentic: 14B for anything that matters ──
    "coding":                 {"model": "qwen2.5-coder:14b",     "latency": "slow"},
    "debugging":              {"model": "qwen2.5-coder:14b",     "latency": "slow"},
    "shell_command":          {"model": "qwen2.5-coder:14b",     "latency": "slow"},
    "skill_writing":          {"model": "qwen2.5-coder:14b",     "latency": "slow"},
    "structured_output":      {"model": "qwen2.5-coder:14b",     "latency": "slow"},
    "file_management":        {"model": "qwen2.5-coder:14b",     "latency": "slow"},
    "data_analysis":          {"model": "qwen2.5-coder:14b",     "latency": "slow"},
    "math":                   {"model": "deepseek-r1:14b",       "latency": "slow"},
    "reasoning":              {"model": "deepseek-r1:14b",       "latency": "slow"},
    "error_recovery":         {"model": "deepseek-r1:14b",       "latency": "slow"},
    "planning":               {"model": "phi4:14b",              "latency": "slow"},
    "research":               {"model": "qwen2.5:14b",           "latency": "slow"},
    "creative_writing":       {"model": "qwen2.5:14b",           "latency": "slow"},
    "agentic_task":           {"model": "qwen2.5:14b",           "latency": "slow"},
    "screenshot_analysis":    {"model": "llama3.2-vision:11b",   "latency": "slow"},
}

# Fallback chains: if primary OOMs or fails, try these in order
FALLBACK_CHAINS = {
    "qwen2.5-coder:14b":   ["qwen2.5-coder:7b",  "llama3.1:8b",  "llama3.2:3b"],
    "deepseek-r1:14b":     ["deepseek-r1:7b",     "qwen2.5:7b",   "llama3.1:8b"],
    "phi4:14b":            ["mistral-nemo:12b",   "mistral:7b",   "llama3.1:8b"],
    "qwen2.5:14b":         ["qwen2.5:7b",         "llama3.1:8b",  "llama3.2:3b"],
    "llama3.2-vision:11b": ["llava:7b",           "llama3.2:3b"],
    "llama3.1:8b":         ["llama3.2:3b"],
    "llava:7b":            ["llama3.2:3b"],
}

DEFAULT = {"model": "llama3.1:8b", "latency": "normal"}


def route_to_model(intent: dict) -> dict:
    category = intent.get("category", "general_chat")
    return MODEL_MAP.get(category, DEFAULT)


def get_fallback(model: str) -> list:
    return FALLBACK_CHAINS.get(model, ["llama3.1:8b", "llama3.2:3b"])

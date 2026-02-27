"""
core/token_budget.py
Adaptive token budget to prevent 20-minute runaway responses on 14B models.
"""

BASE_BUDGETS = {
    "instant":  150,
    "fast":     512,
    "normal":  1024,
    "slow":    2048,
}

# Tasks that need more tokens even on slow models
EXPANSIVE_TASKS = {
    "skill_writing":    4096,
    "coding":          4096,
    "research":        3000,
    "data_analysis":   3000,
    "creative_writing":3000,
    "planning":        2500,
    "debugging":       3000,
    "agentic_task":    3000,
}


def get_token_budget(latency_tier: str, task_type: str) -> int:
    # Check for explicitly expansive task
    if task_type in EXPANSIVE_TASKS:
        return EXPANSIVE_TASKS[task_type]

    return BASE_BUDGETS.get(latency_tier, 1024)

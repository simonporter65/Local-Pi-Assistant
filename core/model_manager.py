"""
core/model_manager.py

Fixes two latency problems:
  1. Cold 14B load (20-40s): keepalive pings keep recently-used models warm.
  2. Over-routing to 14B: try 8B first, escalate only if it signals need.

The escalation signal: if the 8B model outputs "ESCALATE:" it's saying
"I need a bigger model for this." The executor catches it and re-runs on 14B.
This means typical tasks (most of them) never touch 14B at all.

Also fixes heartbeat contention: background tasks pin to 3B only,
leaving 7B/14B free for the user.
"""

import asyncio
import ollama
import time
from typing import Optional

# Models that can signal escalation to a larger model
ESCALATION_SIGNAL = "ESCALATE:"

# How long to keep a model warm after last use (seconds)
KEEPALIVE_TTL = 8 * 60  # 8 minutes

# Ping payload — minimal tokens to keep model loaded
KEEPALIVE_PROMPT = "."


class ModelManager:
    """
    Tracks model usage and keeps recently-used models warm via background pings.
    """
    def __init__(self):
        self._last_used: dict[str, float] = {}
        self._keepalive_task: Optional[asyncio.Task] = None

    def record_use(self, model: str):
        self._last_used[model] = time.time()

    def is_warm(self, model: str) -> bool:
        last = self._last_used.get(model, 0)
        return (time.time() - last) < KEEPALIVE_TTL

    async def start_keepalive_loop(self):
        """Run as an asyncio background task alongside heartbeat."""
        while True:
            await asyncio.sleep(60)  # Check every minute
            now = time.time()
            for model, last_used in list(self._last_used.items()):
                age = now - last_used
                # Ping if model was used recently and is approaching TTL expiry
                if 0 < age < KEEPALIVE_TTL and age > (KEEPALIVE_TTL - 90):
                    await self._ping(model)

    async def _ping(self, model: str):
        """Send a minimal prompt to keep the model loaded in Ollama."""
        try:
            await asyncio.to_thread(
                ollama.generate,
                model=model,
                prompt=KEEPALIVE_PROMPT,
                options={"num_predict": 1, "num_ctx": 64},
            )
            self._last_used[model] = time.time()
            print(f"[MODEL_MGR] Keepalive ping sent to {model}")
        except Exception as e:
            print(f"[MODEL_MGR] Keepalive ping failed for {model}: {e}")


# ── 8B-first escalation routing ───────────────────────────────────────────────

# Map from 8B candidate → 14B escalation target
ESCALATION_MAP = {
    "llama3.1:8b":        "qwen2.5:14b",
    "qwen2.5:7b":         "qwen2.5:14b",
    "deepseek-r1:7b":     "deepseek-r1:14b",
    "qwen2.5-coder:7b":   "qwen2.5-coder:14b",
    "mistral:7b":         "qwen2.5:14b",
}

# Categories that ALWAYS go to 14B (skip 8B-first)
ALWAYS_14B = {
    "skill_writing",   # needs perfect code
    "error_recovery",  # critical correctness
}

# Categories that NEVER need 14B (stay on 3B/8B)
NEVER_14B = {
    "general_chat",
    "summarization",
    "translation",
    "sentiment_analysis",
    "structured_output",
}

ESCALATION_SYSTEM_ADDENDUM = """
IMPORTANT: If this task genuinely requires capabilities beyond what you have,
output exactly: ESCALATE: <one sentence explaining what you need>
Only escalate if truly necessary — most tasks you can handle directly.
"""


def get_model_for_category(category: str, needs_tools: bool) -> dict:
    """
    Returns routing decision: model, escalation_target, tier.
    Prefers 8B over 14B where possible.
    """
    if category in ALWAYS_14B:
        return {"model": "qwen2.5-coder:14b", "escalation_target": None, "tier": "14b_direct"}

    if category in NEVER_14B:
        return {"model": "llama3.2:3b", "escalation_target": None, "tier": "3b"}

    if category in ("coding", "debugging", "shell_command", "math", "reasoning"):
        # Try 8B coder first — escalate to 14B only if it asks
        return {
            "model": "qwen2.5-coder:7b",
            "escalation_target": "qwen2.5-coder:14b",
            "tier": "8b_with_escalation",
        }

    if category in ("research", "planning", "data_analysis", "agentic_task"):
        return {
            "model": "llama3.1:8b",
            "escalation_target": "qwen2.5:14b",
            "tier": "8b_with_escalation",
        }

    if category in ("web_search", "task_management", "file_management"):
        return {"model": "llama3.1:8b", "escalation_target": None, "tier": "8b"}

    if category in ("creative_writing", "image_description"):
        return {"model": "llama3.2:3b", "escalation_target": "llama3.1:8b", "tier": "3b_with_escalation"}

    # Default
    return {"model": "llama3.2:3b", "escalation_target": "llama3.1:8b", "tier": "3b_with_escalation"}


def check_for_escalation(response_text: str) -> Optional[str]:
    """Returns escalation reason if model is asking for a bigger model, else None."""
    import re
    m = re.search(r"ESCALATE:\s*(.+)", response_text)
    return m.group(1).strip() if m else None


# ── Background model policy ───────────────────────────────────────────────────

BACKGROUND_MODEL = "llama3.2:3b"  # Always 3b for heartbeat — never compete with user
BACKGROUND_MODEL_FALLBACK = "qwen2.5:3b"
BACKGROUND_CTX = 4096
BACKGROUND_TOKENS = 1000


def get_background_model() -> dict:
    """Background tasks always use 3B to avoid competing with user's 14B."""
    return {
        "model": BACKGROUND_MODEL,
        "fallback": BACKGROUND_MODEL_FALLBACK,
        "num_ctx": BACKGROUND_CTX,
        "num_predict": BACKGROUND_TOKENS,
    }


# ── Token budget by tier ──────────────────────────────────────────────────────

TOKEN_BUDGETS = {
    "3b":                    512,
    "3b_with_escalation":    600,
    "8b":                   1024,
    "8b_with_escalation":   1200,
    "14b_direct":           2048,
}

CTX_BY_TIER = {
    "3b":                   4096,
    "3b_with_escalation":   4096,
    "8b":                   6144,
    "8b_with_escalation":   6144,
    "14b_direct":           8192,
}


def get_token_budget(tier: str) -> int:
    return TOKEN_BUDGETS.get(tier, 1024)

def get_num_ctx(tier: str) -> int:
    return CTX_BY_TIER.get(tier, 6144)


# ── History summarisation ─────────────────────────────────────────────────────

SUMMARY_THRESHOLD = 5500  # tokens — compress history when we hit this


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def maybe_summarise_history(messages: list, model_manager=None) -> list:
    """
    If conversation history is getting long, compress the middle.
    Keeps the first message (original task) and last 2 exchanges intact.
    Middle is replaced with a compact summary.
    """
    total = sum(estimate_tokens(m.get("content", "")) for m in messages)
    if total < SUMMARY_THRESHOLD:
        return messages  # Fine as-is

    if len(messages) < 6:
        return messages  # Too short to compress

    # Keep first 1 and last 4 messages, summarise the middle
    first = messages[:1]
    middle = messages[1:-4]
    last = messages[-4:]

    if not middle:
        return messages

    middle_text = "\n".join([
        f"{m['role'].upper()}: {m['content'][:200]}"
        for m in middle
    ])

    summary_msg = {
        "role": "user",
        "content": (
            f"[HISTORY SUMMARY — {len(middle)} earlier messages compressed]\n"
            f"Key actions taken so far:\n{middle_text[:800]}\n"
            f"[End of summary. Continuing from most recent exchange below.]"
        )
    }

    compressed = first + [summary_msg] + last
    new_total = sum(estimate_tokens(m.get("content", "")) for m in compressed)
    print(f"[CTX] History compressed: {total} → {new_total} tokens ({len(messages)} → {len(compressed)} messages)")
    return compressed

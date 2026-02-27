"""
core/pipeline_pre.py

REPLACES: classifier.py, rewriter.py, and the early extract_from_message() call.

Before: 3 serial qwen2.5:0.5b calls before the user sees a single token.
        classify_intent()     → 1-2s
        rewrite_prompt()      → 1-2s
        extract_from_message()→ 0.5s
        TOTAL: 2.5 - 4.5s dead time on EVERY message.

After:  1 call, one JSON blob, 1.0 - 1.8s.
        TOTAL: ~1.5s dead time.

Saving: ~2-3s per message, every message, forever.
"""

import json
import ollama
import re
from typing import Optional

CATEGORIES = [
    "general_chat", "coding", "debugging", "math", "reasoning",
    "summarization", "web_search", "data_analysis", "creative_writing",
    "translation", "planning", "shell_command", "file_management",
    "image_description", "screenshot_analysis", "task_management",
    "research", "skill_writing", "agentic_task", "error_recovery",
]

FACT_CATEGORIES = [
    "name", "location", "occupation", "interests", "family",
    "health", "schedule", "preferences", "goals", "projects",
    "skills", "technology",
]

MERGED_PRE_PROMPT = """You are a fast routing pre-processor. Given a user message, return ONE JSON object doing three jobs at once.

CATEGORIES: {categories}

CATEGORY HINTS:
- coding = write new code | debugging = fix broken code | shell_command = run system commands
- skill_writing = create new agent skill/tool | agentic_task = multi-step autonomous work
- research = deep investigation | web_search = quick factual lookup
- general_chat = conversation, questions, anything else

USER MESSAGE: {input}

Return ONLY this JSON (no markdown, no explanation):
{{
  "category": "<one category>",
  "confidence": <0.0-1.0>,
  "needs_tools": <true if web search, file ops, code exec needed>,
  "rewritten": "<rewrite to be clearer and more precise, or copy original if already clear>",
  "facts": [
    {{"category": "<{fact_cats}>", "fact": "<explicit fact about the user if stated>"}}
  ]
}}

facts array: only include if the message explicitly states something about the user (name, job, location, etc). Empty array [] if nothing extractable."""


def run_pre_pipeline(user_message: str) -> dict:
    """
    Single 0.5b call replacing classify + rewrite + extract.
    Returns dict with: category, confidence, needs_tools, rewritten, facts
    """
    # Very short messages don't need rewriting or extraction
    skip_heavy = len(user_message.split()) < 4

    try:
        resp = ollama.generate(
            model="qwen2.5:0.5b",
            prompt=MERGED_PRE_PROMPT.format(
                categories=", ".join(CATEGORIES),
                fact_cats="|".join(FACT_CATEGORIES),
                input=user_message[:400],
            ),
            options={
                "temperature": 0.1,
                "num_predict": 200,
                "num_ctx":     1200,
            }
        )
        text = resp["response"].strip()

        # Extract JSON — handle markdown fences if model adds them
        text = re.sub(r"```json?\s*|\s*```", "", text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            result = json.loads(match.group())

            # Validate category
            if result.get("category") not in CATEGORIES:
                result["category"] = _heuristic_category(user_message)

            # Ensure rewritten is sane
            rewritten = result.get("rewritten", "").strip()
            if not rewritten or len(rewritten) > len(user_message) * 5:
                result["rewritten"] = user_message

            # Ensure facts is a list
            if not isinstance(result.get("facts"), list):
                result["facts"] = []

            result["_source"] = "llm"
            return result

    except (json.JSONDecodeError, Exception):
        pass

    # Heuristic fallback — no LLM needed
    return {
        "category":    _heuristic_category(user_message),
        "confidence":  0.5,
        "needs_tools": _needs_tools(user_message),
        "rewritten":   user_message,
        "facts":       [],
        "_source":     "heuristic",
    }


def _heuristic_category(text: str) -> str:
    t = text.lower()
    rules = [
        (["write a skill", "new skill", "new tool", "create a skill"],     "skill_writing"),
        (["debug", "fix this", "fix the", "error:", "traceback", "exception"], "debugging"),
        (["write a ", "create a ", "build a ", "implement "],              "coding"),
        (["def ", "class ", "function(", "import "],                       "coding"),
        (["bash", "shell", "sudo ", "apt ", "pip install", "systemctl"],   "shell_command"),
        (["calculate", "solve", "integral", "derivative", "equation"],     "math"),
        (["search for", "look up", "find me", "what is the latest"],       "web_search"),
        (["summarize", "tldr", "summary", "shorten"],                      "summarization"),
        (["translate", "in french", "in spanish", "in german"],            "translation"),
        (["plan", "schedule", "roadmap", "steps to", "how do i"],          "planning"),
        (["research", "investigate", "deep dive", "tell me everything"],   "research"),
        (["screenshot", "what's on screen", "what do you see"],            "screenshot_analysis"),
        (["analyze", ".csv", "dataframe", "dataset", "graph"],             "data_analysis"),
    ]
    for keywords, category in rules:
        if any(kw in t for kw in keywords):
            return category
    return "general_chat"


def _needs_tools(text: str) -> bool:
    t = text.lower()
    tool_signals = [
        "search", "fetch", "download", "run", "execute", "install",
        "file", "read", "write", "open", "browse", "screenshot",
        "latest", "current", "today", "news", "weather", "price",
    ]
    return any(s in t for s in tool_signals)


# ── Backwards-compatible shims ────────────────────────────────────────────────
# Other modules can still call classify_intent() and rewrite_prompt() individually.
# They'll use the merged call internally and return their slice.

_last_pre: Optional[dict] = None
_last_input: Optional[str] = None


def classify_intent(user_input: str) -> dict:
    """Backwards-compatible. Returns intent dict."""
    global _last_pre, _last_input
    if _last_input != user_input:
        _last_pre = run_pre_pipeline(user_input)
        _last_input = user_input
    r = _last_pre
    return {
        "category":   r["category"],
        "confidence": r["confidence"],
        "subtask":    user_input[:60],
    }


def rewrite_prompt(user_input: str, intent: dict = None) -> str:
    """Backwards-compatible. Returns rewritten prompt."""
    global _last_pre, _last_input
    if _last_input != user_input:
        _last_pre = run_pre_pipeline(user_input)
        _last_input = user_input
    return _last_pre.get("rewritten", user_input)


def get_extracted_facts(user_input: str) -> list:
    """Returns facts extracted in the same merged call."""
    global _last_pre, _last_input
    if _last_input != user_input:
        _last_pre = run_pre_pipeline(user_input)
        _last_input = user_input
    return _last_pre.get("facts", [])

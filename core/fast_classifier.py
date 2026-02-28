"""
core/fast_classifier.py — Sub-millisecond heuristic classifier.
Falls back to 0.5b only for ambiguous messages.
"""
import re

CODING_WORDS = {"code", "function", "bug", "error", "python", "javascript", "script",
                "debug", "class", "import", "syntax", "compile", "programming", "html",
                "css", "sql", "bash", "terminal", "git", "api", "json", "xml"}

SEARCH_WORDS = {"search", "find", "look up", "google", "what is", "who is", "when did",
                "where is", "how does", "latest", "news", "current", "today", "price"}

TASK_WORDS = {"create", "make", "build", "write", "generate", "plan", "schedule",
              "remind", "task", "todo", "draft", "summarize", "analyse", "analyze"}

MATH_WORDS = {"calculate", "solve", "equation", "math", "formula", "compute", "integral",
              "derivative", "probability", "statistics"}

CREATIVE_WORDS = {"story", "poem", "write me", "creative", "fiction", "imagine", "invent"}

def fast_classify(message: str) -> dict:
    msg = message.lower().strip()
    words = set(re.findall(r'\b\w+\b', msg))

    # Very short messages are always general chat
    if len(msg) < 30 and not words & (CODING_WORDS | MATH_WORDS | SEARCH_WORDS):
        return {"category": "general_chat", "confidence": 0.95,
                "needs_tools": False, "rewritten": message,
                "facts": [], "_source": "heuristic"}

    if words & CODING_WORDS:
        return {"category": "coding", "confidence": 0.9,
                "needs_tools": False, "rewritten": message,
                "facts": [], "_source": "heuristic"}

    if words & MATH_WORDS:
        return {"category": "math", "confidence": 0.9,
                "needs_tools": False, "rewritten": message,
                "facts": [], "_source": "heuristic"}

    if words & SEARCH_WORDS or msg.endswith("?"):
        return {"category": "web_search", "confidence": 0.8,
                "needs_tools": True, "rewritten": message,
                "facts": [], "_source": "heuristic"}

    if words & TASK_WORDS:
        return {"category": "planning", "confidence": 0.85,
                "needs_tools": False, "rewritten": message,
                "facts": [], "_source": "heuristic"}

    if words & CREATIVE_WORDS:
        return {"category": "creative_writing", "confidence": 0.85,
                "needs_tools": False, "rewritten": message,
                "facts": [], "_source": "heuristic"}

    # Ambiguous — return None to signal fallback to 0.5b
    return None

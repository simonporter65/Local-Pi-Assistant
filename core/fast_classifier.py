"""
core/fast_classifier.py — Sub-millisecond heuristic classifier.
Falls back to 0.5b only for truly ambiguous messages.
"""
import re

CODING_WORDS = {"code", "function", "bug", "error", "python", "javascript", "script",
                "debug", "class", "import", "syntax", "compile", "programming", "html",
                "css", "sql", "bash", "terminal", "git", "api", "json", "xml"}

SEARCH_WORDS = {"search", "look up", "google", "latest", "news", "current", "price",
                "weather", "stock", "define", "meaning", "translate"}

TASK_WORDS = {"create", "make", "build", "generate", "plan", "schedule",
              "remind", "task", "todo", "draft", "summarize", "analyse", "analyze"}

MATH_WORDS = {"calculate", "solve", "equation", "math", "formula", "compute", "integral",
              "derivative", "probability", "statistics"}

CREATIVE_WORDS = {"story", "poem", "write me", "creative", "fiction", "imagine", "invent"}

# Phrases that are always conversational — never need 0.5b
CHAT_PHRASES = [
    "do you", "can you", "you ", "your ", "remember", "know about", "about me",
    "my name", "who am", "where do i", "what do i", "i live", "i am", "i'm",
    "pretty good", "not bad", "doing well", "how are", "still there", "you there",
    "what city", "what state", "what country", "tell me", "do you know",
    "what do you", "have you", "are you", "were you", "did you",
]

def fast_classify(message: str) -> dict:
    msg = message.lower().strip()
    words = set(re.findall(r'\b\w+\b', msg))

    # Conversational phrases — always general chat
    if any(p in msg for p in CHAT_PHRASES):
        return {"category": "general_chat", "confidence": 0.95,
                "needs_tools": False, "rewritten": message,
                "facts": [], "_source": "heuristic"}

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

    if words & SEARCH_WORDS:
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

    # Default to general_chat rather than falling back to 0.5b
    return {"category": "general_chat", "confidence": 0.7,
            "needs_tools": False, "rewritten": message,
            "facts": [], "_source": "heuristic_default"}

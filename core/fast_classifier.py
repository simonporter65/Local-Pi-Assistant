"""
core/fast_classifier.py — Sub-millisecond heuristic classifier.
Falls back to general_chat for ambiguous messages rather than mis-routing.
"""
import re

# Specific programming languages and tools — rarely ambiguous
CODING_WORDS = {
    "python", "javascript", "typescript", "golang", "rust", "kotlin",
    "dockerfile", "kubernetes", "terraform", "webpack", "pytest",
    "async", "await", "recursion", "algorithm", "refactor",
    "middleware", "endpoint", "webhook", "api", "json", "yaml",
    "regex", "stdlib", "virtualenv", "dependencies",
}

# Code syntax phrases — unambiguous
CODING_PHRASES = [
    "def ", "class ", "import ", "function(", "() =>", "```",
    "#!/", "syntax error", "stack trace", "null pointer",
    "undefined is not", "cannot read property", "index out of range",
    "traceback (most recent", "keyerror", "typeerror", "valueerror",
    "npm install", "pip install", "git clone", "git commit",
]

SEARCH_WORDS = {
    "search", "google", "latest", "news", "current", "price",
    "weather", "stock", "define", "meaning", "translate",
    "who is", "what is", "when did", "where is",
}

TASK_WORDS = {
    "create", "build", "generate", "plan", "schedule",
    "remind", "task", "todo", "draft", "summarize", "analyse", "analyze",
}

MATH_WORDS = {
    "calculate", "solve", "equation", "formula", "compute", "integral",
    "derivative", "probability", "statistics", "percentage", "convert",
}

CREATIVE_WORDS = {"poem", "story", "fiction", "imagine", "invent", "creative"}

AGENTIC_WORDS = {
    "screenshot", "scrape", "automate", "browser",
    "download", "extract", "monitor",
}

AGENTIC_PHRASES = [
    "browse ", "visit ", "open the ", "go to ", "navigate to ",
    "click on", "fill in", "fill out", "submit the form",
    "http://", "https://", ".com/", ".org/", ".net/",
    "the website", "this website", "this page", "this url",
    "check the site", "check this link",
]

# Personality change requests — detected before general chat phrases
# Note: phrases must be specific enough to avoid false positives on normal chat.
PERSONALITY_CHANGE_PHRASES = [
    # Trait-specific adjustments
    "be funnier", "be warmer", "be sassier", "be wittier", "be cheekier",
    "be more concise", "be more verbose", "be more creative", "be more chaotic",
    "be more humorous", "be more playful", "be more serious", "be more casual",
    "be less verbose", "be less formal", "be less serious",
    "more humorous", "more playful", "less verbose",
    # Name changes
    "change your name", "your name is now", "rename yourself", "call yourself",
    "make yourself more ", "make yourself less ",
    # Explicit personality ops
    "adjust your personality", "update your personality", "change your personality",
    "reset your personality", "set your personality",
    # Queries about personality
    "what's your personality", "what are your settings", "show your settings",
    "what is your personality", "show me your personality",
    "your humor level", "your warmth level", "your verbosity", "your sass level",
]

# Explicit skill invocation — "use the X skill to ..." → always agentic
# Compiled once at module level for fast_classify() performance
_SKILL_INVOKE_RE = re.compile(r'\buse\b.{0,30}\bskill\b', re.I)

# Messages that are conversational — skip expensive routing
CHAT_PHRASES = [
    "do you", "can you", "you ", "your ", "remember", "know about",
    "my name", "who am", "where do i", "what do i", "i live", "i am", "i'm",
    "pretty good", "not bad", "doing well", "how are", "still there", "you there",
    "what city", "what state", "what country", "tell me", "do you know",
    "what do you", "have you", "are you", "were you", "did you",
    "about me", "i work", "i like", "i love", "i hate", "i use",
]


def fast_classify(message: str) -> dict:
    msg = message.lower().strip()
    words = set(re.findall(r'\b\w+\b', msg))

    # Personality change / query — highest priority (before generic chat check)
    if any(p in msg for p in PERSONALITY_CHANGE_PHRASES):
        return {"category": "personality_change", "confidence": 0.95,
                "needs_tools": True, "rewritten": message,
                "facts": [], "_source": "heuristic"}

    # Explicit skill invocation — "use the browser skill to ...", "use X skill" etc.
    # Must fire before CHAT_PHRASES and SEARCH_WORDS to avoid mis-routing
    # e.g. "use the browser skill to check the weather" has "weather" (SEARCH_WORDS)
    # but the user's intent is clearly agentic, not a DuckDuckGo lookup.
    if _SKILL_INVOKE_RE.search(msg):
        return {"category": "agentic_task", "confidence": 0.95,
                "needs_tools": True, "rewritten": message,
                "facts": [], "_source": "heuristic"}

    # Conversational phrases — always general chat
    if any(p in msg for p in CHAT_PHRASES):
        return {"category": "general_chat", "confidence": 0.95,
                "needs_tools": False, "rewritten": message,
                "facts": [], "_source": "heuristic"}

    # Very short messages are conversational
    if len(msg) < 30 and not words & (CODING_WORDS | MATH_WORDS | SEARCH_WORDS):
        return {"category": "general_chat", "confidence": 0.95,
                "needs_tools": False, "rewritten": message,
                "facts": [], "_source": "heuristic"}

    # Coding — require explicit language names OR actual code syntax phrases
    if words & CODING_WORDS or any(p in msg for p in CODING_PHRASES):
        cat = "debugging" if any(p in msg for p in [
            "error", "bug", "fix", "broken", "crash", "fail", "traceback", "exception"
        ]) else "coding"
        return {"category": cat, "confidence": 0.9,
                "needs_tools": False, "rewritten": message,
                "facts": [], "_source": "heuristic"}

    if words & MATH_WORDS:
        return {"category": "math", "confidence": 0.9,
                "needs_tools": False, "rewritten": message,
                "facts": [], "_source": "heuristic"}

    if words & SEARCH_WORDS:
        return {"category": "web_search", "confidence": 0.85,
                "needs_tools": True, "rewritten": message,
                "facts": [], "_source": "heuristic"}

    if words & TASK_WORDS:
        return {"category": "planning", "confidence": 0.8,
                "needs_tools": False, "rewritten": message,
                "facts": [], "_source": "heuristic"}

    if words & CREATIVE_WORDS:
        return {"category": "creative_writing", "confidence": 0.85,
                "needs_tools": False, "rewritten": message,
                "facts": [], "_source": "heuristic"}

    # Browser / agentic task detection — URLs or explicit browse/automate intent
    if words & AGENTIC_WORDS or any(p in msg for p in AGENTIC_PHRASES):
        return {"category": "web_browsing", "confidence": 0.85,
                "needs_tools": True, "rewritten": message,
                "facts": [], "_source": "heuristic"}

    return {"category": "general_chat", "confidence": 0.7,
            "needs_tools": False, "rewritten": message,
            "facts": [], "_source": "heuristic_default"}

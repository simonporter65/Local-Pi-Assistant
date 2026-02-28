"""
autonomous/curiosity.py — ARC asks questions to fill profile gaps.
Called by heartbeat during idle time.
"""
import json

CURIOSITY_PROMPT = """You are {name}, getting to know your user better.

WHAT YOU KNOW SO FAR:
{known_facts}

WHAT YOU ARE UNCERTAIN ABOUT:
{gaps}

Generate ONE natural, friendly question to ask the user to fill in a gap.
Rules:
- Max 20 words
- Sound curious, not like a form
- Never ask about something already known
- Start with a conversational opener
- Return ONLY the question, nothing else
"""

GAPS_TO_CHECK = [
    ("location", "where they live or are based"),
    ("occupation", "what they do for work"),
    ("timezone", "what timezone they are in"),
    ("interests", "what their hobbies or interests are"),
    ("goals", "what they are trying to achieve"),
]

def get_curiosity_question(name: str, user_model, ollama_model: str = "qwen2.5:0.5b") -> str | None:
    """Returns a question to ask the user, or None if nothing needed."""
    import ollama

    # Get known facts
    try:
        facts = user_model.memory.db.execute(
            "SELECT category, fact FROM user_facts WHERE confidence > 0.6"
        ).fetchall()
    except Exception:
        return None

    known_categories = {f[0] for f in facts}
    known_facts = "\n".join([f"- {f[0]}: {f[1]}" for f in facts]) or "Nothing yet."

    # Find gaps
    gaps = [desc for cat, desc in GAPS_TO_CHECK if cat not in known_categories]
    if not gaps:
        return None  # Already know enough

    gaps_str = "\n".join([f"- {g}" for g in gaps[:3]])

    try:
        resp = ollama.generate(
            model=ollama_model,
            prompt=CURIOSITY_PROMPT.format(
                name=name,
                known_facts=known_facts[:400],
                gaps=gaps_str,
            ),
            options={"temperature": 0.8, "num_predict": 40, "num_ctx": 512}
        )
        question = resp["response"].strip()
        if len(question) > 5 and "?" in question:
            return question
    except Exception:
        pass

    return None

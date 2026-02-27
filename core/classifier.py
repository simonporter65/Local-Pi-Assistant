"""
core/classifier.py
Intent classification using qwen2.5:0.5b — stays hot in RAM.
"""

import ollama
import json
import re

CATEGORIES = [
    "general_chat", "coding", "debugging", "math", "reasoning",
    "summarization", "web_search", "data_analysis", "creative_writing",
    "translation", "planning", "shell_command", "file_management",
    "image_description", "screenshot_analysis", "task_management",
    "research", "sentiment_analysis", "structured_output",
    "skill_writing", "error_recovery", "safety_check", "agentic_task",
]

CLASSIFIER_PROMPT = """Classify this user input into exactly one category. Return JSON only, no explanation.

Categories: {categories}

Guidance:
- "coding" = write new code
- "debugging" = fix existing code or errors
- "shell_command" = run system commands, install packages
- "skill_writing" = create a new agent capability/tool
- "agentic_task" = multi-step autonomous tasks
- "research" = deep multi-source investigation
- "web_search" = quick lookup or current info

Input: {input}

JSON format: {{"category": "<category>", "confidence": <0.0-1.0>, "subtask": "<10 word description>"}}"""


def classify_intent(user_input: str) -> dict:
    try:
        response = ollama.generate(
            model="qwen2.5:0.5b",
            prompt=CLASSIFIER_PROMPT.format(
                categories=", ".join(CATEGORIES),
                input=user_input[:500],  # truncate huge inputs
            ),
            options={
                "temperature": 0.1,
                "num_predict": 120,
                "num_ctx": 1024,
            }
        )

        text = response["response"].strip()

        # Robust JSON extraction
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            # Validate category
            if result.get("category") not in CATEGORIES:
                result["category"] = "general_chat"
            return result

    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f"[CLASSIFIER ERROR] {e}")

    # Fallback: keyword heuristics
    return _heuristic_classify(user_input)


def _heuristic_classify(text: str) -> dict:
    text_lower = text.lower()

    heuristics = [
        (["write a skill", "create a skill", "new skill", "new tool"],   "skill_writing"),
        (["debug", "fix this", "error:", "traceback", "exception"],      "debugging"),
        (["def ", "class ", "function", "import ", "#!/"],               "coding"),
        (["bash", "shell", "terminal", "sudo", "apt", "pip install"],    "shell_command"),
        (["math", "calculate", "equation", "integral", "proof"],         "math"),
        (["search for", "look up", "what is the latest", "current"],     "web_search"),
        (["summarize", "tldr", "summary of"],                            "summarization"),
        (["translate", "in french", "in spanish", "in german"],          "translation"),
        (["plan", "schedule", "roadmap", "steps to"],                    "planning"),
        (["analyze", "csv", "dataframe", "dataset", "statistics"],       "data_analysis"),
        (["write a story", "write a poem", "creative"],                  "creative_writing"),
        (["screenshot", "what's on screen", "capture screen"],           "screenshot_analysis"),
        (["image", "photo", "picture", "describe this"],                 "image_description"),
        (["research", "investigate", "deep dive", "comprehensive"],      "research"),
    ]

    for keywords, category in heuristics:
        if any(kw in text_lower for kw in keywords):
            return {"category": category, "confidence": 0.6, "subtask": text[:60]}

    return {"category": "general_chat", "confidence": 0.5, "subtask": text[:60]}

"""
core/validator.py
Validates task output before accepting it as complete.
"""

import re

# Minimum output lengths by category (chars)
MIN_LENGTH = {
    "general_chat":      20,
    "coding":           100,
    "debugging":         50,
    "math":              20,
    "reasoning":         50,
    "summarization":     50,
    "web_search":        50,
    "data_analysis":     50,
    "creative_writing":  80,
    "translation":       10,
    "planning":          80,
    "shell_command":     10,
    "file_management":   10,
    "research":         150,
    "skill_writing":    100,
    "structured_output": 10,
    "agentic_task":      30,
}

# Phrases that indicate the model gave up or refused
FAILURE_PHRASES = [
    "i cannot", "i can't", "i'm unable", "i am unable",
    "as an ai", "i don't have access", "i cannot access",
    "i'm sorry, but", "unfortunately, i cannot",
    "i cannot complete this task",
]

# Phrases that suggest an incomplete response
INCOMPLETE_PHRASES = [
    "to be continued", "in the next step", "i will now",
    "please wait", "working on it",
]


def validate_result(result: dict, intent: dict) -> bool:
    """
    Returns True if the result is acceptable.
    Sets result['failure_reason'] if not.
    """
    if not result:
        result["failure_reason"] = "null result"
        return False

    output = result.get("output", "").strip()
    category = intent.get("category", "general_chat")

    # Must have some output
    if not output:
        result["failure_reason"] = "empty output"
        return False

    # Must meet minimum length
    min_len = MIN_LENGTH.get(category, 20)
    if len(output) < min_len:
        result["failure_reason"] = f"output too short ({len(output)} < {min_len})"
        return False

    output_lower = output.lower()

    # Check for refusal phrases
    for phrase in FAILURE_PHRASES:
        if phrase in output_lower:
            result["failure_reason"] = f"model refused: '{phrase}'"
            return False

    # Check for incompleteness
    for phrase in INCOMPLETE_PHRASES:
        if phrase in output_lower:
            result["failure_reason"] = f"incomplete response: '{phrase}'"
            return False

    # Category-specific checks
    if category == "coding" or category == "debugging":
        # Should have some code-like content
        has_code = (
            "```" in output or
            "def " in output or
            "class " in output or
            "import " in output or
            re.search(r"[a-zA-Z_]\w*\s*\(", output)  # function call pattern
        )
        if not has_code and len(output) < 200:
            result["failure_reason"] = "coding task produced no code"
            return False

    if category == "skill_writing":
        has_skill_structure = (
            "DESCRIPTION" in output and
            "def run" in output
        )
        if not has_skill_structure:
            result["failure_reason"] = "skill_writing task produced no valid skill structure"
            return False

    if category == "math":
        has_number = bool(re.search(r"\d", output))
        if not has_number:
            result["failure_reason"] = "math task produced no numbers"
            return False

    return True

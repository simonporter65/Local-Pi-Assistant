"""
core/rewriter.py
Rewrites user prompts for clarity and task-appropriateness.
Uses qwen2.5:0.5b — fast, stays in RAM.
"""

import ollama

REWRITE_PROMPT = """Rewrite this user prompt to be clearer and more effective for a {category} task.
Keep the original intent exactly. Be more specific and actionable.
Return ONLY the rewritten prompt, no explanation, no preamble.

Original: {prompt}

Rewritten:"""


def rewrite_prompt(prompt: str, intent: dict) -> str:
    category = intent.get("category", "general")

    # Don't rewrite very short prompts — they're probably already clear
    if len(prompt) < 30:
        return prompt

    try:
        response = ollama.generate(
            model="qwen2.5:0.5b",
            prompt=REWRITE_PROMPT.format(category=category, prompt=prompt[:600]),
            options={
                "temperature": 0.3,
                "num_predict": 300,
                "num_ctx": 1024,
            }
        )
        rewritten = response["response"].strip()

        # Sanity check: rewrite shouldn't be empty or wildly longer
        if rewritten and len(rewritten) < len(prompt) * 4:
            return rewritten

    except Exception as e:
        print(f"[REWRITER] Error: {e}, using original")

    return prompt

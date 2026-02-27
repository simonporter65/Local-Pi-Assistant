"""
skills/skill_writer.py
Writes new Python skill files using qwen2.5-coder:14b.
This is the agent's self-improvement mechanism.
"""

DESCRIPTION = (
    "Write a new agent skill (Python module). "
    "Args: skill_name (str), description (str), example_usage (str, optional)"
)

import ollama
import os
import re
import sys

SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))

WRITE_PROMPT = """Write a complete Python skill module for a Raspberry Pi 5 agentic system.

Skill name: {name}
What it does: {description}
Example usage: {example}

STRICT REQUIREMENTS:
1. File must have a top-level string: DESCRIPTION = "one-line description. Args: arg1 (type), ..."
2. File must have: def run(**kwargs) -> str
3. run() must ALWAYS return a string
4. Handle ALL exceptions — never let exceptions propagate
5. Use only: stdlib, requests, beautifulsoup4, pillow, numpy, subprocess, ollama
6. Be production-quality — handle edge cases
7. Add helpful comments

EXAMPLE STRUCTURE:
```python
\"\"\"
skills/example.py
Brief description.
\"\"\"

DESCRIPTION = "What this does. Args: query (str), max_results (int, default 5)"

import os

def run(query: str, max_results: int = 5) -> str:
    try:
        # implementation
        return result
    except Exception as e:
        return f"ERROR: {{e}}"
```

Return ONLY valid Python code. No markdown fences. No explanation before or after."""


def run(skill_name: str, description: str, example_usage: str = "") -> str:
    # Sanitise name
    skill_name = re.sub(r"[^a-z0-9_]", "_", skill_name.lower().strip())
    skill_name = re.sub(r"_+", "_", skill_name).strip("_")

    if not skill_name:
        return "ERROR: Invalid skill name"

    skill_path = os.path.join(SKILLS_DIR, f"{skill_name}.py")

    if os.path.exists(skill_path):
        return f"Skill '{skill_name}' already exists at {skill_path}. Delete it first if you want to rewrite it."

    try:
        response = ollama.generate(
            model="qwen2.5-coder:14b",
            prompt=WRITE_PROMPT.format(
                name=skill_name,
                description=description,
                example=example_usage or "No example provided.",
            ),
            options={
                "temperature": 0.2,
                "num_predict": 4096,
                "num_ctx": 4096,
            }
        )

        code = response["response"].strip()

        # Strip markdown fences if model added them anyway
        code = re.sub(r"^```python\s*", "", code)
        code = re.sub(r"^```\s*", "", code)
        code = re.sub(r"\s*```$", "", code).strip()

        # Validate: must have DESCRIPTION and run
        if "DESCRIPTION" not in code:
            return f"ERROR: Generated code missing DESCRIPTION. Try again with a clearer description."
        if "def run" not in code:
            return f"ERROR: Generated code missing def run(). Try again."

        # Write it
        with open(skill_path, "w", encoding="utf-8") as f:
            f.write(code)
            f.write("\n")

        # Quick syntax check
        try:
            compile(open(skill_path).read(), skill_path, "exec")
        except SyntaxError as e:
            os.remove(skill_path)
            return f"ERROR: Generated code has syntax error: {e}. Try again."

        return (
            f"✓ Skill '{skill_name}' written to {skill_path}\n"
            f"  Call 'reload' in the agent or use registry.reload() to activate it.\n"
            f"  Preview:\n" + "\n".join(code.split("\n")[:15]) + "\n..."
        )

    except Exception as e:
        return f"ERROR writing skill: {e}"

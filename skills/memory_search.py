"""
skills/memory_search.py
Let the agent query its own interaction history semantically.
"""

DESCRIPTION = (
    "Search past agent interactions by semantic similarity. "
    "Args: query (str), top_k (int, default 5)"
)

import sys
import os

# Add parent to path so we can import memory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(query: str, top_k: int = 5) -> str:
    try:
        from memory.store import AgentMemory
        memory = AgentMemory()
        results = memory.semantic_search(query, top_k=top_k)

        if not results:
            return "No relevant past interactions found."

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] Input: {r['input'][:80]}")
            lines.append(f"    Output: {r['output'][:150]}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"Memory search error: {e}"

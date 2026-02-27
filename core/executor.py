"""
core/executor.py
Runs the agentic tool-use loop.
Handles DeepSeek-R1 <think> blocks, skill calls, fallback chains, and OOM recovery.
"""

import ollama
import json
import re
import time
from typing import Callable, Optional

TOOL_USE_SYSTEM = """{system}"""

TOOL_USE_PROMPT = """Task: {prompt}

Remember:
- Use SKILL: {{"name": "...", "args": {{...}}}} to call tools
- Chain multiple skill calls as needed
- Output FINAL: <answer> when complete
- Never give up — try different approaches if one fails
"""


def execute_task(
    prompt: str,
    model: str,
    system: str,
    skills,
    memory,
    token_budget: int = 2048,
    on_skill_call: Optional[Callable] = None,
    on_skill_result: Optional[Callable] = None,
    max_tool_calls: int = 20,
) -> dict:
    """
    Run the full agentic loop with one model.
    Returns result dict with output, success, tool_calls, model keys.
    """
    messages = [
        {"role": "user", "content": TOOL_USE_PROMPT.format(prompt=prompt)}
    ]

    tool_calls = 0
    last_reply = ""
    thinking_log = []

    while tool_calls < max_tool_calls:
        try:
            t0 = time.time()
            response = ollama.chat(
                model=model,
                messages=messages,
                system=system,
                options={
                    "temperature": 0.7,
                    "num_predict": token_budget,
                    "num_ctx": 8192,
                    "num_gpu": 999,  # Ollama caps to available; maximises GPU layers
                }
            )
            elapsed = time.time() - t0

        except ollama.ResponseError as e:
            err = str(e).lower()
            if "out of memory" in err or "oom" in err:
                return {
                    "output": last_reply,
                    "success": False,
                    "failure_reason": f"OOM: {e}",
                    "tool_calls": tool_calls,
                    "model": model,
                    "thinking": thinking_log,
                }
            raise

        raw_reply = response["message"]["content"]
        last_reply = raw_reply

        # ── DeepSeek-R1 <think> block handling ──
        think_match = re.search(r"<think>(.*?)</think>", raw_reply, re.DOTALL)
        reply = raw_reply
        if think_match:
            think_text = think_match.group(1).strip()
            thinking_log.append(think_text)
            # Strip the think block from the reply we process
            reply = raw_reply.replace(think_match.group(0), "").strip()
            print(f"  [dim magenta]🧠 Thinking ({len(think_text)} chars)...[/dim magenta]")

        messages.append({"role": "assistant", "content": raw_reply})

        print(f"  [dim]{elapsed:.1f}s | {len(reply)} chars[/dim]")

        # ── Check for FINAL answer ──
        final_match = re.search(r"FINAL:\s*(.*)", reply, re.DOTALL)
        if final_match:
            final_output = final_match.group(1).strip()
            return {
                "output": final_output,
                "success": True,
                "tool_calls": tool_calls,
                "model": model,
                "thinking": thinking_log,
            }

        # ── Check for SKILL call ──
        skill_match = re.search(r"SKILL:\s*(\{.*?\})", reply, re.DOTALL)
        if skill_match:
            skill_result = _handle_skill_call(
                skill_match.group(1), skills, on_skill_call, on_skill_result
            )
            messages.append({
                "role": "user",
                "content": f"Skill result:\n{skill_result}\n\nContinue. Use more skills or output FINAL: when done."
            })
            tool_calls += 1
            continue

        # ── No FINAL or SKILL — nudge the model ──
        nudge_count = sum(1 for m in messages if m.get("content", "").startswith("Continue."))
        if nudge_count >= 3:
            # Model is stuck — force a final
            messages.append({
                "role": "user",
                "content": (
                    "You have not used any skills or given a final answer. "
                    "Either call a SKILL or output your best answer as:\nFINAL: <answer>"
                )
            })
        else:
            messages.append({
                "role": "user",
                "content": "Continue. Use a SKILL if you need information, or output FINAL: when done."
            })

    # Max tool calls reached — return what we have
    return {
        "output": _extract_best_output(last_reply),
        "success": False,
        "failure_reason": f"Max tool calls ({max_tool_calls}) reached",
        "tool_calls": tool_calls,
        "model": model,
        "thinking": thinking_log,
    }


def _handle_skill_call(
    json_str: str,
    skills,
    on_skill_call: Optional[Callable],
    on_skill_result: Optional[Callable],
) -> str:
    try:
        skill_call = json.loads(json_str)
        skill_name = skill_call.get("name", "")
        skill_args = skill_call.get("args", {})

        if on_skill_call:
            on_skill_call(skill_name, skill_args)

        result = skills.run(skill_name, **skill_args)
        result_str = str(result)

        # Trim massive outputs to avoid blowing context window
        if len(result_str) > 6000:
            result_str = result_str[:5700] + f"\n... [truncated, {len(result_str)} total chars]"

        if on_skill_result:
            on_skill_result(skill_name, result_str)

        return result_str

    except json.JSONDecodeError as e:
        return f"ERROR: Malformed SKILL JSON: {e}. Check your syntax and try again."
    except ValueError as e:
        return f"ERROR: {e}. Available skills: {skills.list_skill_names()}"
    except Exception as e:
        return f"ERROR in skill execution: {e}. Try a different approach."


def _extract_best_output(reply: str) -> str:
    """Pull the most useful content from a reply that didn't have FINAL:"""
    # Remove think blocks
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
    # Remove SKILL: lines
    reply = re.sub(r"^SKILL:.*$", "", reply, flags=re.MULTILINE).strip()
    return reply or "No output generated."

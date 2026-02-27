#!/usr/bin/env python3
"""
Pi 5 Agentic Loop — Main Entry Point
Infinite resourcefulness. Never gives up.
"""

import time
import sys
import json
import signal
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich import box

from core.classifier import classify_intent
from core.rewriter import rewrite_prompt
from core.router import route_to_model, get_fallback
from core.executor import execute_task
from core.validator import validate_result
from core.token_budget import get_token_budget
from skills.registry import SkillRegistry
from memory.store import AgentMemory

console = Console()

# ─────────────────────────────────────────────
# SYSTEM PROMPT — The soul of the agent
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an infinitely resourceful autonomous agent running on a Raspberry Pi 5 with 8GB RAM and 2TB NVMe storage.

CORE DIRECTIVES — these are absolute and non-negotiable:

1. NEVER GIVE UP. If approach A fails, try B, C, D... You have unlimited retries.
2. ALWAYS USE SKILLS. You have tools — use them. Don't guess when you can search.
3. SELF-IMPROVE. If you need a capability you don't have, write a new skill using skill_writer.
4. VERIFY YOUR WORK. After completing a task, check the result makes sense.
5. THINK STEP BY STEP. Break complex problems into sub-problems. Solve each one.
6. USE PAST CONTEXT. Check what you've done before. Don't repeat failures.
7. BE CREATIVE. If the obvious path is blocked, find another way.
8. INSTALL WHAT YOU NEED. You can run bash. Install packages. Write scripts.
9. SEARCH BEFORE ASSUMING. The web has the answer. Use it.
10. REPORT PROGRESS. For long tasks, show intermediate results so the user knows you're working.

SKILL USAGE FORMAT:
To call a skill, output this EXACT format on its own line:
SKILL: {"name": "skill_name", "args": {"key": "value"}}

Wait for the result, then continue reasoning.

When your task is complete, output:
FINAL: <your complete answer here>

AVAILABLE MODELS (you are running on one of these, routed by task type):
- qwen2.5:0.5b    — classifier/router (fast)
- llama3.2:3b     — general chat, summaries
- llama3.1:8b     — web synthesis, research
- qwen2.5-coder:14b — coding, debugging, shell, skill writing (best)
- deepseek-r1:14b  — math, logic, deep reasoning (best)
- phi4:14b         — planning, instruction following (best)
- qwen2.5:14b      — research, general hard tasks (best)
- llava:7b / llama3.2-vision:11b — vision tasks

You are currently using: {model}
Task category: {category}
Available skills: {skills}
Relevant past context:
{context}
"""

# ─────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────

def print_banner():
    console.print(Panel.fit(
        "[bold cyan]Pi 5 Agentic Loop[/bold cyan]\n"
        "[dim]Infinitely resourceful. Never gives up.[/dim]\n"
        "[dim]Type 'quit' to exit, 'status' to see loaded models, 'history' for past tasks[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))

def print_routing_info(intent: dict, model: str, latency: str):
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column(style="dim")
    table.add_column(style="bold")
    table.add_row("Category", f"[cyan]{intent.get('category', '?')}[/cyan]")
    table.add_row("Confidence", f"{intent.get('confidence', 0):.0%}")
    table.add_row("Model", f"[yellow]{model}[/yellow]")
    table.add_row("Latency tier", f"[{'red' if latency == 'slow' else 'green'}]{latency}[/]")
    table.add_row("Subtask", f"[dim]{intent.get('subtask', '')}[/dim]")
    console.print(Panel(table, title="[dim]Routing[/dim]", border_style="dim", expand=False))

def print_result(result: dict, duration: float):
    output = result.get("output", "")
    model = result.get("model", "?")
    tool_calls = result.get("tool_calls", 0)
    success = result.get("success", False)

    status = "[green]✓ Success[/green]" if success else "[red]✗ Partial[/red]"

    console.print()
    console.print(Panel(
        Markdown(output),
        title=f"{status} | [dim]{model} | {tool_calls} tool calls | {duration:.1f}s[/dim]",
        border_style="green" if success else "red",
    ))

def print_skill_call(name: str, args: dict):
    args_str = ", ".join(f"{k}={repr(v)[:40]}" for k, v in args.items())
    console.print(f"  [dim cyan]⚙ {name}({args_str})[/dim cyan]")

def print_skill_result(name: str, result: str):
    preview = result[:120].replace('\n', ' ')
    console.print(f"  [dim green]↳ {preview}{'...' if len(result) > 120 else ''}[/dim green]")

def print_model_fallback(from_model: str, to_model: str, reason: str):
    console.print(f"  [yellow]⚡ Fallback: {from_model} → {to_model} ({reason})[/yellow]")

def print_retry(attempt: int, max_attempts: int, reason: str):
    console.print(f"  [dim red]↻ Retry {attempt}/{max_attempts}: {reason}[/dim red]")


# ─────────────────────────────────────────────
# Command handlers
# ─────────────────────────────────────────────

def handle_status():
    import subprocess
    result = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
    console.print(Panel(result.stdout or "No models currently loaded", title="Ollama Status"))

def handle_history(memory: AgentMemory, n: int = 10):
    rows = memory.db.execute(
        "SELECT timestamp, user_input, intent, model_used, success, duration_ms "
        "FROM interactions ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()

    table = Table(title="Recent Tasks", box=box.SIMPLE)
    table.add_column("Time", style="dim")
    table.add_column("Input")
    table.add_column("Model", style="yellow")
    table.add_column("OK", justify="center")
    table.add_column("ms", justify="right")

    for row in rows:
        ts = row[0][:16] if row[0] else "?"
        inp = row[1][:40] + ("..." if len(row[1]) > 40 else "")
        try:
            intent = json.loads(row[2])
            category = intent.get("category", "?")
        except Exception:
            category = str(row[2])[:15]
        model = (row[3] or "?")[:20]
        ok = "[green]✓[/green]" if row[4] else "[red]✗[/red]"
        ms = str(row[5]) if row[5] else "?"
        table.add_row(ts, inp, model, ok, ms)

    console.print(table)


# ─────────────────────────────────────────────
# Core agent loop
# ─────────────────────────────────────────────

def run_agent(
    user_input: str,
    registry: SkillRegistry,
    memory: AgentMemory,
    max_retries: int = 8,
):
    t_start = time.time()

    # ── 1. Pre-processing pipeline (all on 0.5b, fast) ──
    with console.status("[dim]Classifying intent...[/dim]", spinner="dots"):
        intent = classify_intent(user_input)

    with console.status("[dim]Rewriting prompt...[/dim]", spinner="dots"):
        rewritten = rewrite_prompt(user_input, intent)

    route = route_to_model(intent)
    model = route["model"]
    latency = route["latency"]

    print_routing_info(intent, model, latency)

    # ── 2. Build system prompt with context injection ──
    past = memory.semantic_search(user_input, top_k=3)
    context_str = "\n".join([
        f"- Previously: '{p['input'][:60]}' → '{p['output'][:100]}'"
        for p in past
    ]) or "None yet."

    system = SYSTEM_PROMPT.format(
        model=model,
        category=intent.get("category", "general"),
        skills=registry.list_skills(),
        context=context_str,
    )

    token_budget = get_token_budget(latency, intent.get("category", ""))

    # ── 3. Agentic retry loop ──
    attempt = 0
    result = None
    models_to_try = [model] + get_fallback(model)

    while attempt < max_retries:
        attempt += 1

        # Cycle through fallback models after repeated failures
        current_model = models_to_try[min(attempt - 1, len(models_to_try) - 1)]
        if current_model != model and attempt > 1:
            print_model_fallback(model, current_model, "previous attempt failed")

        console.print(f"\n[dim]Attempt {attempt}/{max_retries} | {current_model}[/dim]")

        try:
            result = execute_task(
                prompt=rewritten,
                model=current_model,
                system=system,
                skills=registry,
                memory=memory,
                token_budget=token_budget,
                on_skill_call=print_skill_call,
                on_skill_result=print_skill_result,
            )

            if validate_result(result, intent):
                break
            else:
                reason = result.get("failure_reason", "validation failed")
                print_retry(attempt, max_retries, reason)
                # Feed failure back into next attempt
                rewritten = (
                    f"Previous attempt failed: {reason}\n"
                    f"Failed output: {result.get('output', '')[:300]}\n"
                    f"Original task: {user_input}\n"
                    f"Try a completely different approach. "
                    f"Use different skills, search the web, or write a new skill."
                )

        except KeyboardInterrupt:
            console.print("\n[yellow]Task interrupted by user.[/yellow]")
            result = {"output": "Interrupted.", "success": False, "tool_calls": 0, "model": current_model}
            break

        except Exception as e:
            reason = str(e)
            print_retry(attempt, max_retries, reason)
            rewritten = (
                f"Previous attempt threw exception: {reason}\n"
                f"Original task: {user_input}\n"
                f"Try a completely different approach."
            )

    duration = time.time() - t_start

    # ── 4. Log to memory ──
    memory.log_interaction(
        user_input=user_input,
        intent=intent,
        model=result.get("model", model) if result else model,
        output=result.get("output", "") if result else "",
        success=result.get("success", False) if result else False,
        tool_calls=result.get("tool_calls", 0) if result else 0,
        duration_ms=int(duration * 1000),
    )

    print_result(result or {"output": "No result.", "success": False}, duration)
    return result


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    print_banner()

    registry = SkillRegistry()
    memory = AgentMemory()

    console.print(f"[dim]Loaded {len(registry.skills)} skills: {', '.join(registry.skills.keys())}[/dim]\n")

    # Graceful shutdown
    def handle_sigterm(sig, frame):
        console.print("\n[yellow]Agent shutting down gracefully.[/yellow]")
        sys.exit(0)
    signal.signal(signal.SIGTERM, handle_sigterm)

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]>[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye.[/yellow]")
            break

        if not user_input:
            continue

        lower = user_input.lower()

        if lower in ("quit", "exit", "bye"):
            console.print("[yellow]Goodbye.[/yellow]")
            break
        elif lower == "status":
            handle_status()
        elif lower == "history":
            handle_history(memory)
        elif lower.startswith("history "):
            try:
                n = int(lower.split()[1])
                handle_history(memory, n)
            except ValueError:
                handle_history(memory)
        elif lower == "skills":
            console.print(Panel(registry.list_skills(), title="Available Skills"))
        elif lower == "reload":
            registry.reload()
            console.print(f"[green]Skills reloaded. {len(registry.skills)} available.[/green]")
        else:
            run_agent(user_input, registry, memory)


if __name__ == "__main__":
    main()

"""
autonomous/heartbeat.py

The heartbeat is the agent's autonomous background loop.
It fires on a schedule, picks tasks from the queue, executes them,
and generates new tasks based on what it learns.

Key design:
- Runs in a separate asyncio task (not a thread)
- Pauses immediately when user sends a message
- Resumes 30s after user interaction completes
- Uses a lighter model for background work to save RAM for user interactions
- Generates its own new tasks after completing work
- Broadcasts status to connected UI clients via SSE
"""

import asyncio
import json
import ollama
import re
import time
from datetime import datetime, timedelta
from typing import Callable, Optional, Set

from autonomous.task_queue import TaskQueue, _in_hours, _in_minutes

# ── Config ────────────────────────────────────────────────────────────────────

HEARTBEAT_INTERVAL   = 5 * 60   # Check queue every 5 minutes
USER_PAUSE_COOLDOWN  = 30       # Wait 30s after user interaction before resuming
MAX_TASK_DURATION    = 10 * 60  # Kill a task after 10 minutes
BACKGROUND_MODEL     = "llama3.2:3b"  # Lighter model for background, save RAM for user
BACKGROUND_MODEL_FALLBACK = "llama3.1:8b"

# Token budget for background tasks — don't be greedy
BACKGROUND_TOKEN_BUDGET = 1500

TASK_EXECUTION_PROMPT = """You are an autonomous background agent working on a task.
You are running silently — the user is NOT watching this interaction.

Your job:
1. Read the task carefully
2. Use your skills to complete it thoroughly
3. Generate follow-up tasks if your work reveals more to do
4. Be self-improving: if you find gaps in your capabilities, write new skills

TASK:
Title: {title}
Type: {task_type}
Description: {description}
Context: {context}

WHAT YOU KNOW ABOUT THE USER:
{user_context}

RECENT COMPLETED TASKS (for continuity):
{recent_tasks}

AVAILABLE SKILLS:
{skills}

SKILL FORMAT: SKILL: {{"name": "...", "args": {{...}}}}
FINAL FORMAT: FINAL: <summary of what you did and what you found>

After FINAL, if you want to add follow-up tasks, output:
NEW_TASKS: [
  {{"title": "...", "description": "...", "task_type": "...", "priority_name": "normal|low|idle"}},
  ...
]

Work autonomously. Use skills. Search the web. Write code. Do real work."""

REFLECT_PROMPT = """Review the agent's recent activity and suggest what it should focus on next.

Recent completed tasks:
{completed_tasks}

User profile:
{user_context}

Current pending task count: {pending_count}

Generate 3-5 new tasks that would make the assistant more useful to this user.
Consider: gaps in skills, things the user will likely ask about, proactive research,
self-improvement opportunities, and maintenance tasks.

Return JSON array:
[{{"title": "...", "description": "...", "task_type": "research|self_improve|prepare|reflect|maintain|custom", "priority_name": "normal|low|idle"}}]
Return ONLY valid JSON."""


class HeartbeatLoop:
    def __init__(
        self,
        task_queue: TaskQueue,
        registry,
        memory,
        user_model,
        broadcast_fn: Callable,   # async fn(event_dict) — sends to UI
    ):
        self.queue = task_queue
        self.registry = registry
        self.memory = memory
        self.user_model = user_model
        self.broadcast = broadcast_fn

        self._paused = False
        self._pause_until: Optional[float] = None
        self._running = False
        self._current_task_id: Optional[int] = None
        self._task_start_time: Optional[float] = None

        # Connected UI clients for status pushes
        self._status_listeners: Set[asyncio.Queue] = set()

    # ── Pause / resume (called by server on user messages) ───────────────

    def pause_for_user(self):
        """Call this when user sends a message. Pauses background work."""
        self._paused = True
        self._pause_until = None  # Indefinite until resume_after_user()
        if self._current_task_id:
            self.queue.pause_running()
            self._current_task_id = None
        asyncio.create_task(self._notify("paused", "User active — background work paused"))

    def resume_after_user(self):
        """Call this N seconds after user interaction completes."""
        self._pause_until = time.time() + USER_PAUSE_COOLDOWN
        self._paused = False
        asyncio.create_task(self._notify("resuming", f"Resuming background work in {USER_PAUSE_COOLDOWN}s..."))

    def is_paused(self) -> bool:
        if self._pause_until and time.time() < self._pause_until:
            return True
        return self._paused

    # ── Main loop ─────────────────────────────────────────────────────────

    async def run(self):
        """The main heartbeat loop. Run this as an asyncio task."""
        self._running = True
        print("[HEARTBEAT] Background loop started")

        # Initial startup delay — let the server stabilise
        await asyncio.sleep(15)

        while self._running:
            try:
                await self._tick()
            except Exception as e:
                print(f"[HEARTBEAT] Loop error: {e}")

            # Sleep until next heartbeat
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    async def _tick(self):
        """One heartbeat cycle."""
        if self.is_paused():
            await self._notify("idle", "Heartbeat: paused for user")
            return

        # Check queue
        task = await asyncio.to_thread(self.queue.next_pending)

        if not task:
            # Nothing to do — trigger a reflection to generate new tasks
            pending = await asyncio.to_thread(self.queue.pending_count)
            if pending == 0:
                await self._notify("reflecting", "Queue empty — reflecting on what to do next...")
                await self._run_reflection()
            else:
                await self._notify("idle", f"{pending} tasks scheduled for later")
            return

        # Execute the task
        await self._execute_task(task)

    # ── Task execution ────────────────────────────────────────────────────

    async def _execute_task(self, task: dict):
        task_id = task["id"]
        title = task["title"]

        await self._notify("working", f"📋 Working on: {title}", task=task)

        await asyncio.to_thread(self.queue.start, task_id)
        self._current_task_id = task_id
        self._task_start_time = time.time()

        try:
            result = await asyncio.wait_for(
                self._run_task_model(task),
                timeout=MAX_TASK_DURATION
            )

            # Extract new tasks if model generated them
            new_tasks = result.pop("new_tasks", [])
            summary = result.get("output", "Task complete.")

            await asyncio.to_thread(self.queue.complete, task_id, summary)

            await self._notify(
                "task_done",
                f"✓ Completed: {title}",
                task=task,
                summary=summary[:200],
            )

            # Add follow-up tasks
            for nt in new_tasks:
                try:
                    new_id = await asyncio.to_thread(
                        self.queue.add,
                        title=nt.get("title", "Follow-up task"),
                        description=nt.get("description", ""),
                        task_type=nt.get("task_type", "custom"),
                        priority_name=nt.get("priority_name", "normal"),
                        parent_id=task_id,
                    )
                    await self._notify("task_added", f"+ Added follow-up: {nt.get('title', '')}")
                except Exception as e:
                    print(f"[HEARTBEAT] Error adding follow-up task: {e}")

            # Log to memory
            await asyncio.to_thread(
                self.memory.log_interaction,
                f"[BACKGROUND] {title}",
                {"category": task["task_type"], "confidence": 1.0},
                BACKGROUND_MODEL,
                summary,
                True, 0,
                int((time.time() - self._task_start_time) * 1000),
            )

            # Update user model from any new info
            await asyncio.to_thread(
                self.user_model.extract_from_exchange,
                f"[background task: {title}]",
                summary
            )

        except asyncio.TimeoutError:
            await asyncio.to_thread(self.queue.fail, task_id, "Timed out")
            await self._notify("task_failed", f"✗ Timed out: {title}", task=task)

        except Exception as e:
            err = str(e)
            await asyncio.to_thread(self.queue.fail, task_id, err)
            await self._notify("task_failed", f"✗ Failed: {title} — {err[:80]}", task=task)

        finally:
            self._current_task_id = None
            self._task_start_time = None

    async def _run_task_model(self, task: dict) -> dict:
        """Run a task through the model with tool use."""
        user_context = await asyncio.to_thread(self.user_model.get_context_for_prompt)
        recent = await asyncio.to_thread(
            lambda: [t["title"] + ": " + (t.get("result_summary") or "")[:80]
                     for t in self.queue.get_recent_completed(5)]
        )

        prompt = TASK_EXECUTION_PROMPT.format(
            title=task["title"],
            task_type=task["task_type"],
            description=task["description"],
            context=json.dumps(task.get("context", {})),
            user_context=user_context,
            recent_tasks="\n".join(recent) or "None yet.",
            skills=self.registry.list_skills(),
        )

        messages = [{"role": "user", "content": prompt}]
        tool_count = 0
        max_tools = 12
        last_reply = ""
        thinking_log = []

        while tool_count < max_tools:
            # Check for pause between tool calls
            if self.is_paused():
                return {
                    "output": f"Task paused (user active). Partial work: {last_reply[:200]}",
                    "new_tasks": [],
                }

            try:
                response = await asyncio.to_thread(
                    ollama.chat,
                    model=BACKGROUND_MODEL,
                    messages=messages,
                    options={
                        "temperature": 0.6,
                        "num_predict": BACKGROUND_TOKEN_BUDGET,
                        "num_ctx": 6144,
                        "num_gpu": 999,
                    }
                )
            except ollama.ResponseError as e:
                if "out of memory" in str(e).lower():
                    # Try fallback model
                    response = await asyncio.to_thread(
                        ollama.chat,
                        model=BACKGROUND_MODEL_FALLBACK,
                        messages=messages,
                        options={"temperature": 0.6, "num_predict": 800, "num_ctx": 4096}
                    )
                else:
                    raise

            raw = response["message"]["content"]
            last_reply = raw

            # DeepSeek think blocks
            think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
            reply = raw
            if think_match:
                thinking_log.append(think_match.group(1).strip())
                reply = raw.replace(think_match.group(0), "").strip()

            messages.append({"role": "assistant", "content": raw})

            # Parse NEW_TASKS
            new_tasks = []
            new_tasks_match = re.search(r"NEW_TASKS:\s*(\[.*?\])", reply, re.DOTALL)
            if new_tasks_match:
                try:
                    new_tasks = json.loads(new_tasks_match.group(1))
                    reply = reply[:new_tasks_match.start()].strip()
                except Exception:
                    pass

            # FINAL answer
            final_match = re.search(r"FINAL:\s*(.*?)(?=NEW_TASKS:|$)", reply, re.DOTALL)
            if final_match:
                return {
                    "output": final_match.group(1).strip(),
                    "new_tasks": new_tasks,
                    "thinking": thinking_log,
                }

            # SKILL call
            skill_match = re.search(r"SKILL:\s*(\{.*?\})", reply, re.DOTALL)
            if skill_match:
                try:
                    sc = json.loads(skill_match.group(1))
                    name = sc.get("name", "")
                    args = sc.get("args", {})

                    await self._notify("skill_call", f"⚙ {name}({json.dumps(args)[:80]})")

                    result = await asyncio.to_thread(self.registry.run, name, **args)
                    result_str = str(result)
                    if len(result_str) > 4000:
                        result_str = result_str[:3800] + "...[truncated]"

                    messages.append({
                        "role": "user",
                        "content": f"Skill '{name}' result:\n{result_str}\n\nContinue. FINAL: or use more skills."
                    })
                    tool_count += 1
                    continue

                except Exception as e:
                    messages.append({
                        "role": "user",
                        "content": f"Skill error: {e}. Try another approach."
                    })
                    tool_count += 1
                    continue

            # Stuck
            messages.append({
                "role": "user",
                "content": "Continue. Use a SKILL, or output FINAL: with your result."
            })

        return {
            "output": _strip_meta(last_reply)[:500] or "Task attempted — no clear result.",
            "new_tasks": [],
        }

    # ── Reflection ────────────────────────────────────────────────────────

    async def _run_reflection(self):
        """When queue is empty, reflect and generate new tasks."""
        try:
            completed = await asyncio.to_thread(self.queue.get_recent_completed, 10)
            user_context = await asyncio.to_thread(self.user_model.get_context_for_prompt)
            pending_count = await asyncio.to_thread(self.queue.pending_count)

            completed_summary = "\n".join([
                f"- {t['title']}: {(t.get('result_summary') or '')[:80]}"
                for t in completed
            ]) or "None yet."

            resp = await asyncio.to_thread(
                ollama.generate,
                model=BACKGROUND_MODEL,
                prompt=REFLECT_PROMPT.format(
                    completed_tasks=completed_summary,
                    user_context=user_context,
                    pending_count=pending_count,
                ),
                options={"temperature": 0.7, "num_predict": 800, "num_ctx": 3000}
            )

            text = resp["response"].strip()
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                return

            new_tasks = json.loads(match.group())
            added = 0
            for nt in new_tasks:
                if isinstance(nt, dict) and nt.get("title"):
                    await asyncio.to_thread(
                        self.queue.add,
                        title=nt["title"],
                        description=nt.get("description", ""),
                        task_type=nt.get("task_type", "custom"),
                        priority_name=nt.get("priority_name", "idle"),
                    )
                    added += 1

            await self._notify("tasks_generated", f"🧠 Reflection complete — added {added} new tasks")

        except Exception as e:
            print(f"[HEARTBEAT] Reflection error: {e}")

    # ── Notifications to UI ───────────────────────────────────────────────

    async def _notify(self, event_type: str, message: str, **kwargs):
        """Broadcast a heartbeat event to all connected UI clients."""
        print(f"[HEARTBEAT] {message}")
        try:
            await self.broadcast({
                "type": f"heartbeat_{event_type}",
                "message": message,
                "timestamp": datetime.now().isoformat(),
                **{k: v for k, v in kwargs.items() if k != "task" or v is None},
                **({"task_title": kwargs["task"]["title"],
                    "task_type": kwargs["task"]["task_type"]}
                   if kwargs.get("task") else {}),
            })
        except Exception as e:
            print(f"[HEARTBEAT] Notify error: {e}")

    def stop(self):
        self._running = False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _strip_meta(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^SKILL:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"NEW_TASKS:.*", "", text, flags=re.DOTALL)
    return text.strip()

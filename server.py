"""
server.py — Final integrated version

New additions:
- POST /personality  — save personality config from setup screen
- GET  /setup        — serve personality picker (redirected to on first run)
- Personality injected into all system prompts
- Assistant name used throughout
"""

import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Set

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent))

from core.classifier import classify_intent
from core.rewriter import rewrite_prompt
from core.router import route_to_model, get_fallback
from core.token_budget import get_token_budget
from memory.store import AgentMemory
from memory.user_model import UserModel
from memory.personality import PersonalityConfig
from proactive.engine import ProactiveEngine
from skills.registry import SkillRegistry
from autonomous.task_queue import TaskQueue
from autonomous.heartbeat import HeartbeatLoop

AGENT_HOME = os.environ.get("AGENT_HOME", str(Path(__file__).parent))
DB_PATH    = os.environ.get("AGENT_DB",   f"{AGENT_HOME}/memory/agent.db")

# ── Global broadcast ──────────────────────────────────────────────────────────
_broadcast_queues: Set[asyncio.Queue] = set()

async def broadcast(event: dict):
    dead = set()
    for q in _broadcast_queues:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            dead.add(q)
    for q in dead:
        _broadcast_queues.discard(q)

# ── Globals ───────────────────────────────────────────────────────────────────
registry:    SkillRegistry     = None
memory:      AgentMemory       = None
user_model:  UserModel         = None
personality: PersonalityConfig = None
proactive:   ProactiveEngine   = None
task_queue:  TaskQueue         = None
heartbeat:   HeartbeatLoop     = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global registry, memory, user_model, personality, proactive, task_queue, heartbeat

    os.makedirs(f"{AGENT_HOME}/memory",      exist_ok=True)
    os.makedirs(f"{AGENT_HOME}/workspace",   exist_ok=True)
    os.makedirs(f"{AGENT_HOME}/screenshots", exist_ok=True)

    registry    = SkillRegistry()
    memory      = AgentMemory(DB_PATH)
    user_model  = UserModel(memory)
    personality = PersonalityConfig(f"{AGENT_HOME}/memory/personality.json")
    proactive   = ProactiveEngine(user_model, memory, registry)
    task_queue  = TaskQueue(DB_PATH)

    heartbeat = HeartbeatLoop(
        task_queue=task_queue,
        registry=registry,
        memory=memory,
        user_model=user_model,
        broadcast_fn=broadcast,
    )
    # asyncio.create_task(heartbeat.run(), name="heartbeat")  # disabled for testing

    name = personality.name or "Assistant"
    print(f"[SERVER] Ready → http://localhost:8765")
    print(f"[SERVER] Assistant: {name} | Configured: {personality.is_configured}")
    print(f"[SERVER] Skills: {len(registry.skills)} | Tasks: {task_queue.summary()}")

    yield
    heartbeat.stop()


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UI_DIR = Path(__file__).parent / "ui"
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    if not personality.is_configured:
        return RedirectResponse("/setup")
    return (UI_DIR / "index.html").read_text()

@app.get("/setup", response_class=HTMLResponse)
async def setup_page():
    return (UI_DIR / "personality.html").read_text()


# ── Personality API ───────────────────────────────────────────────────────────

@app.post("/personality")
async def save_personality(request: Request):
    config = await request.json()
    await asyncio.to_thread(personality.save, config)

    name = config.get("name", "Assistant")
    await asyncio.to_thread(user_model.set_preference, "assistant_name", name)

    # Queue a personalised greeting
    await asyncio.to_thread(
        task_queue.add,
        title=f"Compose greeting as {name}",
        description=(
            f"Your name is {name}. Personality: {config.get('profile','balanced')}. "
            f"Write a warm in-character first greeting (2-3 sentences). "
            f"Save it to workspace as 'greeting.txt'."
        ),
        task_type="prepare",
        priority_name="high",
    )
    return {"status": "saved", "name": name}

@app.get("/personality")
async def get_personality():
    return personality.get()


# ── Global SSE stream ─────────────────────────────────────────────────────────

@app.get("/events")
async def event_stream(request: Request):
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    _broadcast_queues.add(q)

    async def generate():
        try:
            summary = await asyncio.to_thread(task_queue.summary)
            p = personality.get()
            yield _sse({
                "type": "connected",
                "queue_summary": summary,
                "assistant_name": p.get("name") or "Assistant",
                "configured": p.get("configured", False),
            })
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield _sse(event)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            _broadcast_queues.discard(q)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    msg = body.get("message", "").strip()
    if not msg:
        return {"error": "empty"}
    heartbeat.pause_for_user()
    return StreamingResponse(
        _chat_stream(msg),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _chat_stream(user_message: str) -> AsyncGenerator[str, None]:
    def sse(t, **kw):
        return _sse({"type": t, **kw})

    t0 = time.time()
    name = personality.name or "Assistant"

    yield sse("stage", message="Reading your message...")
    await asyncio.sleep(0)
    # Hardcoded routing — bypass 0.5b classifier until more models loaded
    intent   = {"category": "general_chat", "confidence": 1.0, "latency": "fast"}
    category = "general_chat"

    yield sse("stage", message="Remembering what I know about you...")
    await asyncio.sleep(0)

    user_ctx = await asyncio.to_thread(user_model.get_context_for_prompt)
    await asyncio.to_thread(user_model.extract_from_message, user_message)

    model   = "llama3.2:3b"
    latency = "fast"
    budget  = 512

    yield sse("stage", message=f"{name} is thinking...")
    await asyncio.sleep(0)

    rewritten = user_message  # skip rewrite to avoid 0.5b call
    past = await asyncio.to_thread(memory.semantic_search, user_message, 3)
    past_ctx = "\n".join([
        f"- '{p['input'][:50]}' → '{p['output'][:80]}'"
        for p in past
    ]) or "None yet."

    system = personality.get_full_system_prompt(model, category, user_ctx, past_ctx)

    result = None
    models_to_try = [model] + get_fallback(model)

    for attempt in range(6):
        current = models_to_try[min(attempt, len(models_to_try) - 1)]
        if attempt > 0:
            yield sse("stage", message=f"Trying a different approach...")

        try:
            result, skill_events, think_events = await _run_model(
                rewritten, current, system, budget
            )
            for ev in think_events:
                yield sse("thinking", text=ev)
            for ev in skill_events:
                yield sse("skill_call", skill=ev.get("name", "?"), args=ev.get("args", {}))
            if result.get("success"):
                break
        except Exception as e:
            print(f"[CHAT ERROR] attempt {attempt}: {type(e).__name__}: {e}", flush=True)
            rewritten = f"Error: {e}\nOriginal: {user_message}\nTry differently."

    print(f"[DEBUG] result={result}", flush=True)
    yield sse("stage_done", message="Done")

    final = (result or {}).get("output", "Something went wrong — please try again.")
    final = final  # skip personalise_response (uses 0.5b, causes hang)

    yield sse("final", message=final)

    # Log
    dur = int((time.time() - t0) * 1000)
    await asyncio.to_thread(
        memory.log_interaction, user_message, intent,
        (result or {}).get("model", model), final,
        (result or {}).get("success", False),
        (result or {}).get("tool_calls", 0), dur,
    )
    await asyncio.to_thread(user_model.extract_from_exchange, user_message, final)

    # Proactive
    pro = await asyncio.to_thread(proactive.check_after_message, user_message, final)
    if pro:
        yield sse("proactive", message=pro)

    # Follow-up research task
    if category in ("research", "web_search", "planning", "agentic_task", "coding"):
        await asyncio.to_thread(
            task_queue.add,
            title=f"Follow up: {user_message[:55]}",
            description=(
                f"User asked: {user_message}\n"
                f"Response: {final[:300]}\n\n"
                f"Dig deeper. Find additional useful info. Prepare a proactive update."
            ),
            task_type="research",
            priority_name="low",
        )

    heartbeat.resume_after_user()


async def _run_model(prompt, model, system, budget):
    import ollama, re

    messages = [{"role": "user",
                 "content": f"Task: {prompt}\n\nUse SKILL: {{...}} or FINAL: <answer>"}]
    skill_events, think_events = [], []
    tool_count = 0
    last_reply = ""

    while tool_count < 20:
        msgs_with_system = [{"role": "system", "content": system}] + messages
        resp = await asyncio.to_thread(
            ollama.chat, model=model, messages=msgs_with_system,
            options={"temperature": 0.7, "num_predict": budget,
                     "num_ctx": 4096}
        )
        raw = resp["message"]["content"]
        last_reply = raw
        print(f"[DEBUG RAW] {repr(raw[:200])}", flush=True)

        think_m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        reply = raw
        if think_m:
            think_events.append(think_m.group(1).strip()[:400])
            reply = raw.replace(think_m.group(0), "").strip()

        messages.append({"role": "assistant", "content": raw})

        final_m = re.search(r"FINAL:[\s\n]*(.*)", reply, re.DOTALL)
        if final_m:
            return (
                {"output": final_m.group(1).strip(), "success": True,
                 "tool_calls": tool_count, "model": model},
                skill_events, think_events,
            )

        skill_m = re.search(r"SKILL:\s*(\{.*?\})", reply, re.DOTALL)
        if skill_m:
            try:
                sc = json.loads(skill_m.group(1))
                skill_events.append(sc)
                res = await asyncio.to_thread(registry.run, sc["name"], **sc.get("args", {}))
                res_str = str(res)
                if len(res_str) > 6000:
                    res_str = res_str[:5700] + "...[truncated]"
                messages.append({"role": "user",
                                 "content": f"Skill result:\n{res_str}\n\nContinue."})
                tool_count += 1
                continue
            except Exception as e:
                messages.append({"role": "user",
                                 "content": f"Skill error: {e}. Try another."})
                tool_count += 1
                continue

        # Model gave plain response — accept it directly
        if last_reply.strip():
            return (
                {"output": last_reply.strip(), "success": True,
                 "tool_calls": tool_count, "model": model},
                skill_events, think_events,
            )
        messages.append({"role": "user",
                         "content": "Continue. SKILL: or FINAL: required."})

    # Accept plain response if model didn't use FINAL: format
    if last_reply.strip():
        return (
            {"output": last_reply.strip(), "success": True,
             "tool_calls": tool_count, "model": model},
            skill_events, think_events,
        )
    return (
        {"output": last_reply.strip(), "success": False,
         "tool_calls": tool_count, "model": model},
        skill_events, think_events,
    )


# ── Task queue API ────────────────────────────────────────────────────────────

@app.get("/tasks")
async def get_tasks(status: str = None):
    tasks   = await asyncio.to_thread(task_queue.get_all, status)
    summary = await asyncio.to_thread(task_queue.summary)
    return {"tasks": tasks, "summary": summary}

@app.post("/tasks")
async def create_task(request: Request):
    body = await request.json()
    tid = await asyncio.to_thread(
        task_queue.add,
        title=body.get("title", "User task"),
        description=body.get("description", ""),
        task_type=body.get("task_type", "custom"),
        priority_name=body.get("priority_name", "normal"),
    )
    return {"id": tid}

@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: int):
    await asyncio.to_thread(task_queue.cancel, task_id, "Cancelled by user")
    return {"status": "cancelled"}

@app.get("/tasks/summary")
async def task_summary():
    return await asyncio.to_thread(task_queue.summary)

# ── Profile & Proactive ───────────────────────────────────────────────────────

@app.get("/profile")
async def get_profile():
    p = await asyncio.to_thread(user_model.get_display_profile)
    p["assistant_name"] = personality.name
    return p

@app.get("/proactive")
async def get_proactive():
    return {"suggestions": await asyncio.to_thread(proactive.get_sidebar_suggestions)}

@app.get("/proactive/push")
async def proactive_push():
    return {"message": await asyncio.to_thread(proactive.get_push_message)}

# ── Helper ────────────────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8765,
                reload=False, log_level="warning")

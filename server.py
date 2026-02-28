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
from core.pipeline_pre import run_pre_pipeline
from core.fast_classifier import fast_classify
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
    asyncio.create_task(heartbeat.run(), name="heartbeat")

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


# ── Conversation history (last 6 exchanges) ─────────────────────────────────
_conversation_history: list = []
MAX_HISTORY = 6

def _add_to_history(role: str, content: str):
    _conversation_history.append({"role": role, "content": content})
    if len(_conversation_history) > MAX_HISTORY * 2:
        del _conversation_history[:-MAX_HISTORY * 2]

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
    # Track conversation
    _add_to_history("user", user_message)

    # Fast heuristic first — only call 0.5b if ambiguous
    intent = fast_classify(user_message)
    if intent is None:
        intent = await asyncio.to_thread(run_pre_pipeline, user_message)
    category = intent.get("category", "general_chat")

    yield sse("stage", message="Remembering what I know about you...")
    await asyncio.sleep(0)

    user_ctx = await asyncio.to_thread(user_model.get_context_for_prompt)
    await asyncio.to_thread(user_model.extract_from_message, user_message)

    route   = route_to_model(intent)
    model   = route["model"]
    latency = route.get("latency", "fast")
    budget  = get_token_budget(latency, category)

    # Fire quick ack in background while we prep
    ack_task = asyncio.create_task(_quick_ack(user_message, name))

    rewritten = user_message
    # Only search memory if message references past context
    memory_triggers = {"remember", "earlier", "last time", "you said", "we discussed",
                       "before", "previously", "again", "still", "anymore"}
    msg_words = set(user_message.lower().split())
    if msg_words & memory_triggers:
        past = await asyncio.to_thread(memory.semantic_search, user_message, 3)
        past_ctx = "\n".join([
            f"- '{p['input'][:50]}' → '{p['output'][:80]}'"
            for p in past
        ]) or "None yet."
    else:
        past_ctx = "None yet." 

    system = personality.get_full_system_prompt(model, category, user_ctx, past_ctx)

    # Send quick ack once ready
    ack = await ack_task
    if ack:
        yield sse("quick_ack", message=ack)

    yield sse("stage", message=f"{name} is thinking...")
    await asyncio.sleep(0)

    final = "Something went wrong — please try again."
    result = None

    try:
        async for event_type, data in _run_model_streaming(rewritten, model, system, budget, _conversation_history):
            if event_type == "token":
                yield sse("token", text=data)
            elif event_type == "think":
                yield sse("thinking", text=data)
            elif event_type == "skill":
                yield sse("skill_call", skill=data.get("name", "?"), args=data.get("args", {}))
            elif event_type == "done":
                result = data
                final = data.get("output", final)
                break
    except Exception as e:
        print(f"[CHAT ERROR] {type(e).__name__}: {e}", flush=True)
        try:
            result, skill_events, think_events = await _run_model(rewritten, model, system, budget)
            final = result.get("output", final)
        except Exception as e2:
            print(f"[CHAT ERROR fallback] {e2}", flush=True)

    # Track assistant response in history
    if final and final != "Something went wrong — please try again.":
        _add_to_history("assistant", final)

    yield sse("stage_done", message="Done")
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



async def _quick_ack(user_message: str, name: str) -> str:
    """Generate a fast 1-2 sentence acknowledgement using 0.5b model."""
    import ollama
    try:
        resp = await asyncio.to_thread(
            ollama.generate,
            model="qwen2.5:0.5b",
            prompt=(
                f"You are {name}, a helpful assistant. "
                f"The user said: \"{user_message[:150]}\"\n"
                f"Write ONE short sentence (max 15 words) acknowledging you heard them "
                f"and are working on it. Be natural. No FINAL: prefix."
            ),
            options={"temperature": 0.7, "num_predict": 30, "num_ctx": 512}
        )
        ack = resp["response"].strip()
        if len(ack) > 120 or len(ack) < 5:
            return ""
        return ack
    except Exception:
        return ""


async def _iter_stream(stream):
    """Wrap synchronous ollama stream iterator for async use."""
    import concurrent.futures
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        it = iter(stream)
        while True:
            try:
                chunk = await loop.run_in_executor(pool, next, it)
                yield chunk
            except StopIteration:
                break


async def _run_model_streaming(prompt, model, system, budget, history=None):
    """Streams tokens. Yields: ("token", text) | ("skill", ev) | ("think", text) | ("done", dict)"""
    import ollama, re

    # Build messages with conversation history
    messages = list(history or [])
    # Replace last user message with task-formatted version
    if messages and messages[-1]["role"] == "user":
        messages[-1] = {"role": "user", "content": f"Task: {prompt}\n\nUse SKILL: {{...}} or FINAL: <answer>"}
    else:
        messages.append({"role": "user", "content": f"Task: {prompt}\n\nUse SKILL: {{...}} or FINAL: <answer>"})
    skill_events, think_events = [], []
    tool_count = 0
    last_reply = ""
    first_call = True

    while tool_count < 20:
        msgs_with_system = [{"role": "system", "content": system}] + messages

        if first_call:
            first_call = False
            stream = await asyncio.to_thread(
                ollama.chat, model=model, messages=msgs_with_system,
                stream=True,
                options={"temperature": 0.7, "num_predict": budget, "num_ctx": 4096}
            )
            collected = []
            async for chunk in _iter_stream(stream):
                token = chunk.get("message", {}).get("content", "")
                if token:
                    collected.append(token)
                    yield ("token", token)
            raw = "".join(collected)
        else:
            resp = await asyncio.to_thread(
                ollama.chat, model=model, messages=msgs_with_system,
                options={"temperature": 0.7, "num_predict": budget, "num_ctx": 4096}
            )
            raw = resp["message"]["content"]

        last_reply = raw
        think_m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        reply = raw
        if think_m:
            think_events.append(think_m.group(1).strip()[:400])
            reply = raw.replace(think_m.group(0), "").strip()
            yield ("think", think_m.group(1).strip()[:400])

        messages.append({"role": "assistant", "content": raw})

        final_m = re.search(r"FINAL:[\s\n]*(.*)", reply, re.DOTALL)
        if final_m:
            yield ("done", {"output": final_m.group(1).strip(), "success": True,
                            "tool_calls": tool_count, "model": model})
            return

        skill_m = re.search(r"SKILL:\s*(\{.*?\})", reply, re.DOTALL)
        if skill_m:
            try:
                sc = json.loads(skill_m.group(1))
                yield ("skill", sc)
                res = await asyncio.to_thread(registry.run, sc["name"], **sc.get("args", {}))
                res_str = str(res)[:6000]
                messages.append({"role": "user", "content": f"Skill result:\n{res_str}\n\nContinue."})
                tool_count += 1
                continue
            except Exception as e:
                messages.append({"role": "user", "content": f"Skill error: {e}. Try another."})
                tool_count += 1
                continue

        if last_reply.strip():
            yield ("done", {"output": last_reply.strip(), "success": True,
                            "tool_calls": tool_count, "model": model})
            return

        messages.append({"role": "user", "content": "Continue. SKILL: or FINAL: required."})

    yield ("done", {"output": last_reply.strip() or "Something went wrong — please try again.",
                    "success": bool(last_reply.strip()), "tool_calls": tool_count, "model": model})

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

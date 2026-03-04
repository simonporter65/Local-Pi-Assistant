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

from core.fast_classifier import fast_classify
from core.router import route_to_model
from core.token_budget import get_token_budget
from memory.store import AgentMemory
from memory.user_model import UserModel
from memory.personality import PersonalityConfig
from proactive.engine import ProactiveEngine
from skills.registry import SkillRegistry
from autonomous.task_queue import TaskQueue
from autonomous.heartbeat import HeartbeatLoop
from memory.training_collector import TrainingCollector

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
training:    TrainingCollector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global registry, memory, user_model, personality, proactive, task_queue, heartbeat, training

    os.makedirs(f"{AGENT_HOME}/memory",      exist_ok=True)
    os.makedirs(f"{AGENT_HOME}/workspace",   exist_ok=True)
    os.makedirs(f"{AGENT_HOME}/screenshots", exist_ok=True)

    registry    = SkillRegistry()
    memory      = AgentMemory(DB_PATH)
    user_model  = UserModel(memory)
    personality = PersonalityConfig(f"{AGENT_HOME}/memory/personality.json")
    proactive   = ProactiveEngine(user_model, memory, registry)
    task_queue  = TaskQueue(DB_PATH)

    training = TrainingCollector(DB_PATH)

    heartbeat = HeartbeatLoop(
        task_queue=task_queue,
        registry=registry,
        memory=memory,
        user_model=user_model,
        broadcast_fn=broadcast,
        personality=personality,
    )
    asyncio.create_task(heartbeat.run(), name="heartbeat")

    # Keep key models warm — ping every 3 minutes to prevent eviction
    async def _keepalive():
        import ollama
        while True:
            await asyncio.sleep(180)
            try:
                await asyncio.to_thread(
                    ollama.generate, model="qwen3.5:0.8b",
                    prompt="hi", options={"num_predict": 1, "num_ctx": 64},
                    think=False,
                )
                print("[KEEPALIVE] qwen3.5:0.8b warmed", flush=True)
            except Exception:
                pass
            try:
                await asyncio.to_thread(ollama.embeddings, model="nomic-embed-text", prompt="hi")
                print("[KEEPALIVE] nomic-embed-text warmed", flush=True)
            except Exception:
                pass
    asyncio.create_task(_keepalive(), name="keepalive")

    # Pre-warm the model route cache so first request doesn't subprocess
    from core.router import get_installed_models
    await asyncio.to_thread(get_installed_models)

    # Warm key models before announcing readiness — first message will be instant
    import ollama as _ollama
    try:
        await asyncio.to_thread(
            _ollama.generate, model="qwen3.5:0.8b",
            prompt="hi", options={"num_predict": 1, "num_ctx": 64},
            think=False,
        )
        print("[SERVER] Warmed qwen3.5:0.8b", flush=True)
    except Exception as _e:
        print(f"[SERVER] Could not warm qwen3.5:0.8b: {_e}", flush=True)
    # nomic-embed-text uses the embeddings API, not generate
    try:
        await asyncio.to_thread(_ollama.embeddings, model="nomic-embed-text", prompt="warmup")
        print("[SERVER] Warmed nomic-embed-text", flush=True)
    except Exception as _e:
        print(f"[SERVER] Could not warm nomic-embed-text: {_e}", flush=True)

    name = personality.name or "Assistant"

    # Startup greeting — model is already loaded so this is fast
    async def _startup_greeting():
        try:
            ctx = await asyncio.to_thread(user_model.get_context_for_prompt)
            if ctx and "nothing yet" not in ctx.lower() and len(ctx) > 20:
                resp = await asyncio.to_thread(
                    _ollama.generate,
                    model="qwen3.5:0.8b",
                    prompt=(
                        f"You are {name}. You know this about your user:\n{ctx}\n\n"
                        f"Write a warm 1-sentence greeting acknowledging what you remember. "
                        f"Be natural, not robotic. Max 20 words."
                    ),
                    options={"temperature": 0.7, "num_predict": 40, "num_ctx": 512}
                )
                greeting = resp["response"].strip()
                if greeting:
                    await broadcast({"type": "greeting", "message": greeting})
        except Exception as e:
            print(f"[GREETING] {e}")

    asyncio.create_task(_startup_greeting())

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


# ── Conversation history (keyed by session) ─────────────────────────────────
_session_histories: dict = {}
_session_last_used: dict = {}
MAX_HISTORY = 6
MAX_SESSIONS = 50       # Cap total sessions in memory
SESSION_TTL  = 3600     # Drop sessions idle > 1 hour

def _get_history(session_id: str) -> list:
    _session_last_used[session_id] = time.time()
    return _session_histories.setdefault(session_id, [])

def _add_to_history(session_id: str, role: str, msg: str):
    h = _get_history(session_id)
    h.append({"role": role, "content": msg})
    if len(h) > MAX_HISTORY * 2:
        del h[:-MAX_HISTORY * 2]
    # Prune stale sessions if we're over the cap
    if len(_session_histories) > MAX_SESSIONS:
        now = time.time()
        stale = [sid for sid, ts in _session_last_used.items()
                 if now - ts > SESSION_TTL]
        for sid in stale:
            _session_histories.pop(sid, None)
            _session_last_used.pop(sid, None)

# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    msg = body.get("message", "").strip()
    if not msg:
        return {"error": "empty"}
    session_id = request.headers.get("X-Session-ID", "default")
    heartbeat.pause_for_user()
    return StreamingResponse(
        _chat_stream(msg, session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _chat_stream(user_message: str, session_id: str = 'default') -> AsyncGenerator[str, None]:
    def sse(t, **kw):
        print(f"[SSE] {t}: {kw}", flush=True)
        return _sse({"type": t, **kw})

    t0 = time.time()
    name = personality.name or "Assistant"

    # Classify immediately (pure heuristic, ~0ms)
    _add_to_history(session_id, "user", user_message)
    intent = fast_classify(user_message)
    category = intent.get("category", "general_chat")
    print(f"[CLASSIFY] {intent['_source']}: {category} for: {user_message[:50]}", flush=True)

    # Parallel pre-flight: fetch user context + score previous exchange simultaneously
    user_ctx, _ = await asyncio.gather(
        asyncio.to_thread(user_model.get_context_for_prompt),
        asyncio.to_thread(training.score_previous_exchange, user_message, session_id),
    )

    # Fast heuristic extraction (~0ms) — may update facts immediately
    await asyncio.to_thread(user_model.extract_from_message, user_message)
    # Notify UI immediately if any facts were extracted
    asyncio.create_task(broadcast({"type": "profile_updated"}))

    route   = route_to_model(intent)
    model   = route["model"]
    latency = route.get("latency", "fast")
    budget  = get_token_budget(latency, category)

    yield sse("model", name=model, latency=latency, category=category)
    await asyncio.sleep(0)

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

    skills_list = registry.list_skills() if category in {
        "research", "coding", "debugging", "agentic_task", "web_browsing",
        "data_analysis", "file_management", "shell_command",
        "screenshot_analysis", "image_description"
    } else ""
    system = personality.get_full_system_prompt(model, category, user_ctx, past_ctx, skills_list)

    # For web_search: pre-execute the search and inject results directly.
    # More reliable than hoping the model emits the SKILL: format.
    if category == "web_search":
        yield sse("stage", message="Searching the web...")
        await asyncio.sleep(0)
        try:
            search_results = await asyncio.to_thread(
                registry.run, "web_search", query=rewritten, max_results=5
            )
            system = system + f"\n\nWEB SEARCH RESULTS for '{rewritten}':\n{search_results}\n\nSynthesize these results into a helpful, accurate response."
            print(f"[WEB SEARCH] Pre-executed for: {rewritten[:50]}", flush=True)
        except Exception as e:
            print(f"[WEB SEARCH] Pre-execute failed: {e}", flush=True)

    # For web_browsing with a simple URL: pre-fetch the page content.
    # Faster and more reliable than relying on the model to emit SKILL: correctly.
    if category == "web_browsing":
        import re as _re
        url_match = _re.search(r'https?://[^\s"\'<>]+', user_message)
        if url_match:
            found_url = url_match.group(0)
            # Only pre-fetch if it looks like a read request, not interaction
            interaction_words = {"click", "fill", "submit", "login", "sign", "type", "scroll"}
            is_interaction = bool(set(user_message.lower().split()) & interaction_words)
            if not is_interaction:
                yield sse("stage", message="Fetching page...")
                await asyncio.sleep(0)
                try:
                    page_content = await asyncio.to_thread(
                        registry.run, "web_fetch", url=found_url, max_chars=4000
                    )
                    system = system + f"\n\nPAGE CONTENT from {found_url}:\n{str(page_content)[:4000]}\n\nAnswer based on this content."
                    print(f"[WEB_FETCH] Pre-executed for: {found_url[:60]}", flush=True)
                except Exception as e:
                    print(f"[WEB_FETCH] Pre-execute failed: {e}", flush=True)

    # For image/screenshot tasks: pre-run vision skill if a file path is in the message
    if category in {"screenshot_analysis", "image_description"}:
        import re as _re
        img_match = _re.search(r'(/[^\s"\']+\.(?:png|jpg|jpeg|gif|webp))', user_message, _re.I)
        if img_match:
            yield sse("stage", message="Analyzing image...")
            await asyncio.sleep(0)
            try:
                img_path = img_match.group(1)
                question = user_message.replace(img_path, "").strip() or "Describe what you see in detail."
                vision_result = await asyncio.to_thread(
                    registry.run, "vision", image_path=img_path, question=question
                )
                system = system + f"\n\nIMAGE ANALYSIS of {img_path}:\n{vision_result}\n\nAnswer based on this analysis."
                print(f"[VISION] Pre-executed for: {img_path}", flush=True)
            except Exception as e:
                print(f"[VISION] Pre-execute failed: {e}", flush=True)

    yield sse("stage", message=f"{name} is thinking...")
    await asyncio.sleep(0)

    final = "Something went wrong — please try again."
    result = None

    # Use skills for agentic categories (not web_search — handled above)
    use_skills = category in {"research", "coding", "debugging",
                              "agentic_task", "web_browsing", "data_analysis",
                              "file_management", "shell_command",
                              "screenshot_analysis", "image_description"}
    try:
        async for event_type, data in _run_model_streaming(rewritten, model, system, budget, _get_history(session_id), use_skills=use_skills):
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
    except asyncio.TimeoutError:
        print(f"[CHAT TIMEOUT] Model took too long", flush=True)
        final = "Sorry, that took too long. Please try again."
    except Exception as e:
        print(f"[CHAT ERROR] {type(e).__name__}: {e}", flush=True)
        final = "Something went wrong — please try again." 

    # Track assistant response in history
    if final and final != "Something went wrong — please try again.":
        _add_to_history(session_id, "assistant", final)

    yield sse("stage_done", message="Done")
    yield sse("final", message=final)

    # Fire-and-forget — user already has their response, don't block the stream
    asyncio.create_task(_post_response(
        user_message, final, intent, result, model, session_id, t0, category
    ))
    heartbeat.resume_after_user()


async def _post_response(user_message, final, intent, result, model, session_id, t0, category):
    """All post-response work — runs as a background task after final is sent."""
    dur = int((time.time() - t0) * 1000)
    if not final or "went wrong" in final:
        await asyncio.to_thread(
            memory.log_interaction, user_message, intent,
            (result or {}).get("model", model), final,
            False, (result or {}).get("tool_calls", 0), dur,
        )
        return

    sys_prompt = personality.get_full_system_prompt(model, category, "", "")

    # Run log, extract facts, and record training data in parallel
    await asyncio.gather(
        asyncio.to_thread(
            memory.log_interaction, user_message, intent,
            (result or {}).get("model", model), final,
            (result or {}).get("success", True),
            (result or {}).get("tool_calls", 0), dur,
        ),
        asyncio.to_thread(user_model.extract_from_exchange, user_message, final),
        asyncio.to_thread(training.record_exchange, sys_prompt, user_message, final, session_id, model),
    )
    await broadcast({"type": "profile_updated"})

    # Proactive check (rate-limited internally — fast no-op most of the time)
    pro = await asyncio.to_thread(proactive.check_after_message, user_message, final)
    if pro:
        await broadcast({"type": "proactive", "message": pro})

    # Follow-up research task — only if queue isn't already full
    if category in ("research", "web_search", "planning", "agentic_task", "coding"):
        summary = await asyncio.to_thread(task_queue.summary)
        if summary.get("pending", 0) < 10:
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


async def _iter_stream(stream):
    """Wrap synchronous ollama stream iterator for async use."""
    for chunk in stream:
        yield chunk
        await asyncio.sleep(0)  # yield control back to event loop between chunks


async def _run_model_streaming(prompt, model, system, budget, history=None, use_skills=False):
    """Streams tokens. Yields: ("token", text) | ("skill", ev) | ("think", text) | ("done", dict)"""
    import ollama, re

    # Only add skill/final format instruction for categories that need it
    if use_skills:
        user_content = f"Task: {prompt}\n\nUse SKILL: {{...}} or FINAL: <answer>"
    else:
        user_content = prompt

    # Build messages with conversation history
    messages = list(history or [])
    if messages and messages[-1]["role"] == "user":
        messages[-1] = {"role": "user", "content": user_content}
    else:
        messages.append({"role": "user", "content": user_content})
    skill_events, think_events = [], []
    tool_count = 0
    last_reply = ""
    first_call = True
    token_buffer = ""

    while tool_count < 20:
        msgs_with_system = [{"role": "system", "content": system}] + messages

        if first_call:
            first_call = False
            # qwen3 models default to thinking mode — disable for fast chat
            extra = {"think": False} if model.startswith("qwen3") else {}
            stream = await asyncio.to_thread(
                ollama.chat, model=model, messages=msgs_with_system,
                stream=True,
                options={"temperature": 0.7, "num_predict": budget, "num_ctx": 4096},
                **extra
            )
            collected = []
            streaming_started = False
            async for chunk in _iter_stream(stream):
                token = chunk.get("message", {}).get("content", "")
                if not token:
                    continue
                collected.append(token)
                token_buffer += token
                # Buffer the start to detect SKILL:/FINAL: before streaming
                if not streaming_started:
                    if "FINAL:" in token_buffer:
                        # Strip FINAL: prefix and stream the rest
                        token_buffer = token_buffer.split("FINAL:", 1)[1].lstrip()
                        streaming_started = True
                        if token_buffer:
                            yield ("token", token_buffer)
                    elif token_buffer.lstrip().startswith("SKILL:") or token_buffer.lstrip().startswith("{\""):
                        # It's a tool call — don't stream the JSON, let post-stream logic handle it
                        pass  # keep buffering silently
                    elif len(token_buffer) > 40:
                        # Long enough to confirm it's regular content — stream it
                        streaming_started = True
                        yield ("token", token_buffer)
                else:
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
    from autonomous.training_curator import get_training_status
    p = await asyncio.to_thread(user_model.get_display_profile)
    p["assistant_name"] = personality.name
    try:
        ts = await asyncio.to_thread(get_training_status, DB_PATH)
        p["training_total"] = ts.get("total_exchanges", 0)
        p["training_good"] = ts.get("good_quality", 0)
    except Exception:
        pass
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

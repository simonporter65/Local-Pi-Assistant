"""
server.py — Application entry point.

Responsibilities:
  - Define the FastAPI app and lifespan (startup / shutdown)
  - Mount route modules and static files
  - Warm models on startup

All business logic lives in routes/ and state.py.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse

sys.path.insert(0, str(Path(__file__).parent))

from core.log import get_logger
from memory.store import AgentMemory
from memory.user_model import UserModel
from memory.personality import PersonalityConfig
from memory.training_collector import TrainingCollector
from proactive.engine import ProactiveEngine
from skills.registry import SkillRegistry
from autonomous.task_queue import TaskQueue
from autonomous.heartbeat import HeartbeatLoop
from state import AppState

from routes.chat        import router as chat_router
from routes.events      import router as events_router
from routes.personality import router as personality_router
from routes.tasks       import router as tasks_router

logger = get_logger("server")

AGENT_HOME = os.environ.get("AGENT_HOME", str(Path(__file__).parent))
DB_PATH    = os.environ.get("AGENT_DB",   f"{AGENT_HOME}/memory/agent.db")
UI_DIR     = Path(__file__).parent / "ui"


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(f"{AGENT_HOME}/memory",      exist_ok=True)
    os.makedirs(f"{AGENT_HOME}/workspace",   exist_ok=True)
    os.makedirs(f"{AGENT_HOME}/screenshots", exist_ok=True)

    registry    = SkillRegistry()
    memory      = AgentMemory(DB_PATH)
    user_model  = UserModel(memory)
    personality = PersonalityConfig(f"{AGENT_HOME}/memory/personality.json")
    proactive   = ProactiveEngine(user_model, memory, registry)
    task_queue  = TaskQueue(DB_PATH)
    training    = TrainingCollector(DB_PATH)

    arc = AppState(
        registry=registry,
        memory=memory,
        user_model=user_model,
        personality=personality,
        proactive=proactive,
        task_queue=task_queue,
        heartbeat=None,   # set below after arc is created
        training=training,
    )

    heartbeat = HeartbeatLoop(
        task_queue=task_queue,
        registry=registry,
        memory=memory,
        user_model=user_model,
        broadcast_fn=arc.broadcast,
        personality=personality,
    )
    arc.heartbeat = heartbeat
    app.state.arc = arc

    asyncio.create_task(heartbeat.run(), name="heartbeat")

    # Keepalive — ping models every 3 minutes to prevent Ollama eviction
    async def _keepalive():
        import ollama
        while True:
            await asyncio.sleep(180)
            try:
                await asyncio.to_thread(
                    ollama.generate, model="qwen3.5:0.8b",
                    prompt="hi", options={"num_predict": 1, "num_ctx": 1024},
                )
                logger.debug("keepalive: qwen3.5:0.8b warmed")
            except Exception:
                pass
            try:
                await asyncio.to_thread(ollama.embeddings, model="nomic-embed-text", prompt="hi")
                logger.debug("keepalive: nomic-embed-text warmed")
            except Exception:
                pass

    asyncio.create_task(_keepalive(), name="keepalive")

    # Pre-warm model route cache so the first request doesn't subprocess
    from core.router import get_installed_models
    await asyncio.to_thread(get_installed_models)

    # Warm primary models before announcing readiness
    import ollama as _ollama
    try:
        await asyncio.to_thread(
            _ollama.generate, model="qwen3.5:0.8b",
            prompt="hi", options={"num_predict": 1, "num_ctx": 1024},
        )
        logger.info("Warmed qwen3.5:0.8b")
    except Exception as _e:
        logger.warning("Could not warm qwen3.5:0.8b: %s", _e)
    try:
        await asyncio.to_thread(_ollama.embeddings, model="nomic-embed-text", prompt="warmup")
        logger.info("Warmed nomic-embed-text")
    except Exception as _e:
        logger.warning("Could not warm nomic-embed-text: %s", _e)

    # Startup greeting (non-blocking)
    name = personality.name or "Assistant"

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
                    options={"temperature": 0.7, "num_predict": 40, "num_ctx": 512},
                )
                greeting = resp["response"].strip()
                if greeting:
                    await arc.broadcast({"type": "greeting", "message": greeting})
        except Exception as e:
            logger.warning("Startup greeting failed: %s", e)

    asyncio.create_task(_startup_greeting())

    logger.info("Ready → http://localhost:8765")
    logger.info("Assistant: %s | Configured: %s", name, personality.is_configured)
    logger.info("Skills: %d | Tasks: %s", len(registry.skills), task_queue.summary())

    yield

    heartbeat.stop()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

# Include route modules
app.include_router(chat_router)
app.include_router(events_router)
app.include_router(personality_router)
app.include_router(tasks_router)


# ── Core pages ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    arc = request.app.state.arc
    if not arc.personality.is_configured:
        return RedirectResponse("/setup")
    return (UI_DIR / "index.html").read_text()


@app.get("/setup", response_class=HTMLResponse)
async def setup_page():
    return (UI_DIR / "personality.html").read_text()


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8765,
                reload=False, log_level="warning")

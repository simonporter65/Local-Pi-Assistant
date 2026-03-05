"""
routes/tasks.py — Task queue, profile, LoRA, proactive, and skills endpoints.

GET    /tasks              — list tasks
POST   /tasks              — create a task
DELETE /tasks/{task_id}    — cancel a task
GET    /tasks/summary      — queue summary counts
GET    /profile            — user profile + training stats
POST   /lora/opt-in        — respond to personalisation opt-in
GET    /lora/status        — is arc-personal built?
GET    /proactive          — sidebar suggestions
GET    /proactive/push     — push notification message
GET    /skills             — list all loaded skills with descriptions
"""

import asyncio
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from fastapi import APIRouter, Request

from core.log import get_logger

logger = get_logger("tasks_route")
router = APIRouter()


@router.get("/tasks")
async def get_tasks(request: Request, status: str = None):
    arc = request.app.state.arc
    tasks   = await asyncio.to_thread(arc.task_queue.get_all, status)
    summary = await asyncio.to_thread(arc.task_queue.summary)
    return {"tasks": tasks, "summary": summary}


@router.post("/tasks")
async def create_task(request: Request):
    arc = request.app.state.arc
    body = await request.json()
    tid = await asyncio.to_thread(
        arc.task_queue.add,
        title=body.get("title", "User task"),
        description=body.get("description", ""),
        task_type=body.get("task_type", "custom"),
        priority_name=body.get("priority_name", "normal"),
    )
    return {"id": tid}


@router.delete("/tasks/{task_id}")
async def cancel_task(request: Request, task_id: int):
    arc = request.app.state.arc
    await asyncio.to_thread(arc.task_queue.cancel, task_id, "Cancelled by user")
    return {"status": "cancelled"}


@router.get("/tasks/summary")
async def task_summary(request: Request):
    arc = request.app.state.arc
    return await asyncio.to_thread(arc.task_queue.summary)


@router.get("/profile")
async def get_profile(request: Request):
    from autonomous.training_curator import get_training_status
    arc = request.app.state.arc
    db_path = getattr(arc.memory, '_db_path', None)
    p = await asyncio.to_thread(arc.user_model.get_display_profile)
    p["assistant_name"] = arc.personality.name
    try:
        if db_path:
            ts = await asyncio.to_thread(get_training_status, db_path)
            p["training_total"] = ts.get("total_exchanges", 0)
            p["training_good"] = ts.get("good_quality", 0)
    except Exception:
        logger.warning("Failed to load training status", exc_info=True)
    return p


@router.post("/lora/opt-in")
async def lora_opt_in(request: Request, action: str = "yes"):
    """User responds to LoRA training opt-in."""
    arc = request.app.state.arc
    # Resolve DB path — always set on AgentMemory since Phase 4
    db_path = arc.memory._db_path
    db = sqlite3.connect(db_path)
    try:
        if action == "yes":
            db.execute(
                "INSERT OR REPLACE INTO training_meta (key, value) VALUES ('lora_opted_in', '1')"
            )
            db.commit()
            script_path = str(Path(__file__).parent.parent / "scripts" / "lora_train.py")
            arc.task_queue.add(
                title="Build personalised model (arc-personal)",
                description=(
                    f"The user has opted in to on-device personalisation. "
                    f"Run: SKILL: {{\"name\": \"bash_exec\", \"args\": {{\"command\": \"python3 {script_path} --db {db_path}\", \"timeout\": 300}}}} "
                    f"Then report the result. If successful, tell the user to restart the service."
                ),
                task_type="prepare",
                priority_name="low",
            )
            logger.info("LoRA opt-in accepted")
            return {"ok": True, "message": "Personalisation scheduled. Will run overnight."}
        else:
            ask_after = str(date.today() + timedelta(days=30))
            db.execute(
                "INSERT OR REPLACE INTO training_meta (key, value) VALUES ('lora_ask_after', ?)",
                (ask_after,)
            )
            db.commit()
            logger.info("LoRA opt-in snoozed 30 days")
            return {"ok": True, "message": "Snoozed for 30 days."}
    finally:
        db.close()


@router.get("/lora/status")
async def lora_status(request: Request):
    """Return personalisation status: whether arc-personal has been built."""
    arc = request.app.state.arc
    agent_home = str(Path(__file__).parent.parent)
    marker = Path(agent_home) / "memory" / "personal_model.json"
    if marker.exists():
        try:
            info = json.loads(marker.read_text())
            return {"personalised": True, **info}
        except Exception:
            logger.warning("Could not read personal_model.json", exc_info=True)
    return {"personalised": False}


@router.get("/proactive")
async def get_proactive(request: Request):
    arc = request.app.state.arc
    return {"suggestions": await asyncio.to_thread(arc.proactive.get_sidebar_suggestions)}


@router.get("/proactive/push")
async def proactive_push(request: Request):
    arc = request.app.state.arc
    return {"message": await asyncio.to_thread(arc.proactive.get_push_message)}


@router.get("/skills")
async def list_skills(request: Request):
    """Return all loaded skills with name, description, args, and builtin flag."""
    from skills.registry import BUILTIN_SKILLS
    arc = request.app.state.arc
    skills_data = []
    for name, mod in arc.registry.skills.items():
        desc = mod.DESCRIPTION if hasattr(mod, "DESCRIPTION") else {}
        desc_text = desc.get("description", "") if isinstance(desc, dict) else str(desc)
        args = list(desc.get("args", {}).keys()) if isinstance(desc, dict) else []
        skills_data.append({
            "name": name,
            "description": desc_text,
            "args": args,
            "builtin": name in BUILTIN_SKILLS,
        })
    # Custom skills first, then builtins; alphabetical within each group
    skills_data.sort(key=lambda s: (s["builtin"], s["name"]))
    return {"skills": skills_data, "total": len(skills_data)}

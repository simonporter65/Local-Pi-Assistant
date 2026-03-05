"""
routes/personality.py — Personality configuration API.

GET  /personality  — read current config
POST /personality  — save new config (from setup screen)
"""

import asyncio

from fastapi import APIRouter, Request

from core.log import get_logger

logger = get_logger("personality_route")
router = APIRouter()


@router.post("/personality")
async def save_personality(request: Request):
    arc = request.app.state.arc
    config = await request.json()
    await asyncio.to_thread(arc.personality.save, config)

    name = config.get("name", "Assistant")
    await asyncio.to_thread(arc.user_model.set_preference, "assistant_name", name)

    # Queue a personalised greeting to be broadcast when the heartbeat runs
    await asyncio.to_thread(
        arc.task_queue.add,
        title=f"Compose greeting as {name}",
        description=(
            f"Your name is {name}. Personality: {config.get('profile', 'balanced')}. "
            f"Write a warm in-character first greeting (2-3 sentences). "
            f"Save it to workspace as 'greeting.txt'."
        ),
        task_type="prepare",
        priority_name="high",
    )
    logger.info("Personality saved: %s / %s", name, config.get("profile"))
    return {"status": "saved", "name": name}


@router.get("/personality")
async def get_personality(request: Request):
    arc = request.app.state.arc
    return arc.personality.get()

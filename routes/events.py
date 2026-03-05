"""
routes/events.py — Global SSE event stream.

The /events endpoint gives UI clients a long-lived SSE connection that
receives heartbeat status updates, profile changes, proactive notifications,
and any other broadcasts from the server.
"""

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from core.log import get_logger

logger = get_logger("events")
router = APIRouter()


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@router.get("/events")
async def event_stream(request: Request):
    arc = request.app.state.arc
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    arc.broadcast_queues.add(q)

    async def generate():
        try:
            summary = await asyncio.to_thread(arc.task_queue.summary)
            p = arc.personality.get()
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
            arc.broadcast_queues.discard(q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

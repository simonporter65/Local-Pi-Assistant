"""
state.py — Application state container.

Replaces global module-level singletons with a typed dataclass stored on
app.state.arc during the FastAPI lifespan. All route modules access shared
services via `request.app.state.arc` rather than global imports.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Set


@dataclass
class AppState:
    registry:    object          # SkillRegistry
    memory:      object          # AgentMemory
    user_model:  object          # UserModel
    personality: object          # PersonalityConfig
    proactive:   object          # ProactiveEngine
    task_queue:  object          # TaskQueue
    heartbeat:   object          # HeartbeatLoop
    training:    object          # TrainingCollector
    broadcast_queues: Set[asyncio.Queue] = field(default_factory=set)

    async def broadcast(self, event: dict):
        """Broadcast an SSE event to all connected UI clients."""
        dead: Set[asyncio.Queue] = set()
        for q in self.broadcast_queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.add(q)
        for q in dead:
            self.broadcast_queues.discard(q)

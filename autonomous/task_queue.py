"""
autonomous/task_queue.py

The agent's self-managed task queue.
Tasks are stored in SQLite, prioritised, and executed by the heartbeat loop.
The agent itself adds, updates, and reflects on tasks.

Task lifecycle:
  pending → running → done
                    → failed → pending (retry)
                    → cancelled

Task types:
  research      — look something up about the user's interests
  self_improve  — write or improve a skill
  prepare       — pre-compute something the user will likely ask
  remind        — surface a reminder to the user
  reflect       — review recent interactions and update user model
  maintain      — housekeeping (clean logs, check disk, etc.)
  custom        — user-defined or agent-defined arbitrary task
"""

import json
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional


PRIORITIES = {
    "critical": 0,
    "high":     1,
    "normal":   2,
    "low":      3,
    "idle":     4,   # Only runs when nothing else to do
}

TASK_TYPES = [
    "research", "self_improve", "prepare", "remind",
    "reflect", "maintain", "custom"
]


class TaskQueue:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self._ensure_schema()
        self._seed_initial_tasks()

    def _ensure_schema(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                title           TEXT NOT NULL,
                description     TEXT NOT NULL,
                task_type       TEXT DEFAULT 'custom',
                priority        INTEGER DEFAULT 2,
                priority_name   TEXT DEFAULT 'normal',
                status          TEXT DEFAULT 'pending',
                created_at      TEXT,
                scheduled_at    TEXT,
                started_at      TEXT,
                completed_at    TEXT,
                result_summary  TEXT,
                retry_count     INTEGER DEFAULT 0,
                max_retries     INTEGER DEFAULT 2,
                parent_id       INTEGER,
                tags            TEXT DEFAULT '[]',
                context         TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS task_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id     INTEGER NOT NULL,
                timestamp   TEXT,
                event       TEXT,
                detail      TEXT,
                FOREIGN KEY(task_id) REFERENCES tasks(id)
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_status_priority
                ON tasks(status, priority, scheduled_at);
        """)
        self.db.commit()

    def _seed_initial_tasks(self):
        """Seed the queue with bootstrap tasks if completely empty."""
        count = self.db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        if count > 0:
            return

        initial_tasks = [
            {
                "title": "Introduce myself to the user",
                "description": (
                    "Send a warm welcome message to the user explaining what I can do, "
                    "that I'm running privately on their device, and ask them a few questions "
                    "to start building my understanding of them."
                ),
                "task_type": "prepare",
                "priority_name": "high",
                "tags": ["onboarding"],
            },
            {
                "title": "Learn about the Raspberry Pi 5 and my own capabilities",
                "description": (
                    "Research what I can do on this hardware. Check available disk space, "
                    "RAM, which models are loaded, what skills I have. Build a self-inventory "
                    "so I can accurately describe my capabilities to the user."
                ),
                "task_type": "reflect",
                "priority_name": "normal",
                "tags": ["self-awareness"],
            },
            {
                "title": "Write a 'send_notification' skill",
                "description": (
                    "Write a skill that can send desktop or browser notifications to the user. "
                    "Try: notify-send on Linux, or a lightweight WebSocket push to the UI. "
                    "This will let me proactively alert the user to things they care about."
                ),
                "task_type": "self_improve",
                "priority_name": "normal",
                "tags": ["skills"],
            },
            {
                "title": "Write a 'calendar_check' skill",
                "description": (
                    "Write a skill that can read local calendar files (iCal format) or "
                    "query a CalDAV server. Store in workspace. This enables reminders and "
                    "schedule awareness."
                ),
                "task_type": "self_improve",
                "priority_name": "low",
                "tags": ["skills", "calendar"],
            },
            {
                "title": "Reflect on what I know and what I should learn next",
                "description": (
                    "Review my current skill set, the user profile so far, and recent interactions. "
                    "Generate 5 new tasks that would make me more useful to this specific user. "
                    "Add them to my task queue."
                ),
                "task_type": "reflect",
                "priority_name": "low",
                "scheduled_at": _in_hours(2),
                "tags": ["meta"],
            },
        ]

        for task in initial_tasks:
            self.add(**task)

    # ── CRUD ─────────────────────────────────────────────────────────────

    def add(
        self,
        title: str,
        description: str,
        task_type: str = "custom",
        priority_name: str = "normal",
        scheduled_at: Optional[str] = None,
        tags: list = None,
        context: dict = None,
        parent_id: Optional[int] = None,
        max_retries: int = 2,
    ) -> int:
        priority = PRIORITIES.get(priority_name, 2)
        cursor = self.db.execute(
            """INSERT INTO tasks
               (title, description, task_type, priority, priority_name,
                status, created_at, scheduled_at, tags, context, parent_id, max_retries)
               VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)""",
            (
                title, description, task_type, priority, priority_name,
                datetime.now().isoformat(),
                scheduled_at or datetime.now().isoformat(),
                json.dumps(tags or []),
                json.dumps(context or {}),
                parent_id,
                max_retries,
            )
        )
        self.db.commit()
        task_id = cursor.lastrowid
        self._log(task_id, "created", f"priority={priority_name}, type={task_type}")
        return task_id

    def next_pending(self) -> Optional[dict]:
        """Get the highest-priority pending task that's due now."""
        now = datetime.now().isoformat()
        row = self.db.execute(
            """SELECT * FROM tasks
               WHERE status='pending' AND scheduled_at <= ?
               ORDER BY priority ASC, created_at ASC
               LIMIT 1""",
            (now,)
        ).fetchone()
        return _row_to_dict(row) if row else None

    def start(self, task_id: int):
        self.db.execute(
            "UPDATE tasks SET status='running', started_at=? WHERE id=?",
            (datetime.now().isoformat(), task_id)
        )
        self.db.commit()
        self._log(task_id, "started")

    def complete(self, task_id: int, result_summary: str):
        self.db.execute(
            """UPDATE tasks SET status='done', completed_at=?, result_summary=?
               WHERE id=?""",
            (datetime.now().isoformat(), result_summary[:1000], task_id)
        )
        self.db.commit()
        self._log(task_id, "completed", result_summary[:200])

    def fail(self, task_id: int, reason: str):
        row = self.db.execute(
            "SELECT retry_count, max_retries FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        if not row:
            return

        retry_count, max_retries = row
        if retry_count < max_retries:
            # Retry with exponential backoff
            delay_minutes = 5 * (2 ** retry_count)
            retry_at = (datetime.now() + timedelta(minutes=delay_minutes)).isoformat()
            self.db.execute(
                """UPDATE tasks SET status='pending', retry_count=retry_count+1,
                   scheduled_at=? WHERE id=?""",
                (retry_at, task_id)
            )
            self._log(task_id, "retry_scheduled", f"attempt {retry_count+1}, retry in {delay_minutes}m")
        else:
            self.db.execute(
                "UPDATE tasks SET status='failed', completed_at=?, result_summary=? WHERE id=?",
                (datetime.now().isoformat(), f"FAILED: {reason}", task_id)
            )
            self._log(task_id, "failed", reason)
        self.db.commit()

    def cancel(self, task_id: int, reason: str = ""):
        self.db.execute(
            "UPDATE tasks SET status='cancelled', completed_at=? WHERE id=?",
            (datetime.now().isoformat(), task_id)
        )
        self.db.commit()
        self._log(task_id, "cancelled", reason)

    def reschedule(self, task_id: int, when: str):
        self.db.execute(
            "UPDATE tasks SET status='pending', scheduled_at=? WHERE id=?",
            (when, task_id)
        )
        self.db.commit()
        self._log(task_id, "rescheduled", when)

    def pause_running(self):
        """Pause any running task when user interaction takes priority."""
        self.db.execute(
            "UPDATE tasks SET status='pending', started_at=NULL WHERE status='running'"
        )
        self.db.commit()

    def resume_paused(self):
        """No-op — paused tasks automatically become pending again."""
        pass

    def pending_count(self) -> int:
        return self.db.execute(
            "SELECT COUNT(*) FROM tasks WHERE status='pending'"
        ).fetchone()[0]

    def get_all(self, status: str = None, limit: int = 50) -> list:
        if status:
            rows = self.db.execute(
                "SELECT * FROM tasks WHERE status=? ORDER BY priority ASC, created_at ASC LIMIT ?",
                (status, limit)
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM tasks ORDER BY priority ASC, created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_recent_completed(self, n: int = 10) -> list:
        rows = self.db.execute(
            "SELECT * FROM tasks WHERE status='done' ORDER BY completed_at DESC LIMIT ?", (n,)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def summary(self) -> dict:
        counts = {}
        for status in ("pending", "running", "done", "failed", "cancelled"):
            counts[status] = self.db.execute(
                "SELECT COUNT(*) FROM tasks WHERE status=?", (status,)
            ).fetchone()[0]
        return counts

    def _log(self, task_id: int, event: str, detail: str = ""):
        self.db.execute(
            "INSERT INTO task_log (task_id, timestamp, event, detail) VALUES (?, ?, ?, ?)",
            (task_id, datetime.now().isoformat(), event, detail)
        )
        self.db.commit()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _in_hours(n: float) -> str:
    return (datetime.now() + timedelta(hours=n)).isoformat()

def _in_minutes(n: float) -> str:
    return (datetime.now() + timedelta(minutes=n)).isoformat()

def _row_to_dict(row) -> dict:
    if not row:
        return None
    keys = [
        "id", "title", "description", "task_type", "priority", "priority_name",
        "status", "created_at", "scheduled_at", "started_at", "completed_at",
        "result_summary", "retry_count", "max_retries", "parent_id", "tags", "context"
    ]
    d = dict(zip(keys, row))
    try:
        d["tags"] = json.loads(d.get("tags") or "[]")
        d["context"] = json.loads(d.get("context") or "{}")
    except Exception:
        pass
    return d

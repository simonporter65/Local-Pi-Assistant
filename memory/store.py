"""
memory/store.py
Persistent agent memory using SQLite on NVMe.
Supports semantic search via embeddings (nomic-embed-text).
"""

import sqlite3
import json
import os
import numpy as np
from datetime import datetime
from typing import Optional

DB_PATH = os.environ.get("AGENT_DB", "/mnt/nvme/agent/memory/agent.db")

# Fallback to local if NVMe not mounted yet
if not os.path.exists(os.path.dirname(DB_PATH)):
    os.makedirs("memory", exist_ok=True)
    DB_PATH = "memory/agent.db"


class AgentMemory:
    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")  # Better concurrent writes
        self.db.execute("PRAGMA synchronous=NORMAL")  # Faster on NVMe
        self._init_schema()
        print(f"[MEMORY] Database: {db_path}")

    def _init_schema(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS interactions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT    NOT NULL,
                user_input   TEXT    NOT NULL,
                intent       TEXT,
                model_used   TEXT,
                output       TEXT,
                success      INTEGER DEFAULT 0,
                tool_calls   INTEGER DEFAULT 0,
                duration_ms  INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id   INTEGER NOT NULL,
                embedding        TEXT    NOT NULL,
                FOREIGN KEY(interaction_id) REFERENCES interactions(id)
            );

            CREATE TABLE IF NOT EXISTS skills_log (
                name         TEXT    PRIMARY KEY,
                description  TEXT,
                created_at   TEXT,
                call_count   INTEGER DEFAULT 0,
                fail_count   INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS agent_state (
                key         TEXT PRIMARY KEY,
                value       TEXT,
                updated_at  TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_interactions_timestamp
                ON interactions(timestamp);
        """)
        self.db.commit()

    # ── Logging ─────────────────────────────────────

    def log_interaction(
        self,
        user_input: str,
        intent: dict,
        model: str,
        output: str,
        success: bool,
        tool_calls: int,
        duration_ms: int,
    ) -> int:
        cursor = self.db.execute(
            """INSERT INTO interactions
               (timestamp, user_input, intent, model_used, output, success, tool_calls, duration_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                user_input,
                json.dumps(intent),
                model,
                output,
                int(success),
                tool_calls,
                duration_ms,
            )
        )
        self.db.commit()
        interaction_id = cursor.lastrowid

        # Embed asynchronously (best-effort)
        self._embed_and_store(interaction_id, user_input + " " + output[:400])

        return interaction_id

    def log_skill_call(self, name: str, description: str, success: bool):
        self.db.execute("""
            INSERT INTO skills_log (name, description, created_at, call_count, fail_count)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(name) DO UPDATE SET
                call_count = call_count + 1,
                fail_count = fail_count + ?
        """, (name, description, datetime.now().isoformat(), int(not success), int(not success)))
        self.db.commit()

    # ── Embedding ────────────────────────────────────

    def _embed_and_store(self, interaction_id: int, text: str):
        try:
            import ollama
            response = ollama.embeddings(
                model="nomic-embed-text",
                prompt=text[:1000]  # Keep it reasonable
            )
            embedding = response["embedding"]
            self.db.execute(
                "INSERT INTO embeddings (interaction_id, embedding) VALUES (?, ?)",
                (interaction_id, json.dumps(embedding))
            )
            self.db.commit()
        except Exception:
            pass  # Embeddings are best-effort; don't fail the whole interaction

    def semantic_search(self, query: str, top_k: int = 5) -> list:
        """Find past interactions semantically similar to the query."""
        try:
            import ollama
            q_embed = ollama.embeddings(
                model="nomic-embed-text",
                prompt=query[:500]
            )["embedding"]
            q_vec = np.array(q_embed)

            rows = self.db.execute("""
                SELECT i.user_input, i.output, i.intent, e.embedding
                FROM interactions i
                JOIN embeddings e ON i.id = e.interaction_id
                ORDER BY i.id DESC
                LIMIT 300
            """).fetchall()

            scored = []
            for row in rows:
                try:
                    emb = np.array(json.loads(row[3]))
                    # Cosine similarity
                    score = float(
                        np.dot(q_vec, emb) /
                        (np.linalg.norm(q_vec) * np.linalg.norm(emb) + 1e-9)
                    )
                    scored.append((score, row))
                except Exception:
                    continue

            scored.sort(reverse=True)
            return [
                {
                    "input":  r[1][0],
                    "output": r[1][1][:300] if r[1][1] else "",
                    "intent": r[1][2],
                }
                for r in scored[:top_k]
            ]

        except Exception:
            # Fall back to recency-based context
            return self._recent_interactions(top_k)

    def _recent_interactions(self, n: int) -> list:
        rows = self.db.execute(
            "SELECT user_input, output, intent FROM interactions ORDER BY id DESC LIMIT ?",
            (n,)
        ).fetchall()
        return [{"input": r[0], "output": r[1] or "", "intent": r[2]} for r in rows]

    # ── State K/V ────────────────────────────────────

    def get_state(self, key: str):
        row = self.db.execute(
            "SELECT value FROM agent_state WHERE key=?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def set_state(self, key: str, value):
        self.db.execute("""
            INSERT OR REPLACE INTO agent_state (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (key, json.dumps(value), datetime.now().isoformat()))
        self.db.commit()

    # ── Stats ─────────────────────────────────────────

    def stats(self) -> dict:
        total = self.db.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        success = self.db.execute("SELECT COUNT(*) FROM interactions WHERE success=1").fetchone()[0]
        avg_ms = self.db.execute("SELECT AVG(duration_ms) FROM interactions").fetchone()[0]
        top_models = self.db.execute("""
            SELECT model_used, COUNT(*) as n
            FROM interactions
            GROUP BY model_used
            ORDER BY n DESC
            LIMIT 5
        """).fetchall()

        return {
            "total_interactions": total,
            "success_rate": f"{success/total:.0%}" if total else "0%",
            "avg_duration_ms": int(avg_ms) if avg_ms else 0,
            "top_models": dict(top_models),
        }

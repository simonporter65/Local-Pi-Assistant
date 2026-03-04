"""
memory/training_collector.py

Collects implicit feedback from conversations to build LoRA training data.
Scores exchanges based on the user's next message — no explicit ratings needed.
Completely invisible during normal use.

Pipeline: Collection → Curation → Training (overnight) → Deployment
Status: Layer 1 of 4 implemented.
"""

import json
import re
import sqlite3
from datetime import datetime
from typing import Optional


# Signals that the PREVIOUS response was good
POSITIVE_SIGNALS = {
    "thanks", "thank you", "perfect", "exactly", "great", "brilliant",
    "yes", "correct", "right", "good", "nice", "helpful", "awesome",
    "that's it", "that's right", "that works", "got it", "makes sense",
    "interesting", "love it", "excellent", "wonderful", "cheers"
}

# Signals that the PREVIOUS response was bad
NEGATIVE_SIGNALS = {
    "no", "wrong", "incorrect", "that's not", "not what", "try again",
    "actually", "wait", "hmm", "that's wrong", "not right", "missed",
    "didn't", "that isn't", "no that", "not quite", "close but",
    "not exactly", "misunderstood", "i meant", "what i meant"
}

# Minimum quality score to include in training data
MIN_QUALITY_SCORE = 0.6


def _ensure_tables(db: sqlite3.Connection):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS training_exchanges (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            system_prompt   TEXT NOT NULL,
            user_message    TEXT NOT NULL,
            assistant_response TEXT NOT NULL,
            next_message    TEXT,
            implicit_score  REAL DEFAULT NULL,
            quality_score   REAL DEFAULT NULL,
            included        INTEGER DEFAULT 0,
            session_id      TEXT,
            model           TEXT
        );

        CREATE TABLE IF NOT EXISTS training_meta (
            key     TEXT PRIMARY KEY,
            value   TEXT
        );
    """)
    db.commit()


class TrainingCollector:
    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        _ensure_tables(self.db)
        self._last_exchange_id: Optional[int] = None

    def record_exchange(
        self,
        system_prompt: str,
        user_message: str,
        assistant_response: str,
        session_id: str = "default",
        model: str = "",
    ) -> int:
        """Record a conversation exchange. Returns the exchange ID."""
        cur = self.db.execute(
            """INSERT INTO training_exchanges
               (timestamp, system_prompt, user_message, assistant_response, session_id, model)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (datetime.utcnow().isoformat(), system_prompt[:2000],
             user_message[:500], assistant_response[:2000], session_id, model)
        )
        self.db.commit()
        self._last_exchange_id = cur.lastrowid
        return cur.lastrowid

    def score_previous_exchange(self, next_user_message: str, session_id: str = "default"):
        """
        Score the most recent exchange based on the user's follow-up message.
        Called at the START of each new chat message.
        """
        # Find the most recent unscored exchange for this session
        row = self.db.execute(
            """SELECT id FROM training_exchanges
               WHERE session_id = ? AND implicit_score IS NULL
               ORDER BY id DESC LIMIT 1""",
            (session_id,)
        ).fetchone()

        if not row:
            return

        score = self._infer_score(next_user_message)
        self.db.execute(
            "UPDATE training_exchanges SET next_message=?, implicit_score=? WHERE id=?",
            (next_user_message[:300], score, row["id"])
        )
        self.db.commit()

    def _infer_score(self, next_message: str) -> float:
        """Infer quality score from the user's next message. Returns 0.0-1.0."""
        msg = next_message.lower().strip()
        words = set(re.findall(r'\b\w+\b', msg))

        positive_hits = words & POSITIVE_SIGNALS
        negative_hits = words & NEGATIVE_SIGNALS

        # Phrase-level checks
        negative_phrases = ["that's not what", "i meant", "not what i", "try again",
                           "that's wrong", "you misunderstood", "no that's"]
        positive_phrases = ["that's perfect", "exactly right", "that's it", "makes sense",
                           "that works", "great answer", "love that"]

        has_negative_phrase = any(p in msg for p in negative_phrases)
        has_positive_phrase = any(p in msg for p in positive_phrases)

        if has_negative_phrase or (negative_hits and not positive_hits):
            return 0.1
        if has_positive_phrase or len(positive_hits) >= 2:
            return 0.95
        if positive_hits and not negative_hits:
            return 0.8
        if negative_hits:
            return 0.3

        # Neutral — conversation continues naturally, mild positive signal
        return 0.6

    def get_stats(self) -> dict:
        """Return collection statistics."""
        total = self.db.execute("SELECT COUNT(*) FROM training_exchanges").fetchone()[0]
        scored = self.db.execute(
            "SELECT COUNT(*) FROM training_exchanges WHERE implicit_score IS NOT NULL"
        ).fetchone()[0]
        good = self.db.execute(
            "SELECT COUNT(*) FROM training_exchanges WHERE implicit_score >= 0.6"
        ).fetchone()[0]
        included = self.db.execute(
            "SELECT COUNT(*) FROM training_exchanges WHERE included = 1"
        ).fetchone()[0]
        opted_in = self.db.execute(
            "SELECT value FROM training_meta WHERE key='lora_opted_in'"
        ).fetchone()

        return {
            "total_exchanges": total,
            "scored": scored,
            "good_quality": good,
            "curated": included,
            "opted_in": opted_in["value"] == "1" if opted_in else False,
            "ready_to_curate": good >= 20,
            "ready_to_train": included >= 50,
        }

    def set_opted_in(self, value: bool):
        self.db.execute(
            "INSERT OR REPLACE INTO training_meta (key, value) VALUES ('lora_opted_in', ?)",
            ("1" if value else "0",)
        )
        self.db.commit()

    def is_opted_in(self) -> bool:
        row = self.db.execute(
            "SELECT value FROM training_meta WHERE key='lora_opted_in'"
        ).fetchone()
        return row and row["value"] == "1"

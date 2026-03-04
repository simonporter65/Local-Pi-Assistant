"""
autonomous/training_curator.py

Heartbeat task that curates collected exchanges into clean JSONL training data.
Runs at low/idle priority during quiet periods.
Also manages the opt-in conversation with the user.

Pipeline: Collection ✓ → Curation ← HERE → Training → Deployment
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path


LORA_DIR = Path("/mnt/nvme/lora")
TRAINING_DATA_PATH = LORA_DIR / "training_data.jsonl"
MIN_RESPONSE_LEN = 20
MAX_RESPONSE_LEN = 1500
OPT_IN_THRESHOLD = 50   # good exchanges before asking user to opt in
TRAIN_THRESHOLD  = 50   # curated examples before training is possible


def curate_training_data(db_path: str) -> dict:
    """
    Review collected exchanges, score quality, export to JSONL.
    Returns summary of what was processed.
    """
    db = sqlite3.connect(db_path, check_same_thread=False)
    db.row_factory = sqlite3.Row

    # Find good unprocessed exchanges
    rows = db.execute("""
        SELECT id, system_prompt, user_message, assistant_response, implicit_score, model
        FROM training_exchanges
        WHERE implicit_score >= 0.6
        AND included = 0
        AND length(assistant_response) >= ?
        AND length(assistant_response) <= ?
        ORDER BY implicit_score DESC
        LIMIT 100
    """, (MIN_RESPONSE_LEN, MAX_RESPONSE_LEN)).fetchall()

    if not rows:
        return {"curated": 0, "total_curated": _count_curated(db)}

    LORA_DIR.mkdir(parents=True, exist_ok=True)

    curated = 0
    with open(TRAINING_DATA_PATH, "a") as f:
        for row in rows:
            # Quality checks
            response = row["assistant_response"]

            # Skip error messages
            if "something went wrong" in response.lower():
                continue
            if "please try again" in response.lower():
                continue

            # Skip very repetitive responses
            words = response.lower().split()
            if len(set(words)) < len(words) * 0.3:
                continue

            # Format as chat training example
            example = {
                "messages": [
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["user_message"]},
                    {"role": "assistant", "content": response},
                ],
                "metadata": {
                    "score": row["implicit_score"],
                    "model": row["model"],
                    "timestamp": datetime.utcnow().isoformat(),
                }
            }
            f.write(json.dumps(example) + "\n")
            db.execute("UPDATE training_exchanges SET included=1 WHERE id=?", (row["id"],))
            curated += 1

    db.commit()
    total = _count_curated(db)
    db.close()

    return {"curated": curated, "total_curated": total}


def _count_curated(db) -> int:
    return db.execute(
        "SELECT COUNT(*) FROM training_exchanges WHERE included=1"
    ).fetchone()[0]


def should_ask_opt_in(db_path: str) -> bool:
    """Returns True if we have enough data and haven't asked yet."""
    db = sqlite3.connect(db_path, check_same_thread=False)
    db.row_factory = sqlite3.Row

    # Already opted in or explicitly declined recently
    meta = db.execute(
        "SELECT value FROM training_meta WHERE key IN ('lora_opted_in', 'lora_ask_after')"
    ).fetchall()
    meta_dict = {r["key"]: r["value"] for r in meta} if meta else {}

    if meta_dict.get("lora_opted_in") == "1":
        db.close()
        return False

    # Check if we've been told to wait
    ask_after = meta_dict.get("lora_ask_after")
    if ask_after:
        from datetime import date
        if str(date.today()) < ask_after:
            db.close()
            return False

    # Count good exchanges
    good = db.execute(
        "SELECT COUNT(*) FROM training_exchanges WHERE implicit_score >= 0.6"
    ).fetchone()[0]
    db.close()

    return good >= OPT_IN_THRESHOLD


def snooze_opt_in(db_path: str, days: int = 30):
    """Don't ask again for N days."""
    from datetime import date, timedelta
    db = sqlite3.connect(db_path)
    ask_after = str(date.today() + timedelta(days=days))
    db.execute(
        "INSERT OR REPLACE INTO training_meta (key, value) VALUES ('lora_ask_after', ?)",
        (ask_after,)
    )
    db.commit()
    db.close()


def get_training_status(db_path: str) -> dict:
    """Full status for heartbeat reporting."""
    db = sqlite3.connect(db_path, check_same_thread=False)
    db.row_factory = sqlite3.Row

    total = db.execute("SELECT COUNT(*) FROM training_exchanges").fetchone()[0]
    good = db.execute(
        "SELECT COUNT(*) FROM training_exchanges WHERE implicit_score >= 0.6"
    ).fetchone()[0]
    curated = _count_curated(db)
    opted_in = db.execute(
        "SELECT value FROM training_meta WHERE key='lora_opted_in'"
    ).fetchone()
    db.close()

    return {
        "total_exchanges": total,
        "good_quality": good,
        "curated": curated,
        "opted_in": opted_in and opted_in["value"] == "1",
        "ready_to_curate": good >= 20,
        "ready_to_train": curated >= TRAIN_THRESHOLD,
    }

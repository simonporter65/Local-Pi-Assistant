#!/usr/bin/env python3
"""
scripts/lora_train.py

On-device personalisation for ARC on Raspberry Pi 5.

True LoRA fine-tuning of quantized GGUF models requires FP32 weights and is
not feasible on Pi 5 hardware. Instead this script creates a personalised
Ollama model using curated conversation examples embedded in the system prompt.
This achieves meaningful personalisation that works with the existing Ollama stack.

Pipeline:
  1. Read top-N curated exchanges from the database
  2. Select diverse, high-quality examples
  3. Write an Ollama Modelfile with examples as few-shot context
  4. Run `ollama create arc-personal` to build the custom model
  5. Write a marker file so the server switches to arc-personal for chat

Usage (run from project root):
  python3 scripts/lora_train.py [--db /path/to/agent.db] [--base llama3.2:3b] [--dry-run]
"""

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).parent.parent
DEFAULT_DB    = PROJECT_ROOT / "memory" / "agent.db"
MODELFILE_PATH = PROJECT_ROOT / "memory" / "arc-personal.Modelfile"
MARKER_FILE   = PROJECT_ROOT / "memory" / "personal_model.json"
MODEL_NAME    = "arc-personal"
BASE_MODEL    = "llama3.2:3b"

MAX_EXAMPLES  = 10   # Few-shot examples in the Modelfile (context budget)
MIN_RESPONSE  = 30   # Minimum response length to include
MIN_SCORE     = 0.65 # Minimum quality score


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Create personalised ARC model")
    parser.add_argument("--db",      default=str(DEFAULT_DB), help="Path to agent.db")
    parser.add_argument("--base",    default=BASE_MODEL,      help="Base Ollama model")
    parser.add_argument("--dry-run", action="store_true",     help="Show plan, don't create model")
    args = parser.parse_args()

    db_path = args.db
    base    = args.base

    print(f"[lora_train] ARC Personalisation Runner")
    print(f"[lora_train] DB:         {db_path}")
    print(f"[lora_train] Base model: {base}")
    print(f"[lora_train] Target:     {MODEL_NAME}")
    print()

    # Check database
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}")
        print("       Run the assistant first to collect training data.")
        sys.exit(1)

    # Check Ollama is available
    if not shutil.which("ollama"):
        print("ERROR: ollama not found in PATH")
        sys.exit(1)

    # Check base model is available
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    available = [line.split()[0] for line in result.stdout.strip().splitlines()[1:] if line.strip()]
    if base not in available:
        print(f"ERROR: Base model '{base}' not installed.")
        print(f"       Run: ollama pull {base}")
        sys.exit(1)
    print(f"✓ Base model '{base}' is available")

    # Load training data
    examples = load_examples(db_path)
    if not examples:
        print("No curated training examples found yet.")
        print(f"Need {MIN_SCORE:.0%}+ quality exchanges in training_exchanges table.")
        sys.exit(0)

    print(f"✓ Found {len(examples)} training examples (using top {min(MAX_EXAMPLES, len(examples))})")

    # Load user profile
    profile = load_profile(db_path)
    if profile:
        print(f"✓ User profile: {', '.join(k for k in profile)}")

    # Build Modelfile
    modelfile = build_modelfile(base, examples[:MAX_EXAMPLES], profile)
    print(f"\n[Modelfile preview — first 500 chars]\n{modelfile[:500]}...\n")

    if args.dry_run:
        print("[dry-run] Would write Modelfile and run: ollama create arc-personal")
        return

    # Write Modelfile
    MODELFILE_PATH.write_text(modelfile)
    print(f"✓ Modelfile written to {MODELFILE_PATH}")

    # Create the model
    print(f"\nRunning: ollama create {MODEL_NAME} -f {MODELFILE_PATH}")
    print("This may take 1-2 minutes...\n")

    proc = subprocess.run(
        ["ollama", "create", MODEL_NAME, "-f", str(MODELFILE_PATH)],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        print(f"ERROR: ollama create failed:\n{proc.stderr}")
        sys.exit(1)

    print(f"✓ Model '{MODEL_NAME}' created successfully")

    # Write marker file so server.py knows to use arc-personal
    marker = {
        "model":      MODEL_NAME,
        "base":       base,
        "examples":   len(examples[:MAX_EXAMPLES]),
        "created_at": datetime.now().isoformat(),
        "profile_keys": list(profile.keys()),
    }
    MARKER_FILE.write_text(json.dumps(marker, indent=2))
    print(f"✓ Marker file written: {MARKER_FILE}")
    print()
    print(f"Done! The server will use '{MODEL_NAME}' for general chat on next restart.")
    print(f"Run: sudo systemctl restart arc")


# ── Training data ─────────────────────────────────────────────────────────────

def load_examples(db_path: str) -> list:
    """Load highest-quality curated exchanges from the database."""
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute("""
            SELECT user_message, assistant_response, implicit_score
            FROM training_exchanges
            WHERE implicit_score >= ?
              AND included = 1
              AND length(assistant_response) >= ?
              AND assistant_response NOT LIKE '%went wrong%'
              AND assistant_response NOT LIKE '%please try again%'
            ORDER BY implicit_score DESC
            LIMIT ?
        """, (MIN_SCORE, MIN_RESPONSE, MAX_EXAMPLES * 3)).fetchall()

        # De-duplicate by selecting diverse examples (simple topic hashing)
        seen_starters = set()
        examples = []
        for row in rows:
            starter = row["user_message"][:20].lower().strip()
            if starter not in seen_starters:
                seen_starters.add(starter)
                examples.append({
                    "user":      row["user_message"][:400],
                    "assistant": row["assistant_response"][:600],
                    "score":     row["implicit_score"],
                })
            if len(examples) >= MAX_EXAMPLES:
                break

        return examples
    except Exception as e:
        print(f"[WARN] Could not load examples: {e}")
        return []
    finally:
        db.close()


def load_profile(db_path: str) -> dict:
    """Load user profile facts from the database."""
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute("""
            SELECT category, fact, confidence
            FROM user_facts
            WHERE confidence > 0.6
            ORDER BY confidence DESC, updated_at DESC
        """).fetchall()

        profile = {}
        for row in rows:
            cat = row["category"]
            if cat not in profile:
                profile[cat] = []
            if len(profile[cat]) < 2:
                profile[cat].append(row["fact"])

        return profile
    except Exception as e:
        print(f"[WARN] Could not load profile: {e}")
        return {}
    finally:
        db.close()


# ── Modelfile builder ─────────────────────────────────────────────────────────

def build_modelfile(base: str, examples: list, profile: dict) -> str:
    """Build an Ollama Modelfile with personalised system prompt and examples."""

    # Build system prompt
    system_parts = [
        "You are ARC, a local AI personal assistant running on a Raspberry Pi 5.",
        "You are private, fast, and genuinely personal — you know this user well.",
        "",
    ]

    if profile:
        system_parts.append("What you know about this user:")
        for cat, facts in profile.items():
            system_parts.append(f"- {cat.capitalize()}: {', '.join(facts)}")
        system_parts.append("")

    system_parts += [
        "Guidelines:",
        "- Be warm and direct — you know this person",
        "- Give practical, concise answers unless depth is requested",
        "- Remember context from this conversation",
        "- You run locally: no internet, no cloud, fully private",
    ]

    system_prompt = "\n".join(system_parts)

    # Format examples as Modelfile MESSAGE blocks
    example_lines = []
    for ex in examples:
        # Escape any special characters in user/assistant content
        user_msg = ex["user"].replace('"', '\\"')
        asst_msg = ex["assistant"].replace('"', '\\"')
        example_lines.append(f'MESSAGE user "{user_msg}"')
        example_lines.append(f'MESSAGE assistant "{asst_msg}"')
        example_lines.append("")

    modelfile = f"""FROM {base}

SYSTEM """{system_prompt}"""

PARAMETER temperature 0.7
PARAMETER num_ctx 4096
PARAMETER num_predict 1024

"""

    if example_lines:
        modelfile += "# Few-shot examples from curated conversations\n"
        modelfile += "\n".join(example_lines)

    return modelfile


if __name__ == "__main__":
    main()

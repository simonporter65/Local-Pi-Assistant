"""
skills/set_personality.py

Read and update the assistant's personality configuration.
Use this when the user asks to change your name, personality traits,
or any flavor setting (humor, warmth, sass, verbosity, chaos).
"""

import json
import os
from datetime import datetime
from pathlib import Path

DESCRIPTION = {
    "description": (
        "Read or update personality settings. Use when the user asks you to change your name, "
        "be more/less funny, warmer, sassier, more concise/verbose, or more chaotic/creative. "
        "Call with action='get' to read current settings, action='set' to update them."
    ),
    "args": {
        "action": "str — 'get' to read current settings, 'set' to update",
        "name":       "str (optional) — new display name for the assistant",
        "humor":      "int 0-100 (optional) — humor level (50=neutral, 80=very witty)",
        "warmth":     "int 0-100 (optional) — warmth/encouragement (50=neutral, 80=warm)",
        "sass":       "int 0-100 (optional) — sass / alternative perspective (50=neutral, 80=feisty)",
        "verbosity":  "int 0-100 (optional) — verbosity (20=very concise, 80=thorough)",
        "chaos":      "int 0-100 (optional) — creative chaos / unexpected approaches (20=safe, 80=wild)",
    },
}

PERSONALITY_FILE = os.environ.get(
    "PERSONALITY_FILE",
    str(Path(__file__).parent.parent / "memory" / "personality.json"),
)

FLAVOR_KEYS = {"humor", "warmth", "sass", "verbosity", "chaos"}

DEFAULT_FLAVORS = {"humor": 40, "warmth": 60, "sass": 30, "verbosity": 50, "chaos": 20}


def _build_prompt(name: str, flavors: dict) -> str:
    """Build personality prompt from name + flavor values (mirrors PersonalityConfig)."""
    humor     = flavors.get("humor", 50)
    warmth    = flavors.get("warmth", 50)
    sass      = flavors.get("sass", 50)
    verbosity = flavors.get("verbosity", 50)
    chaos     = flavors.get("chaos", 20)

    lines = [f"Your name is {name}.\n\nYour personality:"]
    if humor > 50:
        lines.append("- You are witty with good timing. Use humor naturally.")
    if warmth > 50:
        lines.append("- You are warm and encouraging. You are on the user's side.")
    if sass > 50:
        lines.append("- You offer alternative perspectives when useful.")
    if verbosity < 30:
        lines.append("- Be concise. Every word must earn its place.")
    elif verbosity > 70:
        lines.append("- Be thorough. Don't truncate useful context.")
    if chaos > 65:
        lines.append("- Take creative unexpected approaches sometimes.")

    lines.append(f"\nAlways introduce yourself as {name} if asked who you are.")
    return "\n".join(lines)


def _load_config() -> dict:
    try:
        if os.path.exists(PERSONALITY_FILE):
            with open(PERSONALITY_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {
        "name": "Assistant",
        "flavors": dict(DEFAULT_FLAVORS),
        "profile": "Balanced",
        "configured": False,
    }


def _save_config(cfg: dict):
    os.makedirs(os.path.dirname(PERSONALITY_FILE), exist_ok=True)
    cfg["personality_prompt"] = _build_prompt(cfg.get("name", "Assistant"), cfg.get("flavors", {}))
    cfg["saved_at"] = datetime.now().isoformat()
    cfg["configured"] = True
    with open(PERSONALITY_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def run(action: str = "get", **kwargs) -> str:
    if action == "get":
        cfg = _load_config()
        name = cfg.get("name", "Assistant")
        flavors = cfg.get("flavors", DEFAULT_FLAVORS)
        profile = cfg.get("profile", "Balanced")
        lines = [f"Current personality — {name} ({profile}):"]
        for key in ("humor", "warmth", "sass", "verbosity", "chaos"):
            val = flavors.get(key, DEFAULT_FLAVORS.get(key, 50))
            bar = "█" * (val // 10) + "░" * (10 - val // 10)
            lines.append(f"  {key:10s} [{bar}] {val}/100")
        return "\n".join(lines)

    if action == "set":
        cfg = _load_config()
        changed = []

        new_name = kwargs.get("name")
        if new_name:
            cfg["name"] = str(new_name).strip()
            changed.append(f"name → {cfg['name']}")

        flavors = cfg.setdefault("flavors", dict(DEFAULT_FLAVORS))
        for key in FLAVOR_KEYS:
            if key in kwargs and kwargs[key] is not None:
                try:
                    val = max(0, min(100, int(kwargs[key])))
                    flavors[key] = val
                    changed.append(f"{key} → {val}")
                except (TypeError, ValueError):
                    pass

        if not changed:
            return (
                "No changes applied. Provide at least one of: name, humor, warmth, "
                "sass, verbosity, chaos."
            )

        cfg["profile"] = "Custom"
        _save_config(cfg)
        return (
            f"Personality updated: {', '.join(changed)}. "
            "Changes take effect immediately."
        )

    return f"Unknown action '{action}'. Use 'get' or 'set'."

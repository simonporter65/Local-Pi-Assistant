"""
memory/personality.py

Stores, loads, and applies the user's personality configuration.
The personality prompt is injected into every system prompt,
giving the assistant its configured character.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

PERSONALITY_FILE = os.environ.get(
    "PERSONALITY_FILE",
    str(Path(__file__).parent.parent / "memory" / "personality.json")
)

# Fallback personality if none configured yet
DEFAULT_PERSONALITY = {
    "name": None,
    "flavors": {
        "humor": 40,
        "warmth": 60,
        "sass": 30,
        "verbosity": 50,
        "chaos": 20,
    },
    "personality_prompt": (
        "You are a helpful, warm, and capable assistant. "
        "You communicate clearly and are genuinely interested in helping."
    ),
    "profile": "Balanced",
    "configured": False,
}


class PersonalityConfig:
    def __init__(self, config_path: str = PERSONALITY_FILE):
        self.config_path = config_path
        self._config: dict = {}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path) as f:
                    self._config = json.load(f)
                    self._config["configured"] = True
            else:
                self._config = dict(DEFAULT_PERSONALITY)
        except Exception:
            self._config = dict(DEFAULT_PERSONALITY)

    def save(self, config: dict):
        """Called when user confirms their personality setup."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        config["configured"] = True
        config["saved_at"] = datetime.now().isoformat()
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)
        self._config = config
        print(f"[PERSONALITY] Saved: {config.get('name')} / {config.get('profile')}")

    def get(self) -> dict:
        return self._config

    @property
    def name(self) -> Optional[str]:
        return self._config.get("name")

    @property
    def is_configured(self) -> bool:
        return self._config.get("configured", False)

    @property
    def personality_prompt(self) -> str:
        return self._config.get("personality_prompt", DEFAULT_PERSONALITY["personality_prompt"])

    @property
    def flavor(self) -> dict:
        return self._config.get("flavors", DEFAULT_PERSONALITY["flavors"])

    def get_full_system_prompt(
        self,
        model: str,
        category: str,
        user_context: str,
        past_context: str,
    ) -> str:
        name = self.name or "Assistant"
        personality = self.personality_prompt
        flavor = self.flavor

        # Tone modifiers based on flavor values
        tone_notes = []
        if flavor.get("verbosity", 50) < 30:
            tone_notes.append("Be concise. Short answers unless depth is essential.")
        elif flavor.get("verbosity", 50) > 70:
            tone_notes.append("Be thorough. Don't truncate useful context.")

        if flavor.get("chaos", 20) > 65:
            tone_notes.append("Creative approaches are encouraged. Don't always take the obvious path.")

        tone_str = "\n".join(tone_notes) if tone_notes else ""

        return f"""{personality}

WHAT YOU KNOW ABOUT THIS USER:
{user_context}

RELEVANT PAST INTERACTIONS:
{past_context}

CURRENT TASK: {category}
RUNNING ON: {model}

{tone_str}

SKILL FORMAT: SKILL: {{"name": "...", "args": {{...}}}}
FINAL FORMAT: FINAL: <your complete response>

Remember: you are {name}. Never break character. Never say "As an AI."
"""

    def get_background_system_prompt(self, user_context: str) -> str:
        name = self.name or "Assistant"
        return f"""{self.personality_prompt}

You are running a background task. The user is not watching.
Do real work. Use skills. Be thorough.

USER CONTEXT:
{user_context}

SKILL FORMAT: SKILL: {{"name": "...", "args": {{...}}}}
FINAL FORMAT: FINAL: <summary of what you did>
NEW_TASKS: [{{"title":"...","description":"...","task_type":"...","priority_name":"..."}}]
"""

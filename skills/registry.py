"""
skills/registry.py
Loads and manages all agent skills. Supports hot-reload for newly written skills.
"""

import importlib
import importlib.util
import os
import sys
import json
import threading
import traceback


SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))

SKIP = {"__init__.py", "registry.py"}

# Skills that ship with the codebase — not written by the agent at runtime.
# Custom skills (anything NOT in this set) are shown in all categories.
BUILTIN_SKILLS = {
    "bash_exec", "memory_search", "python_repl", "screenshot",
    "system_info", "web_fetch", "web_search", "workspace",
    "browser", "browser_session", "skill_writer", "vision",
    "set_personality",
}


class SkillRegistry:
    def __init__(self, skills_dir: str = SKILLS_DIR):
        self.skills_dir = skills_dir
        self.skills: dict = {}
        self._lock = threading.Lock()
        self._load_all()

    def _load_all(self):
        for fname in sorted(os.listdir(self.skills_dir)):
            if not fname.endswith(".py") or fname in SKIP:
                continue
            name = fname[:-3]
            self._load_skill(name)

    def _load_skill(self, name: str):
        fpath = os.path.join(self.skills_dir, f"{name}.py")
        try:
            spec = importlib.util.spec_from_file_location(f"skills.{name}", fpath)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            if hasattr(mod, "run") and hasattr(mod, "DESCRIPTION"):
                self.skills[name] = mod
            else:
                print(f"[SKILLS] Skipped {name}: missing 'run' or 'DESCRIPTION'")

        except Exception as e:
            print(f"[SKILLS] Error loading {name}: {e}")
            traceback.print_exc()

    def reload(self):
        """Hot-reload all skills — call this after skill_writer creates a new skill."""
        with self._lock:
            self.skills = {}
            self._load_all()

    def run(self, skill_name: str, **kwargs) -> str:
        with self._lock:
            # Try reloading if skill not found (might be newly written)
            if skill_name not in self.skills:
                self._load_skill(skill_name)

            if skill_name not in self.skills:
                available = ", ".join(self.skills.keys())
                raise ValueError(
                    f"Skill '{skill_name}' not found. Available: {available}"
                )

            return self.skills[skill_name].run(**kwargs)

    def list_skills(self) -> str:
        return json.dumps(
            {name: mod.DESCRIPTION for name, mod in self.skills.items()},
            indent=2
        )

    def list_custom_skills(self) -> str:
        """Return skills written by the agent at runtime (not shipped built-ins).

        These are included in the system prompt for ALL categories so the
        assistant is always aware of skills it has learned (e.g. tell_time).
        """
        custom = {
            name: mod.DESCRIPTION
            for name, mod in self.skills.items()
            if name not in BUILTIN_SKILLS
        }
        return json.dumps(custom, indent=2) if custom else ""

    def list_skill_names(self) -> str:
        return ", ".join(self.skills.keys())

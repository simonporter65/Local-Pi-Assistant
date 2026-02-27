"""
proactive/engine.py

The proactive engine watches what it knows about the user and surfaces
useful messages, reminders, insights, and suggestions without being asked.

This is what separates a chatbot from a real personal assistant.
"""

import json
import ollama
import re
import random
from datetime import datetime, time as dtime
from typing import Optional


# How often we can push each type (in minutes)
PUSH_COOLDOWNS = {
    "reminder":   60,
    "insight":   240,
    "suggestion": 120,
    "weather":    360,
    "news":       480,
}

SIDEBAR_SUGGESTIONS_PROMPT = """Based on what you know about this user, generate 3-4 genuinely useful suggestions for things the assistant could help with right now.

User profile:
{user_context}

Recent activity summary:
{recent_summary}

Current time: {time_str}

Generate suggestions that are:
- Specific to this user's life, not generic
- Immediately actionable
- Varied in type (task, information, reminder, creative)

Return JSON array: [{{"category": "Reminder|Research|Task|Insight", "text": "Natural description", "action": "The message to pre-fill when clicked"}}]
Return ONLY valid JSON."""

PROACTIVE_PUSH_PROMPT = """You are a proactive personal assistant that knows this user well.

User profile:
{user_context}

Recent exchange:
User said: {user_message}
You responded about: {response_summary}

Should you proactively add something useful right now? 
Think about: follow-up info, related reminders, useful context they might not know, next steps.

If YES: return {{"push": true, "message": "Your proactive message here"}}
If NO: return {{"push": false}}
Be selective — only push if genuinely valuable. Don't be annoying.
Return ONLY valid JSON."""


class ProactiveEngine:
    def __init__(self, user_model, memory, registry):
        self.user_model = user_model
        self.memory = memory
        self.registry = registry
        self._last_push: dict = {}
        self._sidebar_cache = []
        self._sidebar_cache_time = None

    # ── Sidebar suggestions (shown passively) ────────────────────────────

    def get_sidebar_suggestions(self) -> list:
        """Generate context-aware suggestions for the sidebar."""
        now = datetime.now()

        # Cache for 15 minutes
        if (self._sidebar_cache_time and
                (now - self._sidebar_cache_time).seconds < 900):
            return self._sidebar_cache

        user_context = self.user_model.get_context_for_prompt()
        recent = self._get_recent_summary()
        time_str = now.strftime("%A, %B %d at %I:%M %p")

        # Fast path: if no profile, return generic helpful suggestions
        if "still getting to know you" in user_context:
            return self._generic_suggestions(time_str)

        try:
            resp = ollama.generate(
                model="qwen2.5:0.5b",
                prompt=SIDEBAR_SUGGESTIONS_PROMPT.format(
                    user_context=user_context[:600],
                    recent_summary=recent[:300],
                    time_str=time_str,
                ),
                options={"temperature": 0.7, "num_predict": 400, "num_ctx": 1500}
            )
            text = resp["response"].strip()
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                suggestions = json.loads(match.group())
                self._sidebar_cache = suggestions[:4]
                self._sidebar_cache_time = now
                return self._sidebar_cache
        except Exception:
            pass

        return self._generic_suggestions(time_str)

    def _generic_suggestions(self, time_str: str) -> list:
        hour = datetime.now().hour
        if hour < 10:
            return [
                {"category": "Morning", "text": "Get a summary of today's priorities", "action": "What should I focus on today?"},
                {"category": "Research", "text": "Check the news", "action": "What's happening in the news today?"},
                {"category": "Task", "text": "Set up your day", "action": "Help me plan my day"},
            ]
        elif hour < 17:
            return [
                {"category": "Task", "text": "Something you need to look up?", "action": "I need help researching "},
                {"category": "Code", "text": "Write or debug code", "action": "Help me with some code: "},
                {"category": "Research", "text": "Deep dive on a topic", "action": "Tell me everything about "},
            ]
        else:
            return [
                {"category": "Evening", "text": "Reflect on today", "action": "Help me summarise what I accomplished today"},
                {"category": "Tomorrow", "text": "Plan for tomorrow", "action": "Help me plan tomorrow"},
                {"category": "Creative", "text": "Explore something interesting", "action": "Tell me something fascinating I probably don't know"},
            ]

    # ── Post-message proactive push ───────────────────────────────────────

    def check_after_message(self, user_message: str, assistant_response: str) -> Optional[str]:
        """After a response, decide if we should push something proactive."""
        # Rate limit
        now = datetime.now()
        last = self._last_push.get("general")
        if last and (now - last).seconds < 300:  # 5 min minimum between pushes
            return None

        user_context = self.user_model.get_context_for_prompt()
        if "still getting to know you" in user_context:
            return None  # Don't push without profile

        try:
            resp = ollama.generate(
                model="qwen2.5:0.5b",
                prompt=PROACTIVE_PUSH_PROMPT.format(
                    user_context=user_context[:400],
                    user_message=user_message[:200],
                    response_summary=assistant_response[:200],
                ),
                options={"temperature": 0.5, "num_predict": 200, "num_ctx": 1500}
            )
            text = resp["response"].strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                result = json.loads(match.group())
                if result.get("push") and result.get("message"):
                    self._last_push["general"] = now
                    return result["message"]
        except Exception:
            pass

        return None

    # ── Time-based push messages ──────────────────────────────────────────

    def get_push_message(self) -> Optional[str]:
        """Called every 10 minutes by the UI to check for time-based pushes."""
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        weekday = now.weekday()  # 0=Mon

        # Morning briefing at 8am
        if hour == 8 and minute < 10:
            return self._morning_briefing()

        # End of day at 5:30pm on weekdays
        if weekday < 5 and hour == 17 and 30 <= minute < 40:
            return self._end_of_day_message()

        # Weekly review on Sunday evening
        if weekday == 6 and hour == 19 and minute < 10:
            return "🗓 It's Sunday evening — want me to help you prepare for the week ahead?"

        return None

    def _morning_briefing(self) -> Optional[str]:
        cooldown_key = f"morning_{datetime.now().date()}"
        if self._last_push.get(cooldown_key):
            return None
        self._last_push[cooldown_key] = datetime.now()

        user_context = self.user_model.get_context_for_prompt()
        name_fact = self.memory.db.execute(
            "SELECT fact FROM user_facts WHERE category='name' LIMIT 1"
        ).fetchone()
        name = f", {name_fact[0]}" if name_fact else ""

        hour = datetime.now().hour
        greeting = "Good morning" if hour < 12 else "Good afternoon"

        return (
            f"{greeting}{name}! ☀️ I'm here and ready. "
            f"Would you like a briefing on anything, or shall we dive straight into your day?"
        )

    def _end_of_day_message(self) -> Optional[str]:
        cooldown_key = f"eod_{datetime.now().date()}"
        if self._last_push.get(cooldown_key):
            return None
        self._last_push[cooldown_key] = datetime.now()

        # Count today's interactions
        today = datetime.now().date().isoformat()
        count = self.memory.db.execute(
            "SELECT COUNT(*) FROM interactions WHERE timestamp LIKE ?",
            (f"{today}%",)
        ).fetchone()[0]

        if count == 0:
            return "Quiet day today — I'm here if you need anything this evening. 🌙"
        return (
            f"You've had {count} conversation{'s' if count != 1 else ''} with me today. "
            f"Want to wrap up or work on anything else before you finish?"
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_recent_summary(self) -> str:
        rows = self.memory.db.execute(
            "SELECT user_input FROM interactions ORDER BY id DESC LIMIT 5"
        ).fetchall()
        return "; ".join(r[0][:60] for r in rows) if rows else "No recent activity"

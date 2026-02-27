"""
memory/user_model.py

The user model is the heart of the personal assistant.
It learns facts, preferences, patterns, and personality from every interaction.
Over time it builds a rich profile that makes the assistant genuinely personal.
"""

import json
import ollama
import re
from datetime import datetime, date
from typing import Optional


# Facts the model tracks with emoji icons for the UI
FACT_CATEGORIES = {
    "name":         "👤",
    "location":     "📍",
    "occupation":   "💼",
    "interests":    "🎯",
    "family":       "👨‍👩‍👧",
    "health":       "🏃",
    "schedule":     "📅",
    "preferences":  "⭐",
    "goals":        "🚀",
    "projects":     "🔧",
    "skills":       "🧠",
    "finances":     "💰",
    "mood":         "😊",
    "communication":"💬",
    "technology":   "💻",
}

EXTRACT_PROMPT = """Extract factual information about the user from this message exchange.
Only extract clear, explicit facts — don't infer or guess.

User message: {user_msg}
Assistant response: {assistant_msg}

Return a JSON array of facts. Each fact: {{"category": "...", "fact": "...", "confidence": 0.0-1.0}}
Categories: name, location, occupation, interests, family, health, schedule, preferences, goals, projects, skills, finances, mood, communication, technology

Return [] if no clear facts found.
Return ONLY the JSON array, nothing else."""

PERSONALISE_PROMPT = """You know the following about this user:
{user_context}

The user said: {user_message}
The assistant's raw response: {response}

Rewrite the response to feel more personal and tailored to this specific user.
Keep the content identical but adjust tone, references, and phrasing to feel like it comes from someone who genuinely knows them.
If nothing needs changing, return the response as-is.
Return ONLY the response text."""

SUGGEST_PROMPT = """Based on what you know about this user:
{user_context}

Recent interaction: {recent}

Generate 3 proactive suggestions the assistant could make.
Each should be genuinely useful based on their actual profile — not generic.

Return JSON: [{{"category": "Reminder|Insight|Suggestion|Alert", "text": "...", "action": "..."}}]
Return ONLY valid JSON."""


class UserModel:
    def __init__(self, memory):
        self.memory = memory
        self._ensure_tables()

    def _ensure_tables(self):
        self.memory.db.executescript("""
            CREATE TABLE IF NOT EXISTS user_facts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                category    TEXT NOT NULL,
                fact        TEXT NOT NULL,
                confidence  REAL DEFAULT 1.0,
                source      TEXT,
                created_at  TEXT,
                updated_at  TEXT,
                confirmed   INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS user_preferences (
                key         TEXT PRIMARY KEY,
                value       TEXT,
                updated_at  TEXT
            );

            CREATE TABLE IF NOT EXISTS user_patterns (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern     TEXT,
                count       INTEGER DEFAULT 1,
                last_seen   TEXT
            );
        """)
        self.memory.db.commit()

    # ── Fact extraction ───────────────────────────────────────────────────

    def extract_from_message(self, user_message: str):
        """Quick extraction from just the user's message (called early in pipeline)."""
        # Fast heuristic extraction — no LLM needed for obvious facts
        self._heuristic_extract(user_message)

    def extract_from_exchange(self, user_message: str, assistant_response: str):
        """Deeper LLM-based extraction from the full exchange."""
        try:
            resp = ollama.generate(
                model="qwen2.5:0.5b",
                prompt=EXTRACT_PROMPT.format(
                    user_msg=user_message[:500],
                    assistant_msg=assistant_response[:300],
                ),
                options={"temperature": 0.1, "num_predict": 400, "num_ctx": 1024}
            )
            text = resp["response"].strip()
            # Find JSON array
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                return
            facts = json.loads(match.group())
            for f in facts:
                if isinstance(f, dict) and f.get("fact") and f.get("category"):
                    self._store_fact(
                        category=f["category"],
                        fact=f["fact"],
                        confidence=float(f.get("confidence", 0.7)),
                        source="llm_extract"
                    )
        except Exception:
            pass

    def _heuristic_extract(self, text: str):
        """Fast pattern-based extraction for common facts."""
        t = text.lower()

        # Name patterns
        for pattern in [r"i'?m ([A-Z][a-z]+)", r"my name is ([A-Z][a-z]+)", r"call me ([A-Z][a-z]+)"]:
            m = re.search(pattern, text)
            if m:
                self._store_fact("name", m.group(1), confidence=0.9, source="heuristic")

        # Location
        for pattern in [r"i (?:live|am) in ([A-Z][a-zA-Z\s]+)", r"based in ([A-Z][a-zA-Z\s,]+)"]:
            m = re.search(pattern, text)
            if m:
                self._store_fact("location", m.group(1).strip(), confidence=0.8, source="heuristic")

        # Occupation
        for pattern in [r"i(?:'m| am) (?:a |an )?([a-z]+ (?:developer|engineer|designer|teacher|doctor|lawyer|student|manager|founder|ceo|cto))", r"i work (?:as a?n? )?([a-z ]+)"]:
            m = re.search(pattern, t)
            if m:
                self._store_fact("occupation", m.group(1).strip(), confidence=0.8, source="heuristic")

        # Interests from common signals
        interest_signals = {
            "coding": ["python", "javascript", "programming", "coding", "software"],
            "music": ["music", "guitar", "piano", "spotify", "playlist"],
            "fitness": ["gym", "running", "workout", "exercise", "yoga"],
            "cooking": ["recipe", "cooking", "food", "chef", "kitchen"],
            "reading": ["book", "reading", "novel", "author", "library"],
            "gaming": ["game", "gaming", "steam", "playstation", "xbox"],
        }
        for interest, keywords in interest_signals.items():
            if any(kw in t for kw in keywords):
                self._store_fact("interests", interest, confidence=0.6, source="heuristic")

        # Time patterns / schedule
        if any(w in t for w in ["morning", "evening", "night", "weekend", "monday", "every day"]):
            pass  # Store as pattern rather than fact

    def _store_fact(self, category: str, fact: str, confidence: float = 0.8, source: str = ""):
        """Store a fact, avoiding near-duplicates."""
        # Check for existing similar fact
        existing = self.memory.db.execute(
            "SELECT id, fact FROM user_facts WHERE category=? ORDER BY updated_at DESC LIMIT 5",
            (category,)
        ).fetchall()

        for row in existing:
            # Simple dedup: if fact is very similar, update confidence
            if _similar(row[1], fact):
                self.memory.db.execute(
                    "UPDATE user_facts SET confidence=MAX(confidence, ?), updated_at=? WHERE id=?",
                    (confidence, datetime.now().isoformat(), row[0])
                )
                self.memory.db.commit()
                return

        self.memory.db.execute(
            """INSERT INTO user_facts (category, fact, confidence, source, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (category, fact, confidence, source,
             datetime.now().isoformat(), datetime.now().isoformat())
        )
        self.memory.db.commit()

    # ── Context building ──────────────────────────────────────────────────

    def get_context_for_prompt(self) -> str:
        """Build a rich context string for injection into the system prompt."""
        facts = self.memory.db.execute(
            "SELECT category, fact, confidence FROM user_facts WHERE confidence > 0.5 ORDER BY confidence DESC, updated_at DESC"
        ).fetchall()

        if not facts:
            return "I'm still getting to know you. Tell me about yourself!"

        by_category = {}
        for row in facts:
            cat, fact, conf = row
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(fact)

        lines = []
        priority = ["name", "location", "occupation", "goals", "projects", "preferences", "interests", "family", "health", "schedule"]
        for cat in priority:
            if cat in by_category:
                facts_str = ", ".join(by_category[cat][:3])
                lines.append(f"- {cat.capitalize()}: {facts_str}")

        for cat, facts_list in by_category.items():
            if cat not in priority:
                lines.append(f"- {cat.capitalize()}: {', '.join(facts_list[:2])}")

        return "\n".join(lines) if lines else "No profile yet."

    def get_display_profile(self) -> dict:
        """For the UI sidebar."""
        facts = self.memory.db.execute(
            "SELECT category, fact FROM user_facts WHERE confidence > 0.5 ORDER BY confidence DESC, updated_at DESC LIMIT 20"
        ).fetchall()

        display_facts = []
        seen_cats = {}
        for cat, fact in facts:
            if cat not in seen_cats:
                seen_cats[cat] = 0
            if seen_cats[cat] < 2:  # Max 2 facts per category in display
                display_facts.append({
                    "icon": FACT_CATEGORIES.get(cat, "•"),
                    "text": fact,
                    "category": cat,
                })
                seen_cats[cat] += 1

        # Get preferred assistant name
        name_row = self.memory.db.execute(
            "SELECT value FROM user_preferences WHERE key='assistant_name'"
        ).fetchone()

        return {
            "facts": display_facts,
            "assistant_name": name_row[0] if name_row else None,
        }

    def set_preference(self, key: str, value: str):
        self.memory.db.execute(
            "INSERT OR REPLACE INTO user_preferences (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, datetime.now().isoformat())
        )
        self.memory.db.commit()

    def get_preference(self, key: str, default=None) -> Optional[str]:
        row = self.memory.db.execute(
            "SELECT value FROM user_preferences WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else default

    # ── Response personalisation ──────────────────────────────────────────

    def personalise_response(self, user_message: str, response: str) -> str:
        """Light personalisation pass — inject context naturally."""
        user_context = self.get_context_for_prompt()
        if user_context == "No profile yet." or len(user_context) < 30:
            return response  # No profile yet, skip

        # Use user's name if known
        name_facts = self.memory.db.execute(
            "SELECT fact FROM user_facts WHERE category='name' AND confidence > 0.7 LIMIT 1"
        ).fetchone()

        # Only do LLM personalisation for longer responses where it's worth it
        if len(response) < 100:
            return response

        try:
            resp = ollama.generate(
                model="qwen2.5:0.5b",
                prompt=PERSONALISE_PROMPT.format(
                    user_context=user_context[:600],
                    user_message=user_message[:200],
                    response=response[:800],
                ),
                options={"temperature": 0.4, "num_predict": 600, "num_ctx": 2048}
            )
            result = resp["response"].strip()
            # Sanity: if output is much shorter, use original
            if len(result) > len(response) * 0.5:
                return result
        except Exception:
            pass

        return response


# ── Helpers ───────────────────────────────────────────────────────────────────

def _similar(a: str, b: str) -> bool:
    """Rough similarity check to avoid storing duplicates."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return True
    # One contains the other
    if a in b or b in a:
        return True
    # Word overlap > 70%
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
    return overlap > 0.7

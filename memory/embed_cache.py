"""
memory/embed_cache.py

Caches embeddings so we don't call nomic-embed-text on every message.
Also skips semantic search entirely for short messages.

Before: nomic-embed-text called on every incoming message = 0.3-0.8s per message
After:  cache hit = ~0.001s, cache miss = 0.3-0.8s (only on new unique messages)

The cache is in-memory (dict), lives for the server process lifetime.
50-entry LRU keeps RAM negligible (~50 * 768 floats * 4 bytes = ~150KB).
"""

import hashlib
import ollama
from collections import OrderedDict
from typing import Optional

EMBED_MODEL = "nomic-embed-text"
CACHE_SIZE  = 50
SHORT_MSG_WORD_THRESHOLD = 6  # Skip embed for messages this short


class EmbedCache:
    def __init__(self, max_size: int = CACHE_SIZE):
        self._cache: OrderedDict[str, list] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def _key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[list]:
        k = self._key(text)
        if k in self._cache:
            self._cache.move_to_end(k)  # LRU update
            self._hits += 1
            return self._cache[k]
        self._misses += 1
        return None

    def set(self, text: str, embedding: list):
        k = self._key(text)
        if k in self._cache:
            self._cache.move_to_end(k)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # Evict oldest
            self._cache[k] = embedding

    def embed(self, text: str) -> Optional[list]:
        """Get embedding, using cache. Returns None on failure."""
        cached = self.get(text)
        if cached is not None:
            return cached
        try:
            resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
            emb = resp.get("embedding", [])
            if emb:
                self.set(text, emb)
            return emb or None
        except Exception:
            return None

    def should_skip(self, text: str) -> bool:
        """Short messages don't benefit from semantic search — use recency instead."""
        return len(text.split()) <= SHORT_MSG_WORD_THRESHOLD

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{100*self._hits/total:.0f}%" if total else "—",
            "cached_entries": len(self._cache),
        }


# ── Module-level singleton (shared across all imports) ────────────────────────
embed_cache = EmbedCache()

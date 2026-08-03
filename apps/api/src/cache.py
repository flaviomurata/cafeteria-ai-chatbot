import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from src.models import SourceCitation


@dataclass(frozen=True)
class CachedChatResponse:
    response: str
    sources: list[SourceCitation]


class ResponseCache:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self._cache: dict[str, dict] = {}
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str) -> str:
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    # 'What is Python?' and 'what is python?'

    def get(self, query: str) -> str | CachedChatResponse | None:
        key = self._make_key(query)

        if key in self._cache:
            entry = self._cache[key]
            # Check TTL
            if time.time() - entry["timestamp"] < self.ttl:
                self._hits += 1
                return entry["response"]
            else:
                del self._cache[key]

        self._misses += 1
        return None

    def set(self, query: str, response: str | CachedChatResponse) -> None:
        key = self._make_key(query)
        self._cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "query": query,
        }

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1%}",
            "cached_entries": len(self._cache),
        }


class QueryEmbeddingCache:
    """Small persistent cache for expensive query-embedding calls."""

    def __init__(self, path: Path, *, max_entries: int = 128):
        self._path = path
        self._max_entries = max_entries
        self._lock = threading.Lock()

    def get(self, model: str, query: str) -> list[float] | None:
        if self._max_entries == 0:
            return None
        with self._lock:
            try:
                with self._connect() as database:
                    row = database.execute(
                        """
                        SELECT embedding
                        FROM query_embeddings
                        WHERE model = ? AND query = ?
                        """,
                        (model, query),
                    ).fetchone()
                    if row is None:
                        return None
                    database.execute(
                        """
                        UPDATE query_embeddings
                        SET last_used_at = ?
                        WHERE model = ? AND query = ?
                        """,
                        (time.time(), model, query),
                    )
                    return json.loads(row[0])
            except (OSError, sqlite3.Error, TypeError, ValueError):
                return None

    def set(self, model: str, query: str, embedding: list[float]) -> None:
        if self._max_entries == 0:
            return
        with self._lock:
            try:
                with self._connect() as database:
                    database.execute(
                        """
                        INSERT INTO query_embeddings (
                            model, query, embedding, last_used_at
                        )
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(model, query) DO UPDATE SET
                            embedding = excluded.embedding,
                            last_used_at = excluded.last_used_at
                        """,
                        (model, query, json.dumps(embedding), time.time()),
                    )
                    database.execute(
                        """
                        DELETE FROM query_embeddings
                        WHERE rowid IN (
                            SELECT rowid
                            FROM query_embeddings
                            ORDER BY last_used_at DESC
                            LIMIT -1 OFFSET ?
                        )
                        """,
                        (self._max_entries,),
                    )
            except (OSError, sqlite3.Error, TypeError, ValueError):
                # A cache must never take down retrieval.
                return

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        database = sqlite3.connect(self._path)
        database.execute(
            """
            CREATE TABLE IF NOT EXISTS query_embeddings (
                model TEXT NOT NULL,
                query TEXT NOT NULL,
                embedding TEXT NOT NULL,
                last_used_at REAL NOT NULL,
                PRIMARY KEY (model, query)
            )
            """
        )
        return database


# uv run python -c "
# import time
# from app.cache import ResponseCache

# cache = ResponseCache(ttl_seconds=3)  # Short TTL for demo

# print('=== CACHE DEMO ===')
# print()

# # Miss
# result = cache.get('What is Python?')
# print(f'1. First lookup: {result}  (miss - nothing cached yet)')

# # Store
# cache.set('What is Python?', 'Python is a programming language.')
# print(f'2. Stored response in cache')

# # Hit
# result = cache.get('What is Python?')
# print(f'3. Second lookup: {result}  (HIT!)')

# # Case insensitive
# result = cache.get('what is python?')
# print(f'4. Lowercase lookup: {result}  (HIT - case insensitive!)')

# # Different query = miss
# result = cache.get('What is JavaScript?')
# print(f'5. Different query: {result}  (miss)')

# # Stats
# print(f'6. Stats: {cache.stats}')

# # Wait for TTL
# print(f'7. Waiting 4 seconds for TTL expiration...')
# time.sleep(4)

# result = cache.get('What is Python?')
# print(f'8. After TTL: {result}  (miss - expired!)')
# print(f'9. Final stats: {cache.stats}')
# "

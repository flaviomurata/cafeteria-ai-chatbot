import hashlib
import time
from dataclasses import dataclass

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

"""Unit tests for the cache layer: src/cache.py.

TTL behaviour is driven by a fake clock rather than `time.sleep`, so the suite
stays fast and deterministic. `src/cache.py` does `import time` and calls
`time.time()`, so swapping the module-level `time` name is enough.
"""

import pytest

from src.cache import ResponseCache

QUERY = "What is for lunch?"
RESPONSE = "Grilled salmon with roasted vegetables."


class FakeClock:
    """Monotonic clock the test advances by hand."""

    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr("src.cache.time", fake)
    return fake


@pytest.fixture
def cache() -> ResponseCache:
    return ResponseCache(ttl_seconds=300)


# --------------------------------------------------------------------------- #
# Basic get / set                                                             #
# --------------------------------------------------------------------------- #


def test_get_returns_none_on_empty_cache(cache: ResponseCache):
    assert cache.get(QUERY) is None


def test_set_then_get_returns_the_response(cache: ResponseCache):
    cache.set(QUERY, RESPONSE)

    assert cache.get(QUERY) == RESPONSE


def test_distinct_queries_do_not_collide(cache: ResponseCache):
    cache.set("What is for lunch?", "Salmon")
    cache.set("What is for dinner?", "Risotto")

    assert cache.get("What is for lunch?") == "Salmon"
    assert cache.get("What is for dinner?") == "Risotto"


def test_get_returns_none_for_unknown_query(cache: ResponseCache):
    cache.set(QUERY, RESPONSE)

    assert cache.get("Do you have vegan options?") is None


def test_set_overwrites_an_existing_entry(cache: ResponseCache):
    cache.set(QUERY, "Old answer")
    cache.set(QUERY, "New answer")

    assert cache.get(QUERY) == "New answer"
    assert cache.stats["cached_entries"] == 1


def test_empty_string_is_a_valid_key(cache: ResponseCache):
    cache.set("", RESPONSE)

    assert cache.get("") == RESPONSE


def test_empty_response_is_returned_not_treated_as_a_miss(cache: ResponseCache):
    cache.set(QUERY, "")

    assert cache.get(QUERY) == ""
    assert cache.stats["hits"] == 1


# --------------------------------------------------------------------------- #
# Key normalization                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "variant",
    [
        "What is for lunch?",
        "what is for lunch?",
        "WHAT IS FOR LUNCH?",
        "  What is for lunch?  ",
        "\tWhat is for lunch?\n",
    ],
)
def test_lookup_ignores_case_and_surrounding_whitespace(
    cache: ResponseCache, variant: str
):
    cache.set(QUERY, RESPONSE)

    assert cache.get(variant) == RESPONSE


def test_set_normalizes_the_key_too(cache: ResponseCache):
    cache.set("  WHAT IS FOR LUNCH?  ", RESPONSE)

    assert cache.get(QUERY) == RESPONSE


def test_normalization_does_not_touch_interior_whitespace(cache: ResponseCache):
    cache.set("what is for lunch?", RESPONSE)

    assert cache.get("what  is  for  lunch?") is None


def test_normalized_variants_share_one_entry(cache: ResponseCache):
    cache.set("What is for lunch?", "First")
    cache.set("WHAT IS FOR LUNCH?", "Second")

    assert cache.stats["cached_entries"] == 1
    assert cache.get("what is for lunch?") == "Second"


# --------------------------------------------------------------------------- #
# TTL expiry                                                                  #
# --------------------------------------------------------------------------- #


def test_entry_survives_until_the_ttl_elapses(clock: FakeClock):
    cache = ResponseCache(ttl_seconds=300)
    cache.set(QUERY, RESPONSE)

    clock.advance(299)

    assert cache.get(QUERY) == RESPONSE


def test_entry_expires_exactly_at_the_ttl(clock: FakeClock):
    """The check is `elapsed < ttl`, so `elapsed == ttl` is already stale."""
    cache = ResponseCache(ttl_seconds=300)
    cache.set(QUERY, RESPONSE)

    clock.advance(300)

    assert cache.get(QUERY) is None


def test_expired_entry_is_evicted_not_just_hidden(clock: FakeClock):
    cache = ResponseCache(ttl_seconds=300)
    cache.set(QUERY, RESPONSE)
    clock.advance(301)

    assert cache.get(QUERY) is None
    assert cache.stats["cached_entries"] == 0


def test_expired_entry_counts_as_a_miss(clock: FakeClock):
    cache = ResponseCache(ttl_seconds=300)
    cache.set(QUERY, RESPONSE)
    clock.advance(301)

    cache.get(QUERY)

    assert cache.stats == {
        "hits": 0,
        "misses": 1,
        "hit_rate": "0.0%",
        "cached_entries": 0,
    }


def test_re_setting_after_expiry_refreshes_the_entry(clock: FakeClock):
    cache = ResponseCache(ttl_seconds=300)
    cache.set(QUERY, "Stale")
    clock.advance(301)
    assert cache.get(QUERY) is None

    cache.set(QUERY, "Fresh")
    clock.advance(299)

    assert cache.get(QUERY) == "Fresh"


def test_zero_ttl_expires_immediately(clock: FakeClock):
    cache = ResponseCache(ttl_seconds=0)
    cache.set(QUERY, RESPONSE)

    assert cache.get(QUERY) is None


def test_only_the_expired_entry_is_dropped(clock: FakeClock):
    cache = ResponseCache(ttl_seconds=300)
    cache.set("old", "Old answer")
    clock.advance(200)
    cache.set("new", "New answer")

    clock.advance(150)  # 'old' is 350s, 'new' is 150s

    assert cache.get("old") is None
    assert cache.get("new") == "New answer"


def test_default_ttl_is_five_minutes():
    assert ResponseCache().ttl == 300


# --------------------------------------------------------------------------- #
# Stats                                                                       #
# --------------------------------------------------------------------------- #


def test_stats_start_at_zero(cache: ResponseCache):
    assert cache.stats == {
        "hits": 0,
        "misses": 0,
        "hit_rate": "0.0%",
        "cached_entries": 0,
    }


def test_set_alone_does_not_move_hit_or_miss_counters(cache: ResponseCache):
    cache.set(QUERY, RESPONSE)

    assert cache.stats == {
        "hits": 0,
        "misses": 0,
        "hit_rate": "0.0%",
        "cached_entries": 1,
    }


def test_stats_count_hits_and_misses(cache: ResponseCache):
    cache.get(QUERY)  # miss
    cache.set(QUERY, RESPONSE)
    cache.get(QUERY)  # hit
    cache.get(QUERY)  # hit

    assert cache.stats == {
        "hits": 2,
        "misses": 1,
        "hit_rate": "66.7%",
        "cached_entries": 1,
    }


@pytest.mark.parametrize(
    ("hits", "misses", "expected_rate"),
    [
        (1, 0, "100.0%"),
        (0, 1, "0.0%"),
        (1, 1, "50.0%"),
        (1, 2, "33.3%"),
        (3, 1, "75.0%"),
    ],
)
def test_hit_rate_formatting(
    cache: ResponseCache, hits: int, misses: int, expected_rate: str
):
    for i in range(misses):
        cache.get(f"missing-{i}")

    cache.set(QUERY, RESPONSE)
    for _ in range(hits):
        cache.get(QUERY)

    assert cache.stats["hit_rate"] == expected_rate


def test_cached_entries_tracks_distinct_keys(cache: ResponseCache):
    cache.set("a", "1")
    cache.set("b", "2")
    cache.set("c", "3")

    assert cache.stats["cached_entries"] == 3


def test_caches_are_independent():
    first = ResponseCache()
    second = ResponseCache()

    first.set(QUERY, RESPONSE)

    assert second.get(QUERY) is None
    assert second.stats["cached_entries"] == 0


# --------------------------------------------------------------------------- #
# Key derivation                                                              #
# --------------------------------------------------------------------------- #


def test_make_key_is_a_sha256_hex_digest(cache: ResponseCache):
    key = cache._make_key(QUERY)

    assert len(key) == 64
    assert set(key) <= set("0123456789abcdef")


def test_make_key_is_stable_and_normalized(cache: ResponseCache):
    assert cache._make_key(QUERY) == cache._make_key("  WHAT IS FOR LUNCH?  ")
    assert cache._make_key(QUERY) != cache._make_key("What is for dinner?")


def test_stored_entry_keeps_the_original_query_for_debugging(
    cache: ResponseCache, clock: FakeClock
):
    cache.set("  WHAT IS FOR LUNCH?  ", RESPONSE)

    entry = cache._cache[cache._make_key(QUERY)]

    assert entry["query"] == "  WHAT IS FOR LUNCH?  "
    assert entry["response"] == RESPONSE
    assert entry["timestamp"] == clock.now


def test_unicode_queries_are_supported(cache: ResponseCache):
    cache.set("Qual é o almoço de hoje?", "Salmão grelhado")

    assert cache.get("QUAL É O ALMOÇO DE HOJE?") == "Salmão grelhado"

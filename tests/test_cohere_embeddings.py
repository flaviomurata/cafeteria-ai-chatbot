from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.partner_knowledge.cohere_embeddings import (
    CohereEmbeddingClient,
    CohereEmbeddingConfigurationError,
)
from src.partner_knowledge.embedding_budget import (
    EmbeddingBudgetExceededError,
    EmbeddingUsageLedger,
)
from src.provider_errors import ProviderRateLimitError


def test_embedding_usage_ledger_reserves_calls_per_utc_month(tmp_path: Path):
    ledger = EmbeddingUsageLedger(tmp_path / "usage.sqlite3", monthly_limit=3)
    august = datetime(2026, 8, 1, tzinfo=timezone.utc)
    september = datetime(2026, 9, 1, tzinfo=timezone.utc)

    usage = ledger.reserve(2, now=august)

    assert usage.month == "2026-08"
    assert usage.used_calls == 2
    assert usage.remaining_calls == 1
    assert ledger.snapshot(now=september).used_calls == 0

    with pytest.raises(EmbeddingBudgetExceededError):
        ledger.reserve(2, now=august)

    assert ledger.snapshot(now=august).used_calls == 2


def test_cohere_embedding_client_requires_an_api_key(tmp_path: Path):
    with pytest.raises(CohereEmbeddingConfigurationError, match="COHERE_API_KEY"):
        CohereEmbeddingClient(
            "",
            ledger=EmbeddingUsageLedger(tmp_path / "usage.sqlite3"),
            sdk_client=object(),
        )


def test_cohere_embedding_client_batches_documents_for_search(tmp_path: Path):
    class FakeCohereClient:
        def __init__(self):
            self.calls: list[dict] = []

        def embed(self, **kwargs):
            self.calls.append(kwargs)
            vectors = [
                [float(index), 0.0, 1.0] for index, _ in enumerate(kwargs["texts"])
            ]
            return SimpleNamespace(embeddings=SimpleNamespace(float=vectors))

    sdk_client = FakeCohereClient()
    client = CohereEmbeddingClient(
        "test-key",
        model="embed-v4.0",
        embedding_dimension=3,
        ledger=EmbeddingUsageLedger(tmp_path / "usage.sqlite3", monthly_limit=2),
        sdk_client=sdk_client,
    )

    vectors = client.embed_documents([f"document-{index}" for index in range(97)])

    assert len(vectors) == 97
    assert [len(call["texts"]) for call in sdk_client.calls] == [96, 1]
    assert all(call["model"] == "embed-v4.0" for call in sdk_client.calls)
    assert all(call["input_type"] == "search_document" for call in sdk_client.calls)
    assert all(call["embedding_types"] == ["float"] for call in sdk_client.calls)
    assert all(call["output_dimension"] == 3 for call in sdk_client.calls)


def test_cohere_embedding_client_uses_search_query_for_questions(tmp_path: Path):
    class FakeCohereClient:
        def __init__(self):
            self.calls: list[dict] = []

        def embed(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(embeddings=SimpleNamespace(float=[[1.0, 0.0, 1.0]]))

    sdk_client = FakeCohereClient()
    client = CohereEmbeddingClient(
        "test-key",
        embedding_dimension=3,
        ledger=EmbeddingUsageLedger(tmp_path / "usage.sqlite3", monthly_limit=1),
        sdk_client=sdk_client,
    )

    assert client.embed_query("What is available?") == [1.0, 0.0, 1.0]
    assert sdk_client.calls == [
        {
            "texts": ["What is available?"],
            "model": "embed-v4.0",
            "input_type": "search_query",
            "output_dimension": 3,
            "embedding_types": ["float"],
        }
    ]


def test_cohere_embedding_client_retries_once_when_server_gives_a_delay(
    tmp_path: Path,
):
    class RateLimitError(RuntimeError):
        status_code = 429
        headers = {"retry-after": "2"}

    class FakeCohereClient:
        def __init__(self):
            self.calls = 0

        def embed(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RateLimitError("try again")
            return SimpleNamespace(embeddings=SimpleNamespace(float=[[1.0, 0.0, 1.0]]))

    sleeps: list[float] = []
    sdk_client = FakeCohereClient()
    ledger = EmbeddingUsageLedger(tmp_path / "usage.sqlite3", monthly_limit=2)
    client = CohereEmbeddingClient(
        "test-key",
        embedding_dimension=3,
        ledger=ledger,
        sdk_client=sdk_client,
        sleep=sleeps.append,
    )

    assert client.embed_query("What is available?") == [1.0, 0.0, 1.0]
    assert sdk_client.calls == 2
    assert sleeps == [2]
    assert ledger.snapshot().used_calls == 2


def test_cohere_embedding_client_preflights_all_document_batches(
    tmp_path: Path,
):
    class FakeCohereClient:
        def __init__(self):
            self.calls = 0

        def embed(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(
                embeddings=SimpleNamespace(float=[[1.0, 0.0, 1.0] for _ in range(96)])
            )

    sdk_client = FakeCohereClient()
    ledger = EmbeddingUsageLedger(tmp_path / "usage.sqlite3", monthly_limit=1)
    client = CohereEmbeddingClient(
        "test-key",
        embedding_dimension=3,
        ledger=ledger,
        sdk_client=sdk_client,
    )

    with pytest.raises(EmbeddingBudgetExceededError):
        client.embed_documents([f"document-{index}" for index in range(97)])

    assert sdk_client.calls == 0
    assert ledger.snapshot().used_calls == 0


def test_cohere_embedding_client_preserves_retry_after_when_retry_is_exhausted(
    tmp_path: Path,
):
    class RateLimitError(RuntimeError):
        status_code = 429
        headers = {"retry-after": "2"}

    class FakeCohereClient:
        def embed(self, **_kwargs):
            raise RateLimitError("try again")

    client = CohereEmbeddingClient(
        "test-key",
        embedding_dimension=3,
        ledger=EmbeddingUsageLedger(tmp_path / "usage.sqlite3", monthly_limit=2),
        sdk_client=FakeCohereClient(),
        sleep=lambda _seconds: None,
    )

    with pytest.raises(ProviderRateLimitError) as error:
        client.embed_query("What is available?")

    assert error.value.retry_after_seconds == 2

"""Cohere embedding adapter with explicit batching and quota accounting."""

import logging
import math
import time
from collections.abc import Callable, Sequence
from numbers import Real
from typing import Any

from src.partner_knowledge.embedding_budget import EmbeddingUsage, EmbeddingUsageLedger
from src.provider_errors import (
    is_provider_rate_limit_error,
    provider_rate_limit_error,
    provider_retry_after_seconds,
)

COHERE_PROVIDER = "cohere"
DEFAULT_EMBEDDING_MODEL = "embed-v4.0"
DEFAULT_EMBEDDING_DIMENSION = 1024
MAX_EMBEDDING_BATCH_SIZE = 96
MAX_RETRY_DELAY_SECONDS = 60
logger = logging.getLogger(__name__)


class CohereEmbeddingError(RuntimeError):
    """Raised when Cohere cannot produce a valid embedding response."""


class CohereEmbeddingConfigurationError(CohereEmbeddingError):
    """Raised when the Cohere embedding configuration is unusable."""


class CohereEmbeddingClient:
    """Small direct-SDK boundary for document and query embeddings."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        ledger: EmbeddingUsageLedger,
        sdk_client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not api_key.strip():
            raise CohereEmbeddingConfigurationError(
                "COHERE_API_KEY is required for Cohere Partner knowledge embeddings."
            )
        if model != DEFAULT_EMBEDDING_MODEL:
            raise CohereEmbeddingConfigurationError(
                "Partner knowledge embeddings must use Cohere embed-v4.0."
            )
        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be positive")
        self._model = model
        self._embedding_dimension = embedding_dimension
        self._ledger = ledger
        self._sleep = sleep
        self._client = (
            sdk_client if sdk_client is not None else _build_sdk_client(api_key)
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    @property
    def collection_metadata(self) -> dict[str, str]:
        return {
            "embedding_provider": COHERE_PROVIDER,
            "embedding_model": self._model,
            "embedding_dimension": str(self._embedding_dimension),
            "embedding_document_input_type": "search_document",
            "embedding_query_input_type": "search_query",
        }

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        if not documents:
            return []
        batches = list(_batches(documents, MAX_EMBEDDING_BATCH_SIZE))
        usage = self._ledger.reserve(len(batches))
        _log_usage("documents", usage, len(batches))
        embeddings: list[list[float]] = []
        for batch in batches:
            embeddings.extend(
                self._embed_texts(
                    batch,
                    input_type="search_document",
                    reservation_already_made=True,
                )
            )
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        return self._embed_texts(
            [query], input_type="search_query", reservation_already_made=False
        )[0]

    @staticmethod
    def projected_document_calls(document_count: int) -> int:
        if document_count < 1:
            return 0
        return math.ceil(document_count / MAX_EMBEDDING_BATCH_SIZE)

    def usage_snapshot(self):
        """Return the current usage without exposing the API key."""
        return self._ledger.snapshot()

    def _embed_texts(
        self,
        texts: list[str],
        *,
        input_type: str,
        reservation_already_made: bool,
    ) -> list[list[float]]:
        attempt = 0
        reserve_before_attempt = not reservation_already_made
        while True:
            if reserve_before_attempt:
                usage = self._ledger.reserve(1)
                _log_usage(input_type, usage, 1)
            try:
                response = self._client.embed(
                    texts=texts,
                    model=self._model,
                    input_type=input_type,
                    output_dimension=self._embedding_dimension,
                    embedding_types=["float"],
                )
            except Exception as exc:
                status_code = _status_code(exc)
                retry_after = provider_retry_after_seconds(exc)
                if _is_authentication_failure(status_code):
                    raise CohereEmbeddingConfigurationError(
                        "Cohere rejected the embedding API key."
                    ) from exc
                if status_code == 429 or is_provider_rate_limit_error(exc):
                    if _can_retry(attempt, retry_after):
                        self._sleep(retry_after)
                        attempt += 1
                        reserve_before_attempt = True
                        continue
                    raise provider_rate_limit_error("Cohere embeddings", exc) from exc
                if _is_transient_status(status_code) and _can_retry(
                    attempt, retry_after
                ):
                    self._sleep(retry_after)
                    attempt += 1
                    reserve_before_attempt = True
                    continue
                raise CohereEmbeddingError(
                    "Cohere failed to create Partner knowledge embeddings."
                ) from exc
            return _validate_response(
                response,
                expected_count=len(texts),
                expected_dimension=self._embedding_dimension,
            )


def _build_sdk_client(api_key: str) -> Any:
    try:
        import cohere
    except ImportError as exc:
        raise CohereEmbeddingConfigurationError(
            "The cohere package is required for Partner knowledge embeddings."
        ) from exc
    return cohere.ClientV2(api_key=api_key, max_retries=0)


def _batches(items: Sequence[str], batch_size: int) -> list[list[str]]:
    return [
        list(items[start : start + batch_size])
        for start in range(0, len(items), batch_size)
    ]


def _validate_response(
    response: Any, *, expected_count: int, expected_dimension: int
) -> list[list[float]]:
    embeddings = getattr(getattr(response, "embeddings", None), "float", None)
    if embeddings is None:
        raise CohereEmbeddingError("Cohere returned no float embeddings.")
    try:
        embedding_count = len(embeddings)
    except TypeError as exc:
        raise CohereEmbeddingError("Cohere returned invalid float embeddings.") from exc
    if embedding_count != expected_count:
        raise CohereEmbeddingError(
            "Cohere returned a different number of embeddings than inputs."
        )
    validated: list[list[float]] = []
    for embedding in embeddings:
        try:
            has_expected_shape = len(embedding) == expected_dimension
            has_numeric_values = all(
                isinstance(value, Real) and not isinstance(value, bool)
                for value in embedding
            )
        except TypeError as exc:
            raise CohereEmbeddingError(
                "Cohere returned an embedding with an unexpected dimension."
            ) from exc
        if not has_expected_shape or not has_numeric_values:
            raise CohereEmbeddingError(
                "Cohere returned an embedding with an unexpected dimension."
            )
        validated.append([float(value) for value in embedding])
    return validated


def _status_code(exc: BaseException) -> int | None:
    for candidate in (
        getattr(exc, "status_code", None),
        getattr(exc, "status", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        try:
            if candidate is not None:
                return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _is_authentication_failure(status_code: int | None) -> bool:
    return status_code in {401, 403}


def _is_transient_status(status_code: int | None) -> bool:
    return status_code in {408, 409} or (
        status_code is not None and 500 <= status_code <= 599
    )


def _can_retry(attempt: int, retry_after: int | None) -> bool:
    return (
        attempt == 0
        and retry_after is not None
        and 0 < retry_after <= MAX_RETRY_DELAY_SECONDS
    )


def _log_usage(operation: str, usage: EmbeddingUsage, projected_calls: int) -> None:
    logger.info(
        "Cohere embedding budget reserved",
        extra={
            "extra_data": {
                "operation": operation,
                "month": usage.month,
                "used_calls": usage.used_calls,
                "remaining_calls": usage.remaining_calls,
                "projected_calls": projected_calls,
            }
        },
    )

import fcntl
import hashlib
import os
import re
import sqlite3
import unicodedata
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

import chromadb
from chromadb.errors import ChromaError

from src.cache import QueryEmbeddingCache
from src.partner_knowledge.constants import DEFAULT_RELEVANCE_THRESHOLD
from src.partner_knowledge.index_storage import (
    ACTIVATION_LOCK_NAME,
    active_collection_name,
    partner_knowledge_index_lock,
)
from src.provider_errors import (
    is_provider_rate_limit_error,
    provider_rate_limit_error,
)

_QUERY_EMBEDDING_LOCK_SLOT_COUNT = 128
_SEMANTIC_CANDIDATE_MULTIPLIER = 4
_SEMANTIC_RELEVANCE_WEIGHT = 0.5
_LEXICAL_RELEVANCE_WEIGHT = 0.5
_MINIMUM_SEMANTIC_SCORE_FOR_LEXICAL_RESCUE = 0.15
_LEXICAL_STOP_WORDS = frozenset(
    {
        "a",
        "as",
        "ao",
        "aos",
        "com",
        "da",
        "das",
        "de",
        "do",
        "dos",
        "e",
        "em",
        "for",
        "is",
        "na",
        "nas",
        "no",
        "nos",
        "o",
        "os",
        "para",
        "por",
        "qual",
        "quais",
        "que",
        "sao",
        "the",
        "um",
        "uma",
        "what",
    }
)
_LEXICAL_TOKEN_ALIASES = {
    "item": {"itens"},
    "itens": {"item"},
    "nivel": {"status"},
    "status": {"nivel"},
}


class PartnerKnowledgeIndexUnavailableError(RuntimeError):
    """Raised when the required persistent Partner knowledge index is unusable."""


@dataclass(frozen=True)
class RetrievedEvidence:
    """Internal evidence selected from the Partner knowledge index."""

    text: str
    document_name: str
    location: str
    technical_location: str
    relevance_score: float


@runtime_checkable
class PartnerKnowledgeRetriever(Protocol):
    """Replaceable boundary for retrieving approved Partner knowledge."""

    def ensure_available(self) -> None:
        """Verify that the backing index can safely serve retrieval requests."""

    def retrieve(self, query: str) -> list[RetrievedEvidence]:
        """Return only evidence that meets the configured relevance threshold."""


class PersistentChromaRetriever:
    """Retrieve grounded evidence from the local persistent Partner knowledge index.

    The adapter uses Cohere semantic retrieval as its primary signal and a bounded
    lexical rerank to recover exact identifiers and fields in structured CSV/JSON
    records. Keeping both signals at this boundary lets a future hosted index replace
    the implementation without route changes.
    """

    def __init__(
        self,
        index_path: Path,
        *,
        embed_query: Callable[[str], list[float]] | None = None,
        candidate_limit: int = 8,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        embedding_model: str = "embed-v4.0",
        query_embedding_cache_size: int = 1024,
        embedding_metadata: Mapping[str, str] | None = None,
        embedding_cache_key: str | None = None,
        query_embedding_cache_path: Path | None = None,
    ):
        self._index_path = index_path
        self._embed_query = embed_query
        self._candidate_limit = candidate_limit
        self._relevance_threshold = relevance_threshold
        self._embedding_model = embedding_model
        self._embedding_metadata = dict(embedding_metadata or {})
        self._embedding_cache_key = embedding_cache_key or embedding_model
        self._query_embedding_cache_path = (
            query_embedding_cache_path or index_path / ".query-embedding-cache.sqlite3"
        )
        self._query_embedding_lock_directory = (
            self._query_embedding_cache_path.parent / ".query-embedding-locks"
        )
        self._query_embedding_cache = QueryEmbeddingCache(
            self._query_embedding_cache_path,
            max_entries=query_embedding_cache_size,
        )

    def ensure_available(self) -> None:
        self._ensure_embedding_profile()
        self._validate_index_path()
        try:
            with partner_knowledge_index_lock(
                self._index_path,
                exclusive=False,
                lock_name=ACTIVATION_LOCK_NAME,
            ):
                self._ensure_collection_available()
        except (ChromaError, OSError, ValueError, sqlite3.Error) as exc:
            raise PartnerKnowledgeIndexUnavailableError(
                f"Partner knowledge index is unreadable at {self._index_path}."
            ) from exc

    def _validate_index_path(self) -> None:
        if not self._index_path.exists():
            raise PartnerKnowledgeIndexUnavailableError(
                "Partner knowledge index does not exist at "
                f"{self._index_path}. Run the Partner knowledge ingestion operation "
                "before starting the API."
            )

        if not self._index_path.is_dir():
            raise PartnerKnowledgeIndexUnavailableError(
                f"Partner knowledge index path is not a directory: {self._index_path}"
            )

        database_path = self._index_path / "chroma.sqlite3"
        if not database_path.is_file():
            raise PartnerKnowledgeIndexUnavailableError(
                "Partner knowledge index is incomplete or unreadable at "
                f"{self._index_path}: expected chroma.sqlite3. "
                "Run the Partner knowledge ingestion operation before starting the API."
            )

    def _ensure_embedding_profile(self) -> None:
        if self._embed_query is None:
            # Construct the fallback adapter before validating the collection so its
            # provider/model/dimension profile is always part of the check.
            self._get_embed_query()

    def _ensure_collection_available(self) -> None:
        client = chromadb.PersistentClient(path=str(self._index_path))
        collection = client.get_collection(active_collection_name(self._index_path))
        if collection.count() == 0:
            raise PartnerKnowledgeIndexUnavailableError(
                "Partner knowledge index contains no indexed records at "
                f"{self._index_path}."
            )
        if self._embedding_metadata and not _metadata_matches(
            collection.metadata or {}, self._embedding_metadata
        ):
            raise PartnerKnowledgeIndexUnavailableError(
                "Partner knowledge index embedding configuration does not match "
                f"the configured Cohere embeddings at {self._index_path}. "
                "Run the Partner knowledge ingestion operation before starting "
                "the API."
            )

    def retrieve(self, query: str) -> list[RetrievedEvidence]:
        self._ensure_embedding_profile()
        self._validate_index_path()
        try:
            with partner_knowledge_index_lock(
                self._index_path,
                exclusive=False,
                lock_name=ACTIVATION_LOCK_NAME,
            ):
                self._ensure_collection_available()
        except (ChromaError, OSError, ValueError, sqlite3.Error) as exc:
            raise PartnerKnowledgeIndexUnavailableError(
                f"Partner knowledge index is unreadable at {self._index_path}."
            ) from exc
        try:
            query_embedding = self._get_query_embedding(query)
        except (OSError, ValueError, sqlite3.Error) as exc:
            raise PartnerKnowledgeIndexUnavailableError(
                f"Partner knowledge index is unreadable at {self._index_path}."
            ) from exc
        try:
            with partner_knowledge_index_lock(
                self._index_path,
                exclusive=False,
                lock_name=ACTIVATION_LOCK_NAME,
            ):
                self._ensure_collection_available()
                client = chromadb.PersistentClient(path=str(self._index_path))
                collection = client.get_collection(
                    active_collection_name(self._index_path)
                )
                semantic_candidate_limit = min(
                    self._candidate_limit * _SEMANTIC_CANDIDATE_MULTIPLIER,
                    collection.count(),
                )
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=semantic_candidate_limit,
                    include=["documents", "metadatas", "distances"],
                )
        except (ChromaError, OSError, ValueError, sqlite3.Error) as exc:
            raise PartnerKnowledgeIndexUnavailableError(
                f"Partner knowledge index is unreadable at {self._index_path}."
            ) from exc

        candidates = []
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=True,
        ):
            # Chroma's cosine distance maps directly onto a [0, 1] semantic score.
            semantic_score = max(0.0, min(1.0, 1.0 - float(distance)))
            lexical_score = _lexical_relevance_score(query, text, metadata)
            relevance_score = _combined_relevance_score(
                semantic_score,
                lexical_score,
                threshold=self._relevance_threshold,
            )
            if relevance_score >= self._relevance_threshold:
                candidates.append(
                    (
                        relevance_score,
                        RetrievedEvidence(
                            text=text,
                            document_name=metadata["document_name"],
                            location=metadata["location"],
                            technical_location=metadata["technical_location"],
                            relevance_score=relevance_score,
                        ),
                    )
                )
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [evidence for _, evidence in candidates[: self._candidate_limit]]

    def _get_query_embedding(self, query: str) -> list[float]:
        normalized_query = query.lower().strip()
        cached_embedding = self._query_embedding_cache.get(
            self._embedding_cache_key, normalized_query
        )
        if cached_embedding is not None:
            return cached_embedding

        # A per-query lock on the shared index volume prevents concurrent identical
        # requests in this process or another API container from spending duplicate
        # Cohere calls before the first result is persisted.
        cache_lock_key = f"{self._embedding_cache_key}\x00{normalized_query}"
        with _query_embedding_file_lock(
            self._query_embedding_lock_directory, cache_lock_key
        ):
            cached_embedding = self._query_embedding_cache.get(
                self._embedding_cache_key, normalized_query
            )
            if cached_embedding is not None:
                return cached_embedding
            try:
                embedding = self._get_embed_query()(normalized_query)
            except Exception as exc:
                if is_provider_rate_limit_error(exc):
                    raise provider_rate_limit_error("Cohere embeddings", exc) from exc
                raise
            self._query_embedding_cache.set(
                self._embedding_cache_key, normalized_query, embedding
            )
            return embedding

    def _get_embed_query(self) -> Callable[[str], list[float]]:
        if self._embed_query is None:
            from src.partner_knowledge.cohere_embeddings import CohereEmbeddingClient
            from src.partner_knowledge.config import get_partner_knowledge_settings
            from src.partner_knowledge.embedding_budget import EmbeddingUsageLedger

            settings = get_partner_knowledge_settings()
            embeddings = CohereEmbeddingClient(
                settings.cohere_api_key,
                model=settings.embedding_model,
                embedding_dimension=settings.embedding_dimension,
                ledger=EmbeddingUsageLedger(
                    self._index_path / ".cohere-embedding-usage.sqlite3",
                    monthly_limit=settings.embedding_monthly_call_limit,
                ),
            )
            self._embed_query = embeddings.embed_query
            self._embedding_metadata = embeddings.collection_metadata
            self._embedding_cache_key = (
                f"cohere:{embeddings.model}:{embeddings.embedding_dimension}"
            )
        return self._embed_query


@contextmanager
def _query_embedding_file_lock(directory: Path, key: str) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(key.encode()).digest()
    slot = int.from_bytes(digest[:2], "big") % _QUERY_EMBEDDING_LOCK_SLOT_COUNT
    lock_path = directory / f"slot-{slot:03d}.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _metadata_matches(
    actual: Mapping[str, object], expected: Mapping[str, str]
) -> bool:
    return all(str(actual.get(key)) == value for key, value in expected.items())


def _combined_relevance_score(
    semantic_score: float,
    lexical_score: float,
    *,
    threshold: float,
) -> float:
    """Combine Cohere similarity with exact business-term overlap.

    A semantic match that already clears the configured threshold remains valid. For
    weaker semantic matches, lexical overlap can recover structured records whose
    exact unit, item, status, or policy terms are meaningful but underrepresented in
    the embedding ranking.
    """

    if semantic_score >= threshold:
        return semantic_score
    if semantic_score < _MINIMUM_SEMANTIC_SCORE_FOR_LEXICAL_RESCUE:
        return semantic_score
    return (
        semantic_score * _SEMANTIC_RELEVANCE_WEIGHT
        + lexical_score * _LEXICAL_RELEVANCE_WEIGHT
    )


def _lexical_relevance_score(
    query: str,
    text: str,
    metadata: Mapping[str, object],
) -> float:
    query_tokens = _content_token_variants(query)
    if not query_tokens:
        return 0.0
    searchable_text = " ".join(
        (
            text,
            str(metadata.get("document_name", "")),
            str(metadata.get("location", "")),
        )
    )
    candidate_tokens = {
        variant
        for token in _normalized_tokens(searchable_text)
        for variant in _token_variants(token)
    }
    matches = sum(bool(variants & candidate_tokens) for variants in query_tokens)
    return matches / len(query_tokens)


def _content_token_variants(value: str) -> list[set[str]]:
    return [
        set(_token_variants(token))
        for token in _normalized_tokens(value)
        if token not in _LEXICAL_STOP_WORDS
    ]


def _normalized_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.findall(r"[a-z0-9]+", ascii_text.lower())


def _token_variants(token: str) -> set[str]:
    variants = {token}
    variants.update(_LEXICAL_TOKEN_ALIASES.get(token, set()))
    if token.endswith("s") and len(token) > 4:
        variants.add(token[:-1])
    if len(token) >= 6:
        variants.add(token[:6])
    return variants

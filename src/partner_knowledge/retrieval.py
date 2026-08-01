from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import chromadb
from chromadb.errors import ChromaError

from src.cache import QueryEmbeddingCache
from src.provider_errors import (
    is_provider_rate_limit_error,
    provider_rate_limit_error,
)


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
    """Availability check for the local persistent Chroma Partner knowledge index.

    Retrieval implementation is added with the ingestion and chat-flow tickets. Keeping
    this adapter at the boundary lets a future hosted index replace it without route
    changes.
    """

    _COLLECTION_NAME = "partner_knowledge"

    def __init__(
        self,
        index_path: Path,
        *,
        embed_query: Callable[[str], list[float]] | None = None,
        candidate_limit: int = 8,
        relevance_threshold: float = 0.75,
        embedding_model: str = "gemini-embedding-001",
        query_embedding_cache_size: int = 128,
        query_embedding_cache_path: Path | None = None,
    ):
        self._index_path = index_path
        self._embed_query = embed_query
        self._candidate_limit = candidate_limit
        self._relevance_threshold = relevance_threshold
        self._embedding_model = embedding_model
        self._query_embedding_cache = QueryEmbeddingCache(
            query_embedding_cache_path or index_path / ".query-embedding-cache.sqlite3",
            max_entries=query_embedding_cache_size,
        )

    def ensure_available(self) -> None:
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

        try:
            client = chromadb.PersistentClient(path=str(self._index_path))
            collection = client.get_collection(self._COLLECTION_NAME)
            if collection.count() == 0:
                raise PartnerKnowledgeIndexUnavailableError(
                    "Partner knowledge index contains no indexed records at "
                    f"{self._index_path}."
                )
        except (ChromaError, OSError, ValueError) as exc:
            raise PartnerKnowledgeIndexUnavailableError(
                f"Partner knowledge index is unreadable at {self._index_path}."
            ) from exc

    def retrieve(self, query: str) -> list[RetrievedEvidence]:
        self.ensure_available()
        try:
            client = chromadb.PersistentClient(path=str(self._index_path))
            collection = client.get_collection(self._COLLECTION_NAME)
            results = collection.query(
                query_embeddings=[self._get_query_embedding(query)],
                n_results=self._candidate_limit,
                include=["documents", "metadatas", "distances"],
            )
        except (ChromaError, OSError, ValueError) as exc:
            raise PartnerKnowledgeIndexUnavailableError(
                f"Partner knowledge index is unreadable at {self._index_path}."
            ) from exc

        evidence = []
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=True,
        ):
            # Chroma's cosine distance maps directly onto a [0, 1] relevance score.
            relevance_score = max(0.0, min(1.0, 1.0 - float(distance)))
            if relevance_score >= self._relevance_threshold:
                evidence.append(
                    RetrievedEvidence(
                        text=text,
                        document_name=metadata["document_name"],
                        location=metadata["location"],
                        technical_location=metadata["technical_location"],
                        relevance_score=relevance_score,
                    )
                )
        return evidence

    def _get_query_embedding(self, query: str) -> list[float]:
        normalized_query = query.lower().strip()
        cached_embedding = self._query_embedding_cache.get(
            self._embedding_model, normalized_query
        )
        if cached_embedding is not None:
            return cached_embedding

        try:
            embedding = self._get_embed_query()(normalized_query)
        except Exception as exc:
            if is_provider_rate_limit_error(exc):
                raise provider_rate_limit_error("Google embeddings", exc) from exc
            raise
        self._query_embedding_cache.set(
            self._embedding_model, normalized_query, embedding
        )
        return embedding

    def _get_embed_query(self) -> Callable[[str], list[float]]:
        if self._embed_query is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            from src.config import get_settings
            from src.partner_knowledge.config import get_partner_knowledge_settings

            settings = get_partner_knowledge_settings()
            embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.embedding_model,
                google_api_key=get_settings().google_api_key,
            )
            self._embed_query = embeddings.embed_query
        return self._embed_query

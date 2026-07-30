import sqlite3
from pathlib import Path
from typing import Protocol, runtime_checkable


class PartnerKnowledgeIndexUnavailableError(RuntimeError):
    """Raised when the required persistent Partner knowledge index is unusable."""


@runtime_checkable
class PartnerKnowledgeRetriever(Protocol):
    """Replaceable boundary for retrieving approved Partner knowledge."""

    def ensure_available(self) -> None:
        """Verify that the backing index can safely serve retrieval requests."""


class PersistentChromaRetriever:
    """Availability check for the local persistent Chroma Partner knowledge index.

    Retrieval implementation is added with the ingestion and chat-flow tickets. Keeping
    this adapter at the boundary lets a future hosted index replace it without route
    changes.
    """

    _DATABASE_FILENAME = "chroma.sqlite3"

    def __init__(self, index_path: Path):
        self._index_path = index_path

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

        database_path = self._index_path / self._DATABASE_FILENAME
        if not database_path.is_file():
            raise PartnerKnowledgeIndexUnavailableError(
                "Partner knowledge index is incomplete or unreadable at "
                f"{self._index_path}: expected {self._DATABASE_FILENAME}. "
                "Run the Partner knowledge ingestion operation before starting the API."
            )

        try:
            connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
            with connection:
                collection_table = connection.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'collections'"
                ).fetchone()
        except sqlite3.Error as exc:
            raise PartnerKnowledgeIndexUnavailableError(
                f"Partner knowledge index is unreadable at {self._index_path}."
            ) from exc

        if collection_table is None:
            raise PartnerKnowledgeIndexUnavailableError(
                "Partner knowledge index is not a valid Chroma index at "
                f"{self._index_path}."
            )

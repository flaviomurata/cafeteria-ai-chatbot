"""Persistent usage accounting for the Cohere trial embedding budget."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.provider_errors import ProviderRateLimitError

DEFAULT_MONTHLY_CALL_LIMIT = 900


@dataclass(frozen=True)
class EmbeddingUsage:
    month: str
    used_calls: int
    remaining_calls: int


class EmbeddingUsageLedgerError(RuntimeError):
    """Raised when embedding usage cannot be tracked safely."""


class EmbeddingBudgetExceededError(ProviderRateLimitError):
    """Raised before a Cohere request would exceed the local trial budget."""

    def __init__(
        self,
        *,
        month: str,
        used_calls: int,
        requested_calls: int,
        monthly_limit: int,
    ):
        self.month = month
        self.used_calls = used_calls
        self.requested_calls = requested_calls
        self.monthly_limit = monthly_limit
        super().__init__("Cohere embeddings")
        self.args = (
            "Cohere embedding budget exhausted for "
            f"{month}: {used_calls} calls used, {requested_calls} requested, "
            f"{monthly_limit} allowed.",
        )


class EmbeddingUsageLedger:
    """Atomically reserve Cohere embedding calls in a shared SQLite file."""

    def __init__(
        self,
        path: Path,
        *,
        monthly_limit: int = DEFAULT_MONTHLY_CALL_LIMIT,
    ):
        if monthly_limit < 1:
            raise ValueError("monthly_limit must be positive")
        self._path = path
        self._monthly_limit = monthly_limit

    def snapshot(self, *, now: datetime | None = None) -> EmbeddingUsage:
        month = _utc_month(now)
        try:
            with self._connect() as database:
                row = database.execute(
                    """
                    SELECT used_calls
                    FROM cohere_embedding_usage
                    WHERE month = ?
                    """,
                    (month,),
                ).fetchone()
        except (OSError, sqlite3.Error) as exc:
            raise EmbeddingUsageLedgerError(
                f"Unable to read Cohere embedding usage ledger at {self._path}."
            ) from exc
        used_calls = int(row[0]) if row is not None else 0
        return EmbeddingUsage(
            month=month,
            used_calls=used_calls,
            remaining_calls=max(0, self._monthly_limit - used_calls),
        )

    def reserve(self, calls: int, *, now: datetime | None = None) -> EmbeddingUsage:
        if calls < 1:
            raise ValueError("calls must be positive")
        month = _utc_month(now)
        try:
            with self._connect() as database:
                database.execute("BEGIN IMMEDIATE")
                row = database.execute(
                    """
                    SELECT used_calls
                    FROM cohere_embedding_usage
                    WHERE month = ?
                    """,
                    (month,),
                ).fetchone()
                used_calls = int(row[0]) if row is not None else 0
                if used_calls + calls > self._monthly_limit:
                    raise EmbeddingBudgetExceededError(
                        month=month,
                        used_calls=used_calls,
                        requested_calls=calls,
                        monthly_limit=self._monthly_limit,
                    )
                database.execute(
                    """
                    INSERT INTO cohere_embedding_usage (month, used_calls)
                    VALUES (?, ?)
                    ON CONFLICT(month) DO UPDATE SET
                        used_calls = excluded.used_calls
                    """,
                    (month, used_calls + calls),
                )
        except EmbeddingBudgetExceededError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise EmbeddingUsageLedgerError(
                f"Unable to reserve Cohere embedding usage at {self._path}."
            ) from exc
        return EmbeddingUsage(
            month=month,
            used_calls=used_calls + calls,
            remaining_calls=self._monthly_limit - used_calls - calls,
        )

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        database = sqlite3.connect(self._path, timeout=5.0)
        database.execute("PRAGMA busy_timeout = 5000")
        database.execute(
            """
            CREATE TABLE IF NOT EXISTS cohere_embedding_usage (
                month TEXT PRIMARY KEY,
                used_calls INTEGER NOT NULL CHECK (used_calls >= 0)
            )
            """
        )
        return database


def _utc_month(now: datetime | None) -> str:
    timestamp = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).strftime("%Y-%m")

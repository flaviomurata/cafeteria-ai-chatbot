"""Atomic selection of the active Chroma Partner knowledge collection."""

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4

COLLECTION_NAME = "partner_knowledge"
STAGING_COLLECTION_PREFIX = "partner_knowledge_staging_"
ACTIVE_COLLECTION_POINTER_NAME = ".partner-knowledge-active"
INDEX_LOCK_NAME = ".partner-knowledge-ingestion.lock"
ACTIVATION_LOCK_NAME = ".partner-knowledge-activation.lock"


class PartnerKnowledgeIndexLockError(OSError):
    """Raised when the shared Partner knowledge lock cannot be acquired."""


@contextmanager
def partner_knowledge_index_lock(
    index_path: Path,
    *,
    exclusive: bool,
    non_blocking: bool = False,
    lock_name: str = ACTIVATION_LOCK_NAME,
    lock_directory: Path | None = None,
) -> Iterator[None]:
    """Coordinate readers and the single writer across processes."""
    lock_root = lock_directory or index_path
    if lock_directory is not None:
        lock_root.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_root / lock_name, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        raise PartnerKnowledgeIndexLockError(
            f"Unable to open the Partner knowledge index lock at {index_path}."
        ) from exc
    try:
        flags = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if non_blocking:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(descriptor, flags)
        except BlockingIOError:
            raise
        except OSError as exc:
            raise PartnerKnowledgeIndexLockError(
                f"Unable to acquire the Partner knowledge index lock at {index_path}."
            ) from exc
        try:
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError as exc:
                raise PartnerKnowledgeIndexLockError(
                    "Unable to release the Partner knowledge index lock at "
                    f"{index_path}."
                ) from exc
    finally:
        try:
            os.close(descriptor)
        except OSError as exc:
            raise PartnerKnowledgeIndexLockError(
                f"Unable to close the Partner knowledge index lock at {index_path}."
            ) from exc


def active_collection_name(index_path: Path) -> str:
    """Return the selected collection, retaining compatibility with old indexes."""
    pointer_path = index_path / ACTIVE_COLLECTION_POINTER_NAME
    if not pointer_path.exists():
        return COLLECTION_NAME
    if not pointer_path.is_file():
        raise ValueError(
            f"Active Partner knowledge pointer is not a file: {pointer_path}"
        )
    try:
        collection_name = pointer_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise OSError(
            f"Unable to read the active Partner knowledge pointer at {pointer_path}."
        ) from exc
    _validate_collection_name(collection_name)
    return collection_name


def activate_collection(index_path: Path, collection_name: str) -> None:
    """Atomically point readers at a fully written collection."""
    _validate_collection_name(collection_name)
    pointer_path = index_path / ACTIVE_COLLECTION_POINTER_NAME
    temporary_path = pointer_path.with_name(f"{pointer_path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(f"{collection_name}\n", encoding="utf-8")
        os.replace(temporary_path, pointer_path)
    except OSError:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


def _validate_collection_name(collection_name: str) -> None:
    if (
        not collection_name
        or any(character in collection_name for character in "/\\\n\r\t")
        or (
            collection_name != COLLECTION_NAME
            and not collection_name.startswith(STAGING_COLLECTION_PREFIX)
        )
    ):
        raise ValueError(
            "Active Partner knowledge pointer contains an invalid collection name."
        )

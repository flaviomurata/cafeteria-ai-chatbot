from pathlib import Path

import chromadb
import pytest

from src.main import app
from src.partner_knowledge.config import PartnerKnowledgeSettings
from src.partner_knowledge.retrieval import (
    PartnerKnowledgeIndexUnavailableError,
    PersistentChromaRetriever,
)


def test_partner_knowledge_settings_accept_configured_retrieval_values(tmp_path: Path):
    settings = PartnerKnowledgeSettings(
        partner_document_source=tmp_path / "documents",
        partner_index_path=tmp_path / "index",
        embedding_model="test-embedding-model",
        retrieval_candidate_limit=7,
        relevance_threshold=0.83,
    )

    assert settings.partner_document_source == tmp_path / "documents"
    assert settings.partner_index_path == tmp_path / "index"
    assert settings.embedding_model == "test-embedding-model"
    assert settings.retrieval_candidate_limit == 7
    assert settings.relevance_threshold == 0.83


def test_partner_knowledge_settings_use_a_domain_environment_prefix(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("PARTNER_KNOWLEDGE_EMBEDDING_MODEL", "test-embedding-model")

    settings = PartnerKnowledgeSettings(_env_file=None)

    assert settings.embedding_model == "test-embedding-model"


def test_persistent_chroma_retriever_rejects_a_missing_index(tmp_path: Path):
    retriever = PersistentChromaRetriever(tmp_path / "missing-index")

    with pytest.raises(PartnerKnowledgeIndexUnavailableError, match="does not exist"):
        retriever.ensure_available()


def test_persistent_chroma_retriever_rejects_an_incomplete_index(tmp_path: Path):
    index_path = tmp_path / "index"
    index_path.mkdir()
    retriever = PersistentChromaRetriever(index_path)

    with pytest.raises(PartnerKnowledgeIndexUnavailableError, match="chroma.sqlite3"):
        retriever.ensure_available()


def test_persistent_chroma_retriever_accepts_a_readable_chroma_index(tmp_path: Path):
    index_path = tmp_path / "index"
    index_path.mkdir()
    client = chromadb.PersistentClient(path=str(index_path))
    collection = client.get_or_create_collection("partner_knowledge")
    collection.add(
        ids=["catalog-product-1"],
        documents=["Coffee beans are approved for Partner orders."],
        embeddings=[[0.1, 0.2, 0.3]],
    )
    retriever = PersistentChromaRetriever(index_path)

    retriever.ensure_available()


def test_persistent_chroma_retriever_rejects_an_empty_index(tmp_path: Path):
    index_path = tmp_path / "index"
    chromadb.PersistentClient(path=str(index_path)).get_or_create_collection(
        "partner_knowledge"
    )
    retriever = PersistentChromaRetriever(index_path)

    with pytest.raises(
        PartnerKnowledgeIndexUnavailableError, match="no indexed records"
    ):
        retriever.ensure_available()


def test_persistent_chroma_retriever_rejects_an_invalid_chroma_database(
    tmp_path: Path,
):
    index_path = tmp_path / "index"
    index_path.mkdir()
    (index_path / "chroma.sqlite3").write_text("not a sqlite database")
    retriever = PersistentChromaRetriever(index_path)

    with pytest.raises(PartnerKnowledgeIndexUnavailableError, match="unreadable"):
        retriever.ensure_available()


@pytest.mark.asyncio
async def test_api_startup_fails_when_the_partner_knowledge_index_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from src import main

    settings = PartnerKnowledgeSettings(partner_index_path=tmp_path / "missing-index")
    monkeypatch.setattr(main, "get_partner_knowledge_settings", lambda: settings)

    with pytest.raises(PartnerKnowledgeIndexUnavailableError, match="does not exist"):
        async with app.router.lifespan_context(app):
            pass

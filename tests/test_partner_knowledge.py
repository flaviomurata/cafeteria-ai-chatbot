import shutil
from pathlib import Path

import chromadb
import pytest

from src.main import app
from src.partner_knowledge.config import PartnerKnowledgeSettings
from src.partner_knowledge.retrieval import (
    PartnerKnowledgeIndexUnavailableError,
    PersistentChromaRetriever,
)


def _embed_documents(documents: list[str]) -> list[list[float]]:
    """Deterministic embeddings keep ingestion tests independent of Gemini."""
    return [[float(len(document)), 0.0, 1.0] for document in documents]


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


def test_ingestion_builds_citation_ready_index_from_only_approved_sources(
    tmp_path: Path,
):
    from src.partner_knowledge.ingestion import PartnerKnowledgeIngestor

    source_path = tmp_path / "documents"
    shutil.copytree(Path("media/cafeteria-documents"), source_path)
    index_path = tmp_path / "index"

    result = PartnerKnowledgeIngestor(
        PartnerKnowledgeSettings(
            partner_document_source=source_path,
            partner_index_path=index_path,
        ),
        embed_documents=_embed_documents,
    ).ingest()

    collection = chromadb.PersistentClient(path=str(index_path)).get_collection(
        "partner_knowledge"
    )
    records = collection.get(include=["documents", "metadatas"])
    document_names = {metadata["document_name"] for metadata in records["metadatas"]}

    assert result.indexed_chunks == len(records["ids"])
    assert result.indexed_chunks > 0
    assert document_names == {
        "Catálogo de Produtos e Ingredientes — Café Aurora",
        "Controle de Estoque",
        "Configuração das Unidades",
        "Manual de Operações das Unidades — Café Aurora",
        "Guia de Atendimento ao Cliente",
        "Política de Despesas e Reembolsos",
    }
    assert all(
        "FAQ" not in name and "Recursos Humanos" not in name for name in document_names
    )
    assert all(metadata["location"] for metadata in records["metadatas"])
    assert any(
        metadata["document_name"] == "Controle de Estoque"
        and "CA-CPS-01" in metadata["location"]
        and "ING-001" in metadata["location"]
        and metadata["technical_location"].startswith("row:")
        for metadata in records["metadatas"]
    )
    assert any("Até R$ 150,00" in document for document in records["documents"])


def test_ingestion_replaces_stale_chunks_when_rebuilt(tmp_path: Path):
    from src.partner_knowledge.ingestion import PartnerKnowledgeIngestor

    source_path = tmp_path / "documents"
    source_path.mkdir()
    inventory_path = source_path / "CA-COM-PLA-001_Controle_de_Estoque.csv"
    inventory_path.write_text(
        "Código da unidade,Código do item,Descrição\nCA-TEST-01,ING-OLD,Item antigo\n",
        encoding="utf-8",
    )
    settings = PartnerKnowledgeSettings(
        partner_document_source=source_path,
        partner_index_path=tmp_path / "index",
    )
    ingestor = PartnerKnowledgeIngestor(settings, embed_documents=_embed_documents)

    ingestor.ingest()
    inventory_path.write_text(
        "Código da unidade,Código do item,Descrição\nCA-TEST-01,ING-NEW,Item atual\n",
        encoding="utf-8",
    )
    ingestor.ingest()

    records = (
        chromadb.PersistentClient(path=str(settings.partner_index_path))
        .get_collection("partner_knowledge")
        .get(include=["documents"])
    )

    assert len(records["ids"]) == 1
    assert "ING-NEW" in records["documents"][0]
    assert "ING-OLD" not in records["documents"][0]


def test_failed_rebuild_leaves_the_current_index_usable(tmp_path: Path):
    from src.partner_knowledge.ingestion import (
        PartnerKnowledgeIngestionError,
        PartnerKnowledgeIngestor,
    )

    source_path = tmp_path / "documents"
    source_path.mkdir()
    (source_path / "CA-COM-PLA-001_Controle_de_Estoque.csv").write_text(
        "Código da unidade,Código do item,Descrição\nCA-TEST-01,ING-OLD,Item antigo\n",
        encoding="utf-8",
    )
    settings = PartnerKnowledgeSettings(
        partner_document_source=source_path,
        partner_index_path=tmp_path / "index",
    )
    PartnerKnowledgeIngestor(settings, embed_documents=_embed_documents).ingest()

    with pytest.raises(PartnerKnowledgeIngestionError, match="Unable to write"):
        PartnerKnowledgeIngestor(
            settings,
            embed_documents=lambda documents: [["not a number"] for _ in documents],
        ).ingest()

    records = (
        chromadb.PersistentClient(path=str(settings.partner_index_path))
        .get_collection("partner_knowledge")
        .get(include=["documents"])
    )

    assert records["documents"] == [
        "Código da unidade: CA-TEST-01\nCódigo do item: ING-OLD\nDescrição: Item antigo"
    ]


def test_ingestion_rejects_a_pdf_without_native_text(tmp_path: Path):
    from src.partner_knowledge.ingestion import (
        PartnerKnowledgeIngestionError,
        PartnerKnowledgeIngestor,
    )

    source_path = tmp_path / "documents"
    source_path.mkdir()
    (source_path / "Catálogo de Produtos e Ingredientes — Café Aurora.pdf").write_bytes(
        b"not a readable PDF"
    )

    with pytest.raises(PartnerKnowledgeIngestionError, match="native text"):
        PartnerKnowledgeIngestor(
            PartnerKnowledgeSettings(
                partner_document_source=source_path,
                partner_index_path=tmp_path / "index",
            ),
            embed_documents=_embed_documents,
        ).ingest()


def test_ingestion_rejects_an_unreadable_docx_with_a_clear_error(tmp_path: Path):
    from src.partner_knowledge.ingestion import (
        PartnerKnowledgeIngestionError,
        PartnerKnowledgeIngestor,
    )

    source_path = tmp_path / "documents"
    source_path.mkdir()
    (source_path / "CA-FIN-POL-001_Politica_de_Despesas_e_Reembolsos.docx").write_bytes(
        b"not a readable DOCX"
    )

    with pytest.raises(PartnerKnowledgeIngestionError, match="native text"):
        PartnerKnowledgeIngestor(
            PartnerKnowledgeSettings(
                partner_document_source=source_path,
                partner_index_path=tmp_path / "index",
            ),
            embed_documents=_embed_documents,
        ).ingest()

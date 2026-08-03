import base64
import json
import shutil
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from pathlib import Path

import chromadb
import pytest
from pydantic import ValidationError

from src.main import app
from src.partner_knowledge.config import PartnerKnowledgeSettings
from src.partner_knowledge.constants import APPROVED_PARTNER_DOCUMENT_NAMES
from src.partner_knowledge.index_storage import active_collection_name
from src.partner_knowledge.retrieval import (
    PartnerKnowledgeIndexUnavailableError,
    PersistentChromaRetriever,
    prepare_runtime_index,
)


def _embed_documents(documents: list[str]) -> list[list[float]]:
    """Deterministic embeddings keep ingestion tests independent of providers."""
    return [[float(len(document)), 0.0, 1.0] for document in documents]


def test_partner_knowledge_settings_accept_configured_retrieval_values(tmp_path: Path):
    settings = PartnerKnowledgeSettings(
        partner_document_source=tmp_path / "documents",
        partner_index_path=tmp_path / "index",
        runtime_data_path=tmp_path / "runtime",
        retrieval_candidate_limit=7,
        relevance_threshold=0.83,
    )

    assert settings.partner_document_source == tmp_path / "documents"
    assert settings.partner_index_path == tmp_path / "index"
    assert settings.embedding_usage_ledger_path == (
        tmp_path / "runtime" / ".cohere-embedding-usage.sqlite3"
    )
    assert settings.runtime_index_path == (
        tmp_path / "runtime" / "partner-knowledge-index"
    )
    assert settings.embedding_model == "embed-v4.0"
    assert settings.retrieval_candidate_limit == 7
    assert settings.relevance_threshold == 0.83


def test_partner_knowledge_settings_use_a_domain_environment_prefix(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("PARTNER_KNOWLEDGE_RELEVANCE_THRESHOLD", "0.91")

    settings = PartnerKnowledgeSettings(_env_file=None)

    assert settings.relevance_threshold == 0.91


def test_partner_knowledge_settings_reject_legacy_gemini_embedding_model(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("PARTNER_KNOWLEDGE_EMBEDDING_MODEL", "gemini-embedding-001")

    with pytest.raises(ValidationError, match="embed-v4.0"):
        PartnerKnowledgeSettings(_env_file=None)


def test_partner_knowledge_settings_default_to_cohere_embeddings(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("PARTNER_KNOWLEDGE_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("PARTNER_KNOWLEDGE_QUERY_EMBEDDING_CACHE_SIZE", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("CO_API_KEY", raising=False)

    settings = PartnerKnowledgeSettings(_env_file=None)

    assert settings.embedding_model == "embed-v4.0"
    assert settings.query_embedding_cache_size == 1024
    assert settings.relevance_threshold == 0.45
    assert settings.cohere_api_key == ""


def test_partner_knowledge_settings_accepts_cohere_api_key_aliases(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("COHERE_API_KEY", "canonical-key")
    settings = PartnerKnowledgeSettings(_env_file=None)
    assert settings.cohere_api_key == "canonical-key"

    monkeypatch.delenv("COHERE_API_KEY")
    monkeypatch.setenv("CO_API_KEY", "sdk-key")
    settings = PartnerKnowledgeSettings(_env_file=None)
    assert settings.cohere_api_key == "sdk-key"


def test_persistent_chroma_retriever_rejects_a_missing_index(tmp_path: Path):
    retriever = PersistentChromaRetriever(tmp_path / "missing-index")

    with pytest.raises(PartnerKnowledgeIndexUnavailableError, match="does not exist"):
        retriever.ensure_available()


def test_persistent_chroma_retriever_rejects_an_incomplete_index(tmp_path: Path):
    index_path = tmp_path / "index"
    index_path.mkdir()
    retriever = PersistentChromaRetriever(
        index_path,
        embed_query=lambda _query: [0.1, 0.2, 0.3],
    )

    with pytest.raises(PartnerKnowledgeIndexUnavailableError, match="chroma.sqlite3"):
        retriever.ensure_available()


def test_runtime_index_is_copied_to_writable_storage(tmp_path: Path):
    source_index_path = tmp_path / "source-index"
    runtime_data_path = tmp_path / "runtime"
    runtime_index_path = runtime_data_path / "partner-knowledge-index"
    collection = chromadb.PersistentClient(
        path=str(source_index_path)
    ).get_or_create_collection("partner_knowledge")
    collection.add(
        ids=["catalog-product-1"],
        documents=["Coffee beans are approved for Partner orders."],
        metadatas=[
            {
                "document_name": "catalog.csv",
                "location": "Row 1",
                "technical_location": "catalog.csv:1",
            }
        ],
        embeddings=[[0.1, 0.2, 0.3]],
    )

    prepare_runtime_index(
        source_index_path,
        runtime_index_path,
        lock_directory=runtime_data_path,
    )

    retriever = PersistentChromaRetriever(
        runtime_index_path,
        embed_query=lambda _query: [0.1, 0.2, 0.3],
    )
    retriever.ensure_available()
    evidence = retriever.retrieve("coffee beans")

    assert [item.text for item in evidence] == [
        "Coffee beans are approved for Partner orders."
    ]


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
    retriever = PersistentChromaRetriever(
        index_path,
        embed_query=lambda _query: [0.1, 0.2, 0.3],
    )

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


def test_persistent_chroma_retriever_rejects_a_legacy_embedding_index(
    tmp_path: Path,
):
    index_path = tmp_path / "index"
    collection = chromadb.PersistentClient(
        path=str(index_path)
    ).get_or_create_collection("partner_knowledge")
    collection.add(
        ids=["legacy-1"],
        documents=["Legacy embedding record."],
        embeddings=[[0.1, 0.2, 0.3]],
    )
    retriever = PersistentChromaRetriever(
        index_path,
        embedding_metadata={
            "embedding_provider": "cohere",
            "embedding_model": "embed-v4.0",
            "embedding_dimension": "3",
        },
    )

    with pytest.raises(
        PartnerKnowledgeIndexUnavailableError, match="embedding configuration"
    ):
        retriever.ensure_available()


def test_lazy_retriever_rejects_a_legacy_embedding_index(tmp_path: Path):
    index_path = tmp_path / "index"
    collection = chromadb.PersistentClient(
        path=str(index_path)
    ).get_or_create_collection("partner_knowledge")
    collection.add(
        ids=["legacy-1"],
        documents=["Legacy embedding record."],
        embeddings=[[0.1, 0.2, 0.3]],
    )

    with pytest.raises(
        PartnerKnowledgeIndexUnavailableError, match="embedding configuration"
    ):
        PersistentChromaRetriever(index_path).ensure_available()


def test_persistent_chroma_retriever_rejects_deleted_or_unapproved_source_metadata(
    tmp_path: Path,
):
    index_path = tmp_path / "index"
    collection = chromadb.PersistentClient(
        path=str(index_path)
    ).get_or_create_collection(
        "partner_knowledge",
        metadata={
            "hnsw:space": "cosine",
            "embedding_provider": "cohere",
            "embedding_model": "embed-v4.0",
            "embedding_dimension": "1024",
        },
    )
    collection.add(
        ids=["deleted-source-1"],
        documents=["Internal HR guidance must not be served."],
        metadatas=[
            {
                "document_name": "Política de Recursos Humanos",
                "location": "Página 1",
                "technical_location": "page:1",
            }
        ],
        embeddings=[[0.1, 0.2, 0.3]],
    )
    retriever = PersistentChromaRetriever(
        index_path,
        embed_query=lambda _query: [0.1, 0.2, 0.3],
        embedding_metadata={
            "embedding_provider": "cohere",
            "embedding_model": "embed-v4.0",
            "embedding_dimension": "1024",
        },
    )

    with pytest.raises(
        PartnerKnowledgeIndexUnavailableError, match="unexpected or incomplete"
    ):
        retriever.ensure_available()


def test_persistent_chroma_retriever_accepts_the_approved_source_document_set(
    tmp_path: Path,
):
    index_path = tmp_path / "index"
    collection = chromadb.PersistentClient(
        path=str(index_path)
    ).get_or_create_collection(
        "partner_knowledge",
        metadata={
            "hnsw:space": "cosine",
            "embedding_provider": "cohere",
            "embedding_model": "embed-v4.0",
            "embedding_dimension": "3",
        },
    )
    document_names = sorted(APPROVED_PARTNER_DOCUMENT_NAMES)
    collection.add(
        ids=[f"approved-{index}" for index in range(len(document_names))],
        documents=[f"Approved content {index}" for index in range(len(document_names))],
        metadatas=[
            {
                "document_name": document_name,
                "location": "Fixture",
                "technical_location": f"fixture:{index}",
            }
            for index, document_name in enumerate(document_names)
        ],
        embeddings=[[0.1, 0.2, 0.3] for _ in document_names],
    )
    retriever = PersistentChromaRetriever(
        index_path,
        embed_query=lambda _query: [0.1, 0.2, 0.3],
        embedding_metadata={
            "embedding_provider": "cohere",
            "embedding_model": "embed-v4.0",
            "embedding_dimension": "3",
        },
    )

    retriever.ensure_available()


def test_persistent_chroma_retriever_returns_only_relevant_citation_ready_evidence(
    tmp_path: Path,
):
    index_path = tmp_path / "index"
    collection = chromadb.PersistentClient(
        path=str(index_path)
    ).get_or_create_collection("partner_knowledge", metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=["catalog-1", "catalog-2"],
        documents=["Café coado usa grãos Arábica.", "Regra não relacionada."],
        metadatas=[
            {
                "document_name": "Catálogo de Produtos e Ingredientes - Café Aurora",
                "location": "Página 2",
                "technical_location": "page:2",
            },
            {
                "document_name": "Catálogo de Produtos e Ingredientes - Café Aurora",
                "location": "Página 3",
                "technical_location": "page:3",
            },
        ],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    retriever = PersistentChromaRetriever(
        index_path,
        embed_query=lambda _: [1.0, 0.0],
        candidate_limit=2,
        relevance_threshold=0.75,
    )

    evidence = retriever.retrieve("Quais grãos o café coado utiliza?")

    assert len(evidence) == 1
    assert (
        evidence[0].document_name == "Catálogo de Produtos e Ingredientes - Café Aurora"
    )
    assert evidence[0].location == "Página 2"
    assert evidence[0].technical_location == "page:2"
    assert evidence[0].relevance_score == 1.0


def test_persistent_chroma_retriever_rescues_a_lexically_aligned_structured_record(
    tmp_path: Path,
):
    index_path = tmp_path / "index"
    collection = chromadb.PersistentClient(
        path=str(index_path)
    ).get_or_create_collection("partner_knowledge", metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[
            "semantic-distractor-1",
            "semantic-distractor-2",
            "semantic-distractor-3",
            "unit-config",
        ],
        documents=[
            "A general product note without unit configuration details.",
            "A general ingredient note without unit configuration details.",
            "A general operations note without unit configuration details.",
            '{"nome":"Centro","servicos":{"delivery":false}}',
        ],
        metadatas=[
            {
                "document_name": "Catálogo de Produtos e Ingredientes - Café Aurora",
                "location": "Página 1",
                "technical_location": "page:1",
            },
            {
                "document_name": "Catálogo de Produtos e Ingredientes - Café Aurora",
                "location": "Página 2",
                "technical_location": "page:2",
            },
            {
                "document_name": "Manual de Operações das Unidades",
                "location": "Seção 1",
                "technical_location": "section:1",
            },
            {
                "document_name": "Configuração das Unidades",
                "location": "Unidade CA-CPS-01 - Centro",
                "technical_location": "json:unidades[0]",
            },
        ],
        embeddings=[
            [0.44, sqrt(1 - 0.44**2)],
            [0.43, sqrt(1 - 0.43**2)],
            [0.42, sqrt(1 - 0.42**2)],
            [0.315, sqrt(1 - 0.315**2)],
        ],
    )
    retriever = PersistentChromaRetriever(
        index_path,
        embed_query=lambda _query: [1.0, 0.0],
        candidate_limit=1,
    )

    evidence = retriever.retrieve("A unidade Centro oferece delivery?")

    assert len(evidence) == 1
    assert evidence[0].document_name == "Configuração das Unidades"
    assert evidence[0].location == "Unidade CA-CPS-01 - Centro"
    assert evidence[0].relevance_score >= 0.45


def test_persistent_chroma_retriever_rejects_a_lexically_aligned_weak_match(
    tmp_path: Path,
):
    index_path = tmp_path / "index"
    collection = chromadb.PersistentClient(
        path=str(index_path)
    ).get_or_create_collection("partner_knowledge", metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=["weak-match"],
        documents=['{"nome":"Centro","servicos":{"delivery":false}}'],
        metadatas=[
            {
                "document_name": "Configuração das Unidades",
                "location": "Unidade CA-CPS-01 - Centro",
                "technical_location": "json:unidades[0]",
            }
        ],
        embeddings=[[0.10, sqrt(1 - 0.10**2)]],
    )
    retriever = PersistentChromaRetriever(
        index_path,
        embed_query=lambda _query: [1.0, 0.0],
        candidate_limit=1,
    )

    evidence = retriever.retrieve("A unidade Centro oferece delivery?")

    assert evidence == []


def test_frozen_cohere_calibration_recovers_representative_partner_evidence(
    tmp_path: Path,
):
    fixture_path = Path(__file__).parent / "fixtures/cohere_retrieval_calibration.json"
    calibration = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert calibration["embedding_model"] == "embed-v4.0"
    assert calibration["embedding_provider"] == "cohere"
    assert calibration["embedding_dimension"] == 1024
    assert calibration["encoding"] == "zlib+base64 little-endian float32"
    assert calibration["relevance_threshold"] == 0.45

    cases = calibration["cases"]
    index_path = tmp_path / "calibration-index"
    chroma_client = chromadb.PersistentClient(path=str(index_path))
    collection = chroma_client.get_or_create_collection(
        "partner_knowledge", metadata={"hnsw:space": "cosine"}
    )
    embedding_dimension = calibration["embedding_dimension"]

    def decode_embedding(encoded: str) -> list[float]:
        raw = zlib.decompress(base64.b64decode(encoded))
        return list(struct.unpack(f"<{embedding_dimension}f", raw))

    current_query_embedding: list[float] = []

    def embed_query(_query: str) -> list[float]:
        return current_query_embedding

    retriever = PersistentChromaRetriever(
        index_path,
        embed_query=embed_query,
        candidate_limit=1,
    )

    for case_index, case in enumerate(cases):
        current_query_embedding = decode_embedding(case["query_embedding"])
        document_embedding = decode_embedding(case["document_embedding"])
        assert len(current_query_embedding) == embedding_dimension
        assert len(document_embedding) == embedding_dimension
        if case_index:
            collection.delete(ids=["calibrated-evidence"])
        collection.add(
            ids=["calibrated-evidence"],
            documents=[case["text"]],
            metadatas=[
                {
                    "document_name": case["document_name"],
                    "location": case["location"],
                    "technical_location": case["technical_location"],
                }
            ],
            embeddings=[document_embedding],
        )

        evidence = retriever.retrieve(case["query"])

        assert evidence, case["id"]
        assert evidence[0].document_name == case["document_name"]
        assert evidence[0].location == case["location"]
        assert sum(
            query_value * document_value
            for query_value, document_value in zip(
                current_query_embedding, document_embedding, strict=True
            )
        ) == pytest.approx(case["semantic_score"], abs=0.0002)
        assert evidence[0].relevance_score >= calibration["relevance_threshold"]


def test_persistent_chroma_retriever_caches_normalized_query_embeddings(
    tmp_path: Path,
):
    index_path = tmp_path / "index"
    collection = chromadb.PersistentClient(
        path=str(index_path)
    ).get_or_create_collection("partner_knowledge", metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=["catalog-1"],
        documents=["Café coado usa grãos Arábica."],
        metadatas=[
            {
                "document_name": "Catálogo de Produtos e Ingredientes - Café Aurora",
                "location": "Página 2",
                "technical_location": "page:2",
            }
        ],
        embeddings=[[1.0, 0.0]],
    )
    calls: list[str] = []

    def embed_query(query: str) -> list[float]:
        calls.append(query)
        return [1.0, 0.0]

    retriever = PersistentChromaRetriever(
        index_path,
        embed_query=embed_query,
        candidate_limit=1,
        relevance_threshold=0.75,
        embedding_model="test-embedding-model",
        query_embedding_cache_size=2,
    )

    assert retriever.retrieve("What is the special?")
    assert retriever.retrieve("  WHAT IS THE SPECIAL?  ")

    assert len(calls) == 1


def test_persistent_chroma_retriever_single_flights_concurrent_cache_misses(
    tmp_path: Path,
):
    index_path = tmp_path / "index"
    collection = chromadb.PersistentClient(
        path=str(index_path)
    ).get_or_create_collection("partner_knowledge", metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=["catalog-1"],
        documents=["Café coado usa grãos Arábica."],
        metadatas=[
            {
                "document_name": "Catálogo de Produtos e Ingredientes - Café Aurora",
                "location": "Página 2",
                "technical_location": "page:2",
            }
        ],
        embeddings=[[1.0, 0.0]],
    )
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def embed_query(_query: str) -> list[float]:
        nonlocal calls
        with calls_lock:
            calls += 1
        entered.set()
        assert release.wait(timeout=2)
        return [1.0, 0.0]

    retriever = PersistentChromaRetriever(
        index_path,
        embed_query=embed_query,
        candidate_limit=1,
        relevance_threshold=0.75,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(retriever.retrieve, "same question")
        assert entered.wait(timeout=2)
        second = executor.submit(retriever.retrieve, "same question")
        release.set()
        assert first.result()
        assert second.result()

    assert calls == 1


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

    settings = PartnerKnowledgeSettings(
        partner_index_path=tmp_path / "missing-index",
        cohere_api_key="test-key-not-used",
    )
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
        active_collection_name(index_path)
    )
    records = collection.get(include=["documents", "metadatas"])
    document_names = {metadata["document_name"] for metadata in records["metadatas"]}

    assert result.indexed_chunks == len(records["ids"])
    assert result.indexed_chunks > 0
    assert document_names == {
        "Catálogo de Produtos e Ingredientes - Café Aurora",
        "Controle de Estoque",
        "Configuração das Unidades",
        "Manual de Operações das Unidades - Café Aurora",
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
        .get_collection(active_collection_name(settings.partner_index_path))
        .get(include=["documents"])
    )

    assert len(records["ids"]) == 1
    assert "ING-NEW" in records["documents"][0]
    assert "ING-OLD" not in records["documents"][0]


def test_ingestion_skips_an_unchanged_cohere_index(tmp_path: Path):
    from src.partner_knowledge.ingestion import PartnerKnowledgeIngestor

    source_path = tmp_path / "documents"
    source_path.mkdir()
    (source_path / "CA-COM-PLA-001_Controle_de_Estoque.csv").write_text(
        "Código da unidade,Código do item,Descrição\nCA-TEST-01,ING-001,Item atual\n",
        encoding="utf-8",
    )
    settings = PartnerKnowledgeSettings(
        partner_document_source=source_path,
        partner_index_path=tmp_path / "index",
    )
    embedding_metadata = {
        "embedding_provider": "cohere",
        "embedding_model": "embed-v4.0",
        "embedding_dimension": "3",
        "embedding_document_input_type": "search_document",
        "embedding_query_input_type": "search_query",
    }
    calls = 0

    def embed_documents(documents: list[str]) -> list[list[float]]:
        nonlocal calls
        calls += 1
        return _embed_documents(documents)

    ingestor = PartnerKnowledgeIngestor(
        settings,
        embed_documents=embed_documents,
        embedding_metadata=embedding_metadata,
    )

    first_result = ingestor.ingest()
    second_result = ingestor.ingest()

    assert first_result.skipped is False
    assert second_result.skipped is True
    assert calls == 1
    metadata = (
        chromadb.PersistentClient(path=str(settings.partner_index_path))
        .get_collection(active_collection_name(settings.partner_index_path))
        .metadata
    )
    assert metadata["embedding_provider"] == "cohere"
    assert metadata["embedding_model"] == "embed-v4.0"
    assert metadata["embedding_dimension"] == "3"
    assert metadata["source_fingerprint"]


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
        .get_collection(active_collection_name(settings.partner_index_path))
        .get(include=["documents"])
    )

    assert records["documents"] == [
        "Código da unidade: CA-TEST-01\nCódigo do item: ING-OLD\nDescrição: Item antigo"
    ]


def test_failed_activation_leaves_the_previous_index_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from src.partner_knowledge.ingestion import (
        PartnerKnowledgeIngestionError,
        PartnerKnowledgeIngestor,
    )

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
    embedding_metadata = {
        "embedding_provider": "cohere",
        "embedding_model": "embed-v4.0",
        "embedding_dimension": "3",
    }
    ingestor = PartnerKnowledgeIngestor(
        settings,
        embed_documents=_embed_documents,
        embedding_metadata=embedding_metadata,
    )
    ingestor.ingest()
    previous_collection = active_collection_name(settings.partner_index_path)

    inventory_path.write_text(
        "Código da unidade,Código do item,Descrição\nCA-TEST-01,ING-NEW,Item atual\n",
        encoding="utf-8",
    )

    def fail_activation(*_args, **_kwargs):
        raise OSError("simulated pointer failure")

    monkeypatch.setattr(
        "src.partner_knowledge.ingestion.activate_collection", fail_activation
    )

    with pytest.raises(PartnerKnowledgeIngestionError, match="activate"):
        ingestor.ingest()

    assert active_collection_name(settings.partner_index_path) == previous_collection
    records = (
        chromadb.PersistentClient(path=str(settings.partner_index_path))
        .get_collection(previous_collection)
        .get(include=["documents"])
    )
    assert "ING-OLD" in records["documents"][0]


def test_ingestion_rejects_a_pdf_without_native_text(tmp_path: Path):
    from src.partner_knowledge.ingestion import (
        PartnerKnowledgeIngestionError,
        PartnerKnowledgeIngestor,
    )

    source_path = tmp_path / "documents"
    source_path.mkdir()
    (source_path / "Catálogo de Produtos e Ingredientes - Café Aurora.pdf").write_bytes(
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

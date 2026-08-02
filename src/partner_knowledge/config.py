from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.partner_knowledge.embedding_budget import DEFAULT_MONTHLY_CALL_LIMIT


class PartnerKnowledgeSettings(BaseSettings):
    """Configuration for the persistent Partner knowledge index."""

    model_config = SettingsConfigDict(
        env_prefix="PARTNER_KNOWLEDGE_",
        env_file=".env",
        extra="ignore",
    )

    partner_document_source: Path = Path("media/cafeteria-documents")
    partner_index_path: Path = Path("data/partner-knowledge-index")
    cohere_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("COHERE_API_KEY", "CO_API_KEY"),
    )
    embedding_model: Literal["embed-v4.0"] = "embed-v4.0"
    embedding_dimension: Literal[256, 512, 1024, 1536] = 1024
    query_embedding_cache_size: int = Field(default=1024, ge=0)
    retrieval_candidate_limit: int = Field(default=8, ge=1)
    relevance_threshold: float = Field(default=0.75, ge=0, le=1)
    evidence_verifier_model: str = "gemini-3.6-flash"
    evidence_verifier_timeout_seconds: int = Field(default=30, ge=1)

    @property
    def embedding_usage_ledger_path(self) -> Path:
        return self.partner_index_path / ".cohere-embedding-usage.sqlite3"

    @property
    def embedding_monthly_call_limit(self) -> int:
        return DEFAULT_MONTHLY_CALL_LIMIT


@lru_cache
def get_partner_knowledge_settings() -> PartnerKnowledgeSettings:
    return PartnerKnowledgeSettings()

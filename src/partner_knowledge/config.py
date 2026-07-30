from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PartnerKnowledgeSettings(BaseSettings):
    """Configuration for the persistent Partner knowledge index."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    partner_document_source: Path = Path("media/cafeteria-documents")
    partner_index_path: Path = Path("data/partner-knowledge-index")
    embedding_model: str = "gemini-embedding-2"
    retrieval_candidate_limit: int = Field(default=8, ge=1)
    relevance_threshold: float = Field(default=0.75, ge=0, le=1)


@lru_cache
def get_partner_knowledge_settings() -> PartnerKnowledgeSettings:
    return PartnerKnowledgeSettings()

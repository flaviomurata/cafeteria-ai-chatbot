from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    google_api_key: str = ""
    primary_model: str = "gemini-3.6-flash"
    fallback_model: str = "gemini-3.5-flash"

    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    lanchain_project: str = "production-api"

    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
    }

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""Shared fixtures.

The environment is pinned *before* any `src.*` import: `get_settings()` is
`lru_cache`d and `src.main` evaluates `get_settings().rate_limit` at import time
to build the `@limiter.limit` decorator. Setting these afterwards would have no
effect. `load_dotenv()` in `src.main` does not override existing env vars, so
these win over the developer's real `.env`.
"""

import os

RATE_LIMIT_PER_MINUTE = 5

os.environ.update(
    {
        "APP_ENV": "test",
        "GOOGLE_API_KEY": "test-key-not-used",
        "LANGCHAIN_TRACING_V2": "false",
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_API_KEY": "",
        "RATE_LIMIT": f"{RATE_LIMIT_PER_MINUTE}/minute",
        "CACHE_TTL_SECONDS": "300",
    }
)

from dataclasses import dataclass, field  # noqa: E402

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from src.cache import ResponseCache  # noqa: E402
from src.main import (  # noqa: E402
    app,
    get_agent,
    get_cache,
    get_metrics,
    get_partner_knowledge_retriever,
    get_security,
    limiter,
)
from src.monitoring import MetricsCollector  # noqa: E402
from src.partner_knowledge.retrieval import RetrievedEvidence  # noqa: E402
from src.security import SecurityPipeline  # noqa: E402

DEFAULT_AGENT_RESPONSE = "Today's special is grilled salmon with roasted vegetables."

STATE_ATTRS = ("agent", "security", "cache", "metrics", "partner_knowledge_retriever")


class FakeAgent:
    """Stand-in for `ProductionAgent`.

    The agent is the only component that reaches an external service (Gemini),
    so it is the only one replaced. Security, cache, and metrics are exercised
    for real.
    """

    def __init__(
        self,
        response: str = DEFAULT_AGENT_RESPONSE,
        model_used: str = "primary",
        raises: Exception | None = None,
    ):
        self.response = response
        self.model_used = model_used
        self.raises = raises
        self.calls: list[str] = []
        self.evidence_calls: list[list[RetrievedEvidence]] = []

    def invoke(
        self, message: str, evidence: list[RetrievedEvidence] | None = None
    ) -> dict:
        self.calls.append(message)
        if evidence is not None:
            self.evidence_calls.append(evidence)
        if self.raises is not None:
            raise self.raises
        return {
            "response": self.response,
            "model_used": self.model_used,
            "error": None,
        }


class FakePartnerKnowledgeRetriever:
    def __init__(self):
        self.evidence = [
            RetrievedEvidence(
                text="Today's special is grilled salmon with roasted vegetables.",
                document_name="Catálogo de Produtos e Ingredientes — Café Aurora",
                location="Página 1",
                technical_location="page:1",
                relevance_score=0.99,
            )
        ]
        self.queries: list[str] = []

    def ensure_available(self) -> None:
        pass

    def retrieve(self, query: str) -> list[RetrievedEvidence]:
        self.queries.append(query)
        return self.evidence


@dataclass
class Components:
    """The four objects `lifespan` would normally put on `app.state`."""

    agent: FakeAgent = field(default_factory=FakeAgent)
    security: SecurityPipeline = field(default_factory=SecurityPipeline)
    cache: ResponseCache = field(default_factory=lambda: ResponseCache(ttl_seconds=300))
    metrics: MetricsCollector = field(default_factory=MetricsCollector)
    partner_knowledge_retriever: FakePartnerKnowledgeRetriever = field(
        default_factory=FakePartnerKnowledgeRetriever
    )


def _clear_app_state() -> None:
    for attr in STATE_ATTRS:
        if hasattr(app.state, attr):
            delattr(app.state, attr)


@pytest.fixture
def components() -> Components:
    """Fresh components per test — `app` is a module-level singleton."""
    return Components()


@pytest.fixture
def agent(components: Components) -> FakeAgent:
    return components.agent


@pytest.fixture
def cache(components: Components) -> ResponseCache:
    return components.cache


@pytest.fixture
def metrics(components: Components) -> MetricsCollector:
    return components.metrics


@pytest.fixture
def partner_knowledge_retriever(
    components: Components,
) -> FakePartnerKnowledgeRetriever:
    return components.partner_knowledge_retriever


@pytest.fixture
async def client(components: Components):
    app.dependency_overrides[get_agent] = lambda: components.agent
    app.dependency_overrides[get_security] = lambda: components.security
    app.dependency_overrides[get_cache] = lambda: components.cache
    app.dependency_overrides[get_metrics] = lambda: components.metrics
    app.dependency_overrides[get_partner_knowledge_retriever] = lambda: (
        components.partner_knowledge_retriever
    )

    # `/health` reads `app.state` directly instead of going through DI, and
    # ASGITransport does not run lifespan events, so populate it by hand.
    for attr in STATE_ATTRS:
        setattr(app.state, attr, getattr(components, attr))

    # The limiter's storage is process-global and survives across tests. Off by
    # default so unrelated tests can't exhaust each other's quota.
    limiter.reset()
    limiter.enabled = False

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
    _clear_app_state()
    limiter.enabled = True
    limiter.reset()


@pytest.fixture
async def bare_client():
    """Client for an app whose components were never initialized."""
    _clear_app_state()
    limiter.reset()
    limiter.enabled = False

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    limiter.enabled = True
    limiter.reset()


@pytest.fixture
async def rate_limited_client(client: AsyncClient):
    """`client`, with rate limiting actually turned on."""
    limiter.reset()
    limiter.enabled = True
    yield client
    limiter.enabled = False
    limiter.reset()

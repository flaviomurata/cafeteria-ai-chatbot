from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

# Whitespace is stripped before the length checks run, so a blank message is a
# 422 rather than an empty prompt forwarded to the agent.
MessageStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=10000),
]


class ChatRequest(BaseModel):
    message: MessageStr = Field(
        ...,
        description="User's message to the agent",
    )
    thread_id: str = Field(
        default="default",
        description="Conversation thread ID",
    )


class SourceCitation(BaseModel):
    """Public, verification-ready reference for a grounded answer."""

    document_name: str
    location: str


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    model_used: str
    cached: bool = False
    processing_time_ms: float
    sources: list[SourceCitation] = Field(default_factory=list)
    security_notes: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class HealthResponse(BaseModel):
    status: str = "healthy"
    environment: str
    version: str = "1.1.0"
    checks: dict = {}


class MetricsResponse(BaseModel):
    total_requests: int
    total_errors: int
    error_rate: str
    avg_latency_ms: float
    cache_hit_rate: str
    total_input_tokens: int
    total_output_tokens: int


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None

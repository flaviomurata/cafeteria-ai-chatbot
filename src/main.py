from contextlib import asynccontextmanager
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from langsmith import traceable
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.agent import ProductionAgent
from src.cache import CachedChatResponse, ResponseCache
from src.config import get_settings
from src.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MetricsResponse,
    SourceCitation,
)
from src.monitoring import MetricsCollector, RequestTimer, get_logger
from src.partner_knowledge.config import get_partner_knowledge_settings
from src.partner_knowledge.constants import SCOPE_REFUSAL
from src.partner_knowledge.retrieval import (
    PartnerKnowledgeRetriever,
    PersistentChromaRetriever,
    RetrievedEvidence,
)
from src.partner_knowledge.verification import (
    EvidenceVerifier,
    MaterialClaim,
    ProductionEvidenceVerifier,
    VerificationResult,
    claim_links_are_valid,
)
from src.security import SecurityPipeline

load_dotenv()

logger = get_logger()


def get_security(request: Request) -> SecurityPipeline:
    return request.app.state.security


def get_cache(request: Request) -> ResponseCache:
    return request.app.state.cache


def get_metrics(request: Request) -> MetricsCollector:
    return request.app.state.metrics


def get_agent(request: Request) -> ProductionAgent:
    return request.app.state.agent


def get_partner_knowledge_retriever(request: Request) -> PartnerKnowledgeRetriever:
    return request.app.state.partner_knowledge_retriever


def get_evidence_verifier(request: Request) -> EvidenceVerifier:
    return request.app.state.evidence_verifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    from fastapi.concurrency import run_in_threadpool

    settings = get_settings()
    partner_knowledge_settings = get_partner_knowledge_settings()

    logger.info(
        "Starting production API...",
        extra={
            "extra_data": {
                "environment": settings.app_env,
                "primary_model": settings.primary_model,
                "tracing_enabled": settings.langchain_tracing_v2,
            }
        },
    )

    app.state.security = SecurityPipeline()
    app.state.cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    app.state.metrics = MetricsCollector()
    app.state.partner_knowledge_retriever = PersistentChromaRetriever(
        partner_knowledge_settings.partner_index_path,
        candidate_limit=partner_knowledge_settings.retrieval_candidate_limit,
        relevance_threshold=partner_knowledge_settings.relevance_threshold,
    )
    await run_in_threadpool(app.state.partner_knowledge_retriever.ensure_available)
    app.state.agent = ProductionAgent()
    app.state.evidence_verifier = ProductionEvidenceVerifier()

    logger.info("All components initialized. Ready to serve requests.")

    yield

    logger.info("Shutting down...", extra={"extra_data": app.state.metrics.summary})


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Production LangGraph API",
    description=(
        "A production-ready chat API with security, caching, and observability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning(
        "Rate limit exceeded",
        extra={
            "extra_data": {
                "client_ip": get_remote_address(request),
                "detail": str(exc),
            }
        },
    )
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": "Too many requests. Please slow down.",
        },
    )


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name="chat_endpoint")
async def chat(
    request: Request,  # noqa: ARG001 -- required by slowapi's @limiter.limit
    body: ChatRequest,
    security: Annotated[SecurityPipeline, Depends(get_security)],
    cache: Annotated[ResponseCache, Depends(get_cache)],
    metrics: Annotated[MetricsCollector, Depends(get_metrics)],
    agent: Annotated[ProductionAgent, Depends(get_agent)],
    partner_knowledge_retriever: Annotated[
        PartnerKnowledgeRetriever, Depends(get_partner_knowledge_retriever)
    ],
    evidence_verifier: Annotated[EvidenceVerifier, Depends(get_evidence_verifier)],
):
    with RequestTimer() as timer:
        security_notes = []

        # ---- Step 1: Security Check ----
        is_allowed, cleaned_message, notes = security.check_input(body.message)
        security_notes.extend(notes)

        if not is_allowed:
            logger.warning(
                "Request blocked by security",
                extra={
                    "extra_data": {
                        "reason": notes,
                        "thread_id": body.thread_id,
                    }
                },
            )
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=400,
                detail="Your message was blocked by our security filters.",
            )

        # Sanitizing strips delimiter runs, so a message that passed validation
        # can still end up empty (e.g. "---"). Don't hand that to the agent.
        if not cleaned_message:
            logger.warning(
                "Request empty after sanitization",
                extra={
                    "extra_data": {
                        "thread_id": body.thread_id,
                    }
                },
            )
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=400,
                detail="Your message was empty after sanitization.",
            )

        # ---- Step 2: Cache Lookup ----
        cached_response = cache.get(cleaned_message)
        if cached_response is not None:
            if isinstance(cached_response, CachedChatResponse):
                response_text = cached_response.response
                sources = cached_response.sources
            else:
                response_text, sources = cached_response, []
            metrics.record_request(latency_ms=0, cache_hit=True)
            logger.info(
                "Cache hit",
                extra={
                    "extra_data": {
                        "thread_id": body.thread_id,
                    }
                },
            )
            return {
                "response": response_text,
                "thread_id": body.thread_id,
                "model_used": "cache",
                "cached": True,
                "processing_time_ms": 0,
                "sources": sources,
                "security_notes": security_notes,
            }

        # ---- Step 3: Retrieve Partner knowledge, then invoke the agent ----
        evidence = await run_in_threadpool(
            partner_knowledge_retriever.retrieve, cleaned_message
        )
        if not evidence:
            return _scope_refusal(body.thread_id, security_notes, metrics)
        selected_evidence = evidence[:3]
        sources = _build_sources(selected_evidence)
        try:
            result = await run_in_threadpool(
                agent.invoke, cleaned_message, selected_evidence
            )
        except Exception as e:
            logger.error(
                f"Agent invocation failed: {e}",
                extra={
                    "extra_data": {
                        "thread_id": body.thread_id,
                        "error": str(e),
                    }
                },
            )
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing your request.",
            )

        response_text = result["response"]
        model_used = result["model_used"]
        if model_used == "error_handler":
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=503,
                detail="The grounded answer service is temporarily unavailable.",
            )
        if response_text.strip() == SCOPE_REFUSAL:
            return _scope_refusal(body.thread_id, security_notes, metrics)
        try:
            claims = [MaterialClaim.model_validate(claim) for claim in result["claims"]]
        except (KeyError, TypeError, ValueError):
            return _scope_refusal(body.thread_id, security_notes, metrics)
        if not claim_links_are_valid(claims, selected_evidence):
            return _scope_refusal(body.thread_id, security_notes, metrics)
        verification = await run_in_threadpool(
            evidence_verifier.verify, response_text, claims, selected_evidence
        )
        if not isinstance(verification, VerificationResult) or not verification.accepts(
            claims
        ):
            return _scope_refusal(body.thread_id, security_notes, metrics)

        # ---- Step 4: Output Validation ----
        validated_response, output_warnings = security.check_output(response_text)
        security_notes.extend(output_warnings)

        # ---- Step 5: Cache Store ----
        cache.set(
            cleaned_message,
            CachedChatResponse(response=validated_response, sources=sources),
        )

    # ---- Step 6: Log & Record Metrics ----
    input_tokens = int(len(cleaned_message.split()) * 1.3)
    output_tokens = int(len(validated_response.split()) * 1.3)

    metrics.record_request(
        latency_ms=timer.elapsed_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit=False,
    )

    if security_notes:
        logger.info(
            "Security notes",
            extra={
                "extra_data": {
                    "notes": security_notes,
                    "thread_id": body.thread_id,
                }
            },
        )

    logger.info(
        "Request completed",
        extra={
            "extra_data": {
                "thread_id": body.thread_id,
                "model_used": model_used,
                "latency_ms": round(timer.elapsed_ms, 2),
            }
        },
    )

    return {
        "response": validated_response,
        "thread_id": body.thread_id,
        "model_used": model_used,
        "cached": False,
        "processing_time_ms": round(timer.elapsed_ms, 2),
        "sources": sources,
        "security_notes": security_notes,
    }


def _build_sources(evidence: list[RetrievedEvidence]) -> list[SourceCitation]:
    sources: list[SourceCitation] = []
    seen: set[tuple[str, str]] = set()
    for item in evidence:
        key = (item.document_name, item.location)
        if key not in seen:
            seen.add(key)
            sources.append(
                SourceCitation(
                    document_name=item.document_name,
                    location=item.location,
                )
            )
        if len(sources) == 3:
            break
    return sources


def _scope_refusal(
    thread_id: str, security_notes: list[str], metrics: MetricsCollector
) -> dict:
    metrics.record_request(latency_ms=0, error=False)
    return {
        "response": SCOPE_REFUSAL,
        "thread_id": thread_id,
        "model_used": "grounding_refusal",
        "cached": False,
        "processing_time_ms": 0,
        "sources": [],
        "security_notes": security_notes,
    }


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    settings = get_settings()

    checks = {
        "agent": getattr(request.app.state, "agent", None) is not None,
        "security": getattr(request.app.state, "security", None) is not None,
        "cache": getattr(request.app.state, "cache", None) is not None,
    }

    all_healthy = all(checks.values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "environment": settings.app_env,
        "checks": checks,
    }


@app.get("/metrics", response_model=MetricsResponse)
async def metrics_endpoint(metrics: MetricsCollector = Depends(get_metrics)):
    return metrics.summary


@app.get("/cache/stats")
async def cache_stats(cache: ResponseCache = Depends(get_cache)):
    return cache.stats

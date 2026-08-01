"""Deterministic external-boundary adapters for the local Compose E2E mode."""

from src.partner_knowledge.retrieval import RetrievedEvidence
from src.partner_knowledge.verification import (
    MaterialClaim,
    VerificationResult,
)

_FIXTURE_EVIDENCE = RetrievedEvidence(
    text="The daily special is grilled salmon with roasted vegetables.",
    document_name="E2E fixture — Café Aurora",
    location="Fixture: daily special",
    technical_location="fixture:daily-special",
    relevance_score=1.0,
)
_FIXTURE_RESPONSE = "The E2E fixture special is grilled salmon with roasted vegetables."


class LocalPartnerKnowledgeRetriever:
    """Return one deterministic citation-ready fixture without external calls."""

    def ensure_available(self) -> None:
        pass

    def retrieve(self, _query: str) -> list[RetrievedEvidence]:
        return [_FIXTURE_EVIDENCE]


class LocalAgent:
    """Generate a deterministic grounded answer for Compose E2E checks."""

    def invoke(self, _message: str, _evidence: list[RetrievedEvidence]) -> dict:
        return {
            "response": _FIXTURE_RESPONSE,
            "claims": [
                MaterialClaim(
                    text=_FIXTURE_RESPONSE,
                    evidence_ids=["evidence-1"],
                ).model_dump()
            ],
            "model_used": "local-e2e",
            "error": None,
        }


class LocalEvidenceVerifier:
    """Accept the fixture claim while keeping the verifier API seam exercised."""

    def verify(
        self,
        _answer: str,
        claims: list[MaterialClaim],
        _evidence: list[RetrievedEvidence],
    ) -> VerificationResult:
        return VerificationResult(
            verdict="verified",
            verified_claim_indexes=list(range(len(claims))),
        )

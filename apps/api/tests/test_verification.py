from langchain_core.messages import AIMessage

from src.partner_knowledge.retrieval import RetrievedEvidence
from src.partner_knowledge.verification import (
    MaterialClaim,
    ProductionEvidenceVerifier,
)


def test_evidence_verifier_parses_json_wrapped_in_a_gemini_code_fence(monkeypatch):
    class FencedJsonLLM:
        def __init__(self, **_kwargs):
            pass

        def invoke(self, _messages):
            return AIMessage(
                content=(
                    "```json\n"
                    '{"verdict":"verified","verified_claim_indexes":[0],'
                    '"conflicting_evidence_ids":[]}\n'
                    "```"
                )
            )

    monkeypatch.setattr(
        "src.partner_knowledge.verification.ChatGoogleGenerativeAI", FencedJsonLLM
    )

    result = ProductionEvidenceVerifier().verify(
        "The special is grilled salmon.",
        [
            MaterialClaim(
                text="The special is grilled salmon.", evidence_ids=["evidence-1"]
            )
        ],
        [
            RetrievedEvidence(
                text="The special is grilled salmon.",
                document_name="E2E fixture",
                location="Fixture: special",
                technical_location="fixture:special",
                relevance_score=1.0,
            )
        ],
    )

    assert result.verdict == "verified"

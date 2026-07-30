"""Independent verification for material claims in grounded answers."""

from typing import Literal, Protocol, runtime_checkable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, ValidationError

from src.config import get_settings
from src.partner_knowledge.config import get_partner_knowledge_settings
from src.partner_knowledge.retrieval import RetrievedEvidence


class MaterialClaim(BaseModel):
    """A business statement and the retrieved evidence assigned to support it."""

    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class GeneratedGroundedAnswer(BaseModel):
    """The generation contract that is evaluated before an answer is returned."""

    answer: str = Field(min_length=1)
    claims: list[MaterialClaim] = Field(min_length=1)


class VerificationResult(BaseModel):
    """The verifier's approval for the generated material claims."""

    verdict: Literal["verified", "rejected"]
    verified_claim_indexes: list[int] = Field(default_factory=list)

    def accepts(self, claims: list[MaterialClaim]) -> bool:
        return self.verdict == "verified" and set(self.verified_claim_indexes) == set(
            range(len(claims))
        )


@runtime_checkable
class EvidenceVerifier(Protocol):
    """Replaceable boundary for independently checking answer claims."""

    def verify(
        self,
        answer: str,
        claims: list[MaterialClaim],
        evidence: list[RetrievedEvidence],
    ) -> VerificationResult:
        """Approve only claims completely supported by their linked evidence."""


class ProductionEvidenceVerifier:
    """A separately configured model invocation for evidence verification."""

    def __init__(self):
        settings = get_settings()
        partner_knowledge_settings = get_partner_knowledge_settings()
        self._llm = ChatGoogleGenerativeAI(
            model=partner_knowledge_settings.evidence_verifier_model,
            temperature=0,
            timeout=partner_knowledge_settings.evidence_verifier_timeout_seconds,
            max_retries=0,
            api_key=settings.google_api_key,
        )

    def verify(
        self,
        answer: str,
        claims: list[MaterialClaim],
        evidence: list[RetrievedEvidence],
    ) -> VerificationResult:
        evidence_by_id = "\n\n".join(
            f"{evidence_id}: {item.text}"
            for evidence_id, item in numbered_evidence(evidence)
        )
        declared_claims = "\n".join(
            f"{index}: {claim.text} (evidence: {', '.join(claim.evidence_ids)})"
            for index, claim in enumerate(claims)
        )
        response = self._llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an evidence verifier independent from the answer "
                        "generator. Assess each claim only against its declared "
                        "evidence. Approve a claim only when its linked evidence "
                        "directly supports it, and reject the answer if its claims "
                        "omit any material fact stated in the answer. "
                        "Return JSON exactly matching this schema: "
                        '{"verdict":"verified"|"rejected",'
                        '"verified_claim_indexes":[integer]}. '
                        "A verified verdict requires "
                        "every supplied claim to be approved."
                    )
                ),
                HumanMessage(
                    content=(
                        f"<answer>\n{answer}\n</answer>\n\n"
                        f"<evidence>\n{evidence_by_id}\n</evidence>\n\n"
                        f"<claims>\n{declared_claims}\n</claims>"
                    )
                ),
            ]
        )
        try:
            return VerificationResult.model_validate_json(
                _message_text(response.content)
            )
        except ValidationError as exc:
            raise ValueError("Evidence verifier returned an invalid result") from exc


def claim_links_are_valid(
    claims: list[MaterialClaim], evidence: list[RetrievedEvidence]
) -> bool:
    """Ensure every generated claim links only to selected retrieved evidence."""
    valid_ids = {evidence_id for evidence_id, _ in numbered_evidence(evidence)}
    return bool(claims) and all(
        bool(claim.evidence_ids) and set(claim.evidence_ids).issubset(valid_ids)
        for claim in claims
    )


def numbered_evidence(
    evidence: list[RetrievedEvidence],
) -> list[tuple[str, RetrievedEvidence]]:
    return [(f"evidence-{index}", item) for index, item in enumerate(evidence, start=1)]


def _message_text(content: str | list) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        block if isinstance(block, str) else block.get("text", "")
        for block in content
        if isinstance(block, (str, dict))
    )

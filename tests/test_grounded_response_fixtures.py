import json
from pathlib import Path

FIXTURE_PATH = Path(__file__).parent / "fixtures/grounded_response_cases.json"
EXPECTED_OUTCOMES = {
    "grounded",
    "scope_refusal",
    "document_conflict",
    "security_rejection",
    "grounding_service_unavailable",
}


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_grounded_response_fixture_has_unique_cases_and_approved_sources():
    fixture = _fixture()
    cases = fixture["cases"]
    approved_documents = set(fixture["approved_documents"])

    assert fixture["version"] == 1
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(case["expected_outcome"] in EXPECTED_OUTCOMES for case in cases)
    assert "FAQ" not in " ".join(fixture["approved_documents"])
    assert "Recursos Humanos" not in " ".join(fixture["approved_documents"])
    assert all(
        source["document_name"] in approved_documents
        for case in cases
        if case["execution"] == "live"
        for source in case["required_sources"]
    )


def test_every_approved_document_has_a_live_supported_case():
    fixture = _fixture()
    approved_documents = set(fixture["approved_documents"])
    covered_documents = {
        source["document_name"]
        for case in fixture["cases"]
        if case["execution"] == "live" and case["expected_outcome"] == "grounded"
        for source in case["required_sources"]
    }

    assert covered_documents == approved_documents

"""Opt-in live-provider evaluation against a staging HTTP deployment."""

import asyncio
import json
import os
import unicodedata
from pathlib import Path

import httpx
import pytest

if os.getenv("RUN_LIVE_EVAL") != "1":
    pytest.skip(
        "Live grounding evaluation requires RUN_LIVE_EVAL=1",
        allow_module_level=True,
    )


BASE_URL = os.getenv("E2E_BASE_URL")
EXPECTED_ENVIRONMENT = os.getenv("E2E_EXPECTED_ENVIRONMENT", "staging")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("E2E_TIMEOUT_SECONDS", "60"))
SCOPE_REFUSAL = (
    "Só posso responder a perguntas apoiadas pelo conhecimento dos "
    "Parceiros do Café Aurora."
)
FIXTURE_PATH = Path(__file__).parents[1] / "fixtures/grounded_response_cases.json"
CASES = [
    case
    for case in json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]
    if case["execution"] == "live"
]


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(
        character for character in decomposed if not unicodedata.combining(character)
    ).casefold()


@pytest.fixture
async def staging_client():
    if not BASE_URL:
        pytest.fail("RUN_LIVE_EVAL=1 requires E2E_BASE_URL")
    if EXPECTED_ENVIRONMENT != "staging":
        pytest.fail(
            "Live grounding evaluation must target E2E_EXPECTED_ENVIRONMENT=staging"
        )

    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=timeout) as client:
        last_error = "staging service did not become ready"
        for _ in range(60):
            try:
                response = await client.get("/health")
                if response.status_code == 200:
                    health = response.json()
                    if health.get("environment") != EXPECTED_ENVIRONMENT:
                        pytest.fail(
                            "Live grounding evaluation reached an unexpected "
                            "environment: "
                            f"{health.get('environment')!r}"
                        )
                    break
                last_error = f"health returned HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            await asyncio.sleep(1)
        else:
            pytest.fail(last_error)
        yield client


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
async def test_live_grounding_case(case: dict, staging_client: httpx.AsyncClient):
    response = await staging_client.post("/chat", json={"message": case["question"]})
    body = response.json()
    outcome = case["expected_outcome"]

    if outcome == "grounded":
        assert response.status_code == 200
        assert body["model_used"] not in {
            "cache",
            "grounding_refusal",
            "grounding_conflict",
        }
        public_sources = body["sources"]
        assert all(source in public_sources for source in case["required_sources"])
        normalized_response = _normalize(body["response"])
        assert all(
            _normalize(fact) in normalized_response for fact in case["required_facts"]
        )
        assert all(
            _normalize(fact) not in normalized_response
            for fact in case["forbidden_facts"]
        )
    elif outcome == "scope_refusal":
        assert response.status_code == 200
        assert body["response"] == SCOPE_REFUSAL
        assert body["model_used"] == "grounding_refusal"
        assert body["sources"] == []
    elif outcome == "security_rejection":
        assert response.status_code == 400
    else:
        raise AssertionError(f"Unsupported live evaluation outcome: {outcome}")

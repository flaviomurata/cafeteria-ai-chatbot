import pytest
from langchain_core.messages import AIMessage

from src.agent import ProductionAgent
from src.partner_knowledge.retrieval import RetrievedEvidence
from src.provider_errors import ProviderRateLimitError


def test_agent_parses_json_wrapped_in_a_gemini_code_fence(monkeypatch):
    class FencedJsonLLM:
        def __init__(self, **_kwargs):
            pass

        def invoke(self, _messages):
            return AIMessage(
                content=(
                    "```json\n"
                    '{"answer":"The special is grilled salmon.",'
                    '"claims":[{"text":"The special is grilled salmon.",'
                    '"evidence_ids":["evidence-1"]}]}\n'
                    "```"
                )
            )

    monkeypatch.setattr("src.agent.ChatGoogleGenerativeAI", FencedJsonLLM)

    result = ProductionAgent().invoke("What is for lunch?", [])

    assert result["response"] == "The special is grilled salmon."
    assert result["claims"] == [
        {
            "text": "The special is grilled salmon.",
            "evidence_ids": ["evidence-1"],
        }
    ]


@pytest.mark.parametrize("opening_fence", ["```JSON", "```"])
def test_agent_parses_json_with_other_supported_fence_headers(
    monkeypatch, opening_fence
):
    class FencedJsonLLM:
        def __init__(self, **_kwargs):
            pass

        def invoke(self, _messages):
            return AIMessage(
                content=(
                    f"{opening_fence}\n"
                    '{"answer":"The special is grilled salmon.",'
                    '"claims":[{"text":"The special is grilled salmon.",'
                    '"evidence_ids":["evidence-1"]}]}\n'
                    "```"
                )
            )

    monkeypatch.setattr("src.agent.ChatGoogleGenerativeAI", FencedJsonLLM)

    result = ProductionAgent().invoke("What is for lunch?", [])

    assert result["response"] == "The special is grilled salmon."


class RaisingLLM:
    def __init__(self, error: Exception | None = None):
        self.calls = 0
        self.error = error or ProviderRateLimitError("Gemini", retry_after_seconds=12)

    def invoke(self, _messages):
        self.calls += 1
        raise self.error


def test_agent_does_not_call_fallback_after_a_provider_rate_limit(monkeypatch):
    llms: list[RaisingLLM] = []

    def build_llm(**_kwargs):
        llm = RaisingLLM()
        llms.append(llm)
        return llm

    monkeypatch.setattr("src.agent.ChatGoogleGenerativeAI", build_llm)

    agent = ProductionAgent()
    with pytest.raises(ProviderRateLimitError) as error:
        agent.invoke(
            "What is for lunch?",
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

    assert len(llms) == 2
    assert llms[0].calls == 1
    assert llms[1].calls == 0
    assert error.value.retry_after_seconds == 12


def test_fallback_provider_quota_is_returned_to_the_api(monkeypatch):
    llms: list[RaisingLLM] = []
    errors = [
        RuntimeError("primary unavailable"),
        ProviderRateLimitError("Gemini", retry_after_seconds=9),
    ]

    def build_llm(**_kwargs):
        llm = RaisingLLM(errors.pop(0))
        llms.append(llm)
        return llm

    monkeypatch.setattr("src.agent.ChatGoogleGenerativeAI", build_llm)

    agent = ProductionAgent()
    with pytest.raises(ProviderRateLimitError) as error:
        agent.invoke("What is for lunch?", [])

    assert [llm.calls for llm in llms] == [1, 1]
    assert error.value.retry_after_seconds == 9


def test_raw_provider_quota_message_preserves_retry_interval(monkeypatch):
    class RawRateLimitError(RuntimeError):
        pass

    def build_llm(**_kwargs):
        return RaisingLLM(RawRateLimitError("429 RESOURCE_EXHAUSTED; retry in 8.2s"))

    monkeypatch.setattr("src.agent.ChatGoogleGenerativeAI", build_llm)

    with pytest.raises(ProviderRateLimitError) as error:
        ProductionAgent().invoke("What is for lunch?", [])

    assert error.value.retry_after_seconds == 9

from types import SimpleNamespace

import pytest

from agent.llm import ClaudeLLM, LLMError, strict_schema
from agent.schemas import Challenge, ClaimJudgment, ClaimStatus, ParsedThesis, SourceRef


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


@pytest.mark.parametrize("model", [ParsedThesis, ClaimJudgment, Challenge, SourceRef])
def test_strict_schema_is_strict(model):
    schema = strict_schema(model)
    assert "$ref" not in str(schema) and "$defs" not in schema
    for node in _walk(schema):
        if node.get("type") == "object" and "properties" in node:
            assert node["additionalProperties"] is False
            assert node["required"] == list(node["properties"])


def test_walker_keeps_every_property():
    assert set(strict_schema(SourceRef)["properties"]) == set(SourceRef.model_fields)


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.kwargs = None
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _resp(stop_reason, text=""):
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber"),
        content=[SimpleNamespace(type="text", text=text)],
    )


def test_claude_llm_parses_and_sends_fallbacks():
    payload = ClaimJudgment(status=ClaimStatus.SUPPORTED, reasoning="ok", evidence=[], follow_up_query=None)
    client = FakeClient(_resp("end_turn", payload.model_dump_json()))
    out = ClaudeLLM(client=client, model="m").structured(system="s", user="u", output=ClaimJudgment)
    assert out == payload
    assert client.kwargs["fallbacks"] == "default"
    assert client.kwargs["output_config"]["format"]["type"] == "json_schema"


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_claude_llm_raises_on_bad_stop(stop):
    with pytest.raises(LLMError):
        ClaudeLLM(client=FakeClient(_resp(stop)), model="m").structured(system="s", user="u", output=ClaimJudgment)


def test_claude_llm_raises_on_invalid_json():
    with pytest.raises(LLMError):
        ClaudeLLM(client=FakeClient(_resp("end_turn", '{"status": "maybe"}')), model="m").structured(
            system="s", user="u", output=ClaimJudgment
        )

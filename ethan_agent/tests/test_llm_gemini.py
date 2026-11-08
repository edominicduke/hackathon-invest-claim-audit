"""GeminiLLM against a fake client: request shape, effort mapping, block handling."""

from types import SimpleNamespace

import pytest

pytest.importorskip("google.genai")

import agent  # noqa: E402
from agent.llm import LLMError  # noqa: E402
from agent.llm_gemini import GeminiLLM  # noqa: E402
from agent.schemas import ClaimJudgment, ClaimStatus  # noqa: E402

PAYLOAD = ClaimJudgment(status=ClaimStatus.SUPPORTED, reasoning="ok [S1].", evidence=[], follow_up_query=None)


class FakeModels:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def resp(text=PAYLOAD.model_dump_json(), finish="STOP", block=None, candidates=True):
    return SimpleNamespace(
        text=text,
        prompt_feedback=SimpleNamespace(block_reason=block) if block else None,
        candidates=[SimpleNamespace(finish_reason=finish)] if candidates else [],
    )


def make(response, model="gemini-3.8-flash"):
    models = FakeModels(response)
    return GeminiLLM(model=model, client=SimpleNamespace(models=models)), models


def test_parses_and_sends_schema_and_system_prompt():
    llm, models = make(resp())
    assert llm.structured(system="SYS", user="USER", output=ClaimJudgment, effort="high") == PAYLOAD
    cfg = models.kwargs["config"]
    assert models.kwargs["contents"] == "USER" and cfg.system_instruction == "SYS"
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema["additionalProperties"] is False
    assert cfg.thinking_config.thinking_level.value == "HIGH"


def test_gemini_2_uses_thinking_budget():
    llm, models = make(resp(), model="gemini-2.5-flash")
    llm.structured(system="s", user="u", output=ClaimJudgment, effort="low")
    tc = models.kwargs["config"].thinking_config
    assert tc.thinking_budget == 1024 and tc.thinking_level is None


@pytest.mark.parametrize(
    "response",
    [resp(block="SAFETY"), resp(finish="SAFETY"), resp(finish="PROHIBITED_CONTENT"),
     resp(finish="MAX_TOKENS"), resp(candidates=False), resp(text='{"status": "maybe"}')],
)
def test_blocks_and_bad_output_raise_llm_error(response):
    llm, _ = make(response)
    with pytest.raises(LLMError):
        llm.structured(system="s", user="u", output=ClaimJudgment)


def test_provider_switch(monkeypatch):
    monkeypatch.setenv("AUDITOR_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    assert isinstance(agent.make_llm(), GeminiLLM)
    with pytest.raises(ValueError):
        agent.make_llm("openai")


# ------------------------------------------------------------ fast mode + retries

from google.genai import errors as genai_errors  # noqa: E402


class FlakyModels:
    """Raises the given errors in order, then returns ``response``."""

    def __init__(self, failures, response):
        self.failures = list(failures)
        self.response = response
        self.calls = 0

    def generate_content(self, **_kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return self.response


def err(cls, code, status):
    return cls(code, {"error": {"code": code, "message": "m", "status": status}})


def flaky(failures, attempts=5):
    models = FlakyModels(failures, resp())
    sleeps = []
    llm = GeminiLLM(model="gemini-3.8-flash", client=SimpleNamespace(models=models),
                    attempts=attempts, sleep=sleeps.append)
    return llm, models, sleeps


@pytest.mark.parametrize("effort,normal,fast", [("high", "HIGH", "MEDIUM"), ("medium", "MEDIUM", "LOW"), ("low", "LOW", "LOW")])
def test_fast_mode_lowers_thinking_one_step(effort, normal, fast):
    for is_fast, expected in ((False, normal), (True, fast)):
        models = FakeModels(resp())
        GeminiLLM(model="gemini-3.8-flash", client=SimpleNamespace(models=models), fast=is_fast).structured(
            system="s", user="u", output=ClaimJudgment, effort=effort)
        assert models.kwargs["config"].thinking_config.thinking_level.value == expected


def test_fast_mode_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_FAST", "1")
    assert GeminiLLM(model="m", client=SimpleNamespace(models=None)).fast is True


def test_retries_rate_limit_and_deadline_with_backoff_and_logs(caplog):
    llm, models, sleeps = flaky([err(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED"),
                                 err(genai_errors.ServerError, 504, "DEADLINE_EXCEEDED")])
    with caplog.at_level("INFO", logger="agent.llm"):
        assert llm.structured(system="s", user="u", output=ClaimJudgment) == PAYLOAD
    assert models.calls == 3 and sleeps == [2.0]  # 429 backs off; 504 retries at once, lighter
    text = caplog.text
    assert "429 RESOURCE_EXHAUSTED" in text and "retrying in 2s" in text
    assert "504 DEADLINE_EXCEEDED" in text and "retrying once with thinking MEDIUM -> LOW" in text
    assert "ClaimJudgment" in text and "thinking=MEDIUM" in text  # the success line


def test_gives_up_after_max_attempts():
    llm, models, sleeps = flaky([err(genai_errors.ServerError, 503, "UNAVAILABLE")] * 3, attempts=3)
    with pytest.raises(LLMError, match="after 3 attempts"):
        llm.structured(system="s", user="u", output=ClaimJudgment)
    assert models.calls == 3 and len(sleeps) == 2


def test_client_errors_are_not_retried():
    llm, models, sleeps = flaky([err(genai_errors.ClientError, 400, "INVALID_ARGUMENT")])
    with pytest.raises(LLMError, match="400"):
        llm.structured(system="s", user="u", output=ClaimJudgment)
    assert models.calls == 1 and not sleeps


def test_client_timeout_is_retried():
    class ReadTimeout(Exception):
        pass

    llm, models, _ = flaky([ReadTimeout("timed out")])
    assert llm.structured(system="s", user="u", output=ClaimJudgment) == PAYLOAD
    assert models.calls == 2


@pytest.mark.parametrize("fast,env,expected", [("", None, 4), ("1", None, 1), ("1", "3", 3)])
def test_worker_default_follows_fast_mode(monkeypatch, fast, env, expected):
    monkeypatch.setenv("AUDITOR_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_FAST", fast)
    monkeypatch.setenv("AUDITOR_TEAM_COMPONENTS", "0")  # don't touch the real track-record folder
    if env:
        monkeypatch.setenv("AUDITOR_MAX_WORKERS", env)
    else:
        monkeypatch.delenv("AUDITOR_MAX_WORKERS", raising=False)
    assert agent.build_default_auditor().max_workers == expected


# ------------------------------------------------------------ network errors + call logs

def test_dns_failure_is_retried_and_logged(caplog):
    class ConnectError(Exception):  # same class name httpx raises for "getaddrinfo failed"
        pass

    llm, models, sleeps = flaky([ConnectError("[Errno 11001] getaddrinfo failed")])
    with caplog.at_level("INFO", logger="agent.llm"):
        assert llm.structured(system="s", user="u", output=ClaimJudgment) == PAYLOAD
    assert models.calls == 2 and sleeps == [2.0]
    assert "network error (ConnectError)" in caplog.text
    assert "-> ClaimJudgment" in caplog.text and "<- ClaimJudgment" in caplog.text


def test_final_failure_is_logged(caplog):
    llm, _, _ = flaky([err(genai_errors.ClientError, 400, "INVALID_ARGUMENT")])
    with caplog.at_level("INFO", logger="agent.llm"), pytest.raises(LLMError):
        llm.structured(system="s", user="u", output=ClaimJudgment)
    assert "x  ClaimJudgment" in caplog.text and "FAILED" in caplog.text


def test_heartbeat_while_waiting(caplog):
    import time

    from agent.llm import CallLog

    with caplog.at_level("INFO", logger="agent.llm"):
        with CallLog("Slow", "test", heartbeat_s=0.05):
            time.sleep(0.18)
    assert caplog.text.count("still waiting") >= 2


# ------------------------------------------------------------ slow requests

def test_deadline_retried_once_with_less_thinking(caplog):
    llm, models, sleeps = flaky([err(genai_errors.ServerError, 504, "DEADLINE_EXCEEDED")])
    seen = []
    original = models.generate_content
    models.generate_content = lambda **kw: (seen.append(kw["config"].thinking_config.thinking_level.value), original(**kw))[1]
    with caplog.at_level("INFO", logger="agent.llm"):
        assert llm.structured(system="s", user="u", output=ClaimJudgment, effort="high") == PAYLOAD
    assert seen == ["HIGH", "MEDIUM"] and not sleeps  # no backoff wait for slowness
    assert "retrying once with thinking HIGH -> MEDIUM" in caplog.text


def test_second_deadline_gives_up_instead_of_looping():
    llm, models, _ = flaky([err(genai_errors.ServerError, 504, "DEADLINE_EXCEEDED")] * 4)
    with pytest.raises(LLMError, match="too slow"):
        llm.structured(system="s", user="u", output=ClaimJudgment, effort="high")
    assert models.calls == 2  # not 5 x 120s


# ------------------------------------------------------------ thinking cap + quotas

@pytest.mark.parametrize("cap,effort,expected", [("LOW", "high", "LOW"), ("LOW", "low", "LOW"),
                                                 ("MEDIUM", "high", "MEDIUM"), ("MEDIUM", "low", "LOW")])
def test_max_thinking_cap(monkeypatch, cap, effort, expected):
    monkeypatch.setenv("GEMINI_MAX_THINKING", cap)
    models = FakeModels(resp())
    GeminiLLM(model="gemini-3.8-flash", client=SimpleNamespace(models=models)).structured(
        system="s", user="u", output=ClaimJudgment, effort=effort)
    assert models.kwargs["config"].thinking_config.thinking_level.value == expected


def test_max_thinking_cap_on_gemini_2_budget(monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_THINKING", "low")
    models = FakeModels(resp())
    GeminiLLM(model="gemini-2.5-flash", client=SimpleNamespace(models=models)).structured(
        system="s", user="u", output=ClaimJudgment, effort="high")
    assert models.kwargs["config"].thinking_config.thinking_budget == 512


def test_bad_cap_rejected(monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_THINKING", "tiny")
    with pytest.raises(ValueError):
        GeminiLLM(model="m", client=SimpleNamespace(models=None))


def quota_429(quota_id, retry="17s"):
    return genai_errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "m", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota_id}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry},
    ]}})


def test_per_minute_quota_waits_server_suggested_delay(caplog):
    llm, models, sleeps = flaky([quota_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")])
    with caplog.at_level("INFO", logger="agent.llm"):
        assert llm.structured(system="s", user="u", output=ClaimJudgment) == PAYLOAD
    assert sleeps == [17.0] and "PerMinute" in caplog.text


def test_daily_quota_fails_fast_with_advice():
    llm, models, sleeps = flaky([quota_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier")])
    with pytest.raises(LLMError, match="daily quota.*GEMINI_MODEL"):
        llm.structured(system="s", user="u", output=ClaimJudgment)
    assert models.calls == 1 and not sleeps


# ------------------------------------------------------------ automatic model switching

import agent.llm_gemini as gem  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_exhausted(monkeypatch):
    monkeypatch.setattr(gem, "_EXHAUSTED", set())
    # The app defaults to LOW for every model (tested below). Most tests here check the
    # effort -> thinking mapping itself, so they run with the cap lifted.
    monkeypatch.setenv("GEMINI_MAX_THINKING", "HIGH")
    monkeypatch.delenv("GEMINI_FALLBACK_MODELS", raising=False)
    monkeypatch.delenv("GEMINI_AUTO_SWITCH", raising=False)


DAILY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"


class QuotaModels:
    """Each model name in ``exhausted`` returns a daily-quota 429; others succeed."""

    def __init__(self, exhausted=(), listed=()):
        self.exhausted = set(exhausted)
        self.listed = listed
        self.used = []

    def generate_content(self, *, model, config, **_):
        self.used.append((model, config.thinking_config))
        if model in self.exhausted:
            raise quota_429(DAILY, retry="23666s")
        return resp()

    def list(self):
        return [SimpleNamespace(name=f"models/{n}", supported_actions=a) for n, a in self.listed]


def switching_llm(models, model="gemini-3.8-flash"):
    return GeminiLLM(model=model, client=SimpleNamespace(models=models), sleep=lambda s: None)


def test_daily_quota_switches_to_fallback_and_stays_there(monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash, gemini-3.5-flash-lite")
    models = QuotaModels(exhausted={"gemini-3.8-flash"})
    llm = switching_llm(models)
    with caplog.at_level("INFO", logger="agent.llm"):
        assert llm.structured(system="s", user="u", output=ClaimJudgment) == PAYLOAD
        llm.structured(system="s", user="u", output=ClaimJudgment)
    assert [m for m, _ in models.used] == ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash"]
    assert llm.model == "gemini-3.5-flash"
    assert "switching to gemini-3.5-flash" in caplog.text


def test_walks_the_chain_until_a_model_has_quota(monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash,gemini-3.5-flash-lite")
    models = QuotaModels(exhausted={"gemini-3.8-flash", "gemini-3.5-flash"})
    llm = switching_llm(models)
    llm.structured(system="s", user="u", output=ClaimJudgment)
    assert llm.model == "gemini-3.5-flash-lite"


def test_all_exhausted_raises_clear_error(monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash")
    llm = switching_llm(QuotaModels(exhausted={"gemini-3.8-flash", "gemini-3.5-flash"}))
    with pytest.raises(LLMError, match="No fallback model left"):
        llm.structured(system="s", user="u", output=ClaimJudgment)


def test_auto_switch_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash")
    monkeypatch.setenv("GEMINI_AUTO_SWITCH", "0")
    models = QuotaModels(exhausted={"gemini-3.8-flash"})
    with pytest.raises(LLMError, match="daily quota"):
        switching_llm(models).structured(system="s", user="u", output=ClaimJudgment)
    assert len(models.used) == 1


def test_404_model_also_switches(monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash")
    llm, models, _ = flaky([err(genai_errors.ClientError, 404, "NOT_FOUND")])
    llm.structured(system="s", user="u", output=ClaimJudgment)
    assert llm.model == "gemini-3.5-flash" and models.calls == 2


def test_discovery_keeps_text_flash_models_newest_first():
    gc = ["generateContent"]
    models = QuotaModels(listed=[
        ("gemini-2.5-flash", gc), ("gemini-3.5-flash-lite", gc), ("gemini-3.5-flash", gc),
        ("gemini-3.8-flash-tts", gc), ("gemini-3.1-pro-preview", gc), ("gemini-3.1-flash-image", gc),
        ("text-embedding-004", ["embedContent"]), ("gemini-3.8-flash", gc),
    ])
    llm = switching_llm(models)
    assert llm.fallback_models() == ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash"]


def test_switch_to_gemini_2_recomputes_thinking(monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash")
    models = QuotaModels(exhausted={"gemini-3.8-flash"})
    switching_llm(models).structured(system="s", user="u", output=ClaimJudgment, effort="low")
    first, second = models.used[0][1], models.used[1][1]
    assert first.thinking_level is not None and second.thinking_budget == 1024 and second.thinking_level is None


def test_parallel_threads_dont_skip_a_good_model(monkeypatch):
    # Thread A already switched 3.8 -> 3.5. Thread B then reports 3.8 exhausted:
    # it must adopt 3.5, not mark 3.5 exhausted too.
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash,gemini-3.5-flash-lite")
    llm = switching_llm(QuotaModels())
    assert llm._switch_from("gemini-3.8-flash", "daily quota") and llm.model == "gemini-3.5-flash"
    assert llm._switch_from("gemini-3.8-flash", "daily quota") and llm.model == "gemini-3.5-flash"
    assert "gemini-3.5-flash" not in gem._EXHAUSTED



# ------------------------------------------------------------ default: LOW everywhere

@pytest.mark.parametrize("model", ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-pro-preview"])
@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_default_is_low_for_every_model_and_effort(monkeypatch, model, effort):
    monkeypatch.delenv("GEMINI_MAX_THINKING")
    models = FakeModels(resp())
    GeminiLLM(model=model, client=SimpleNamespace(models=models), fast=False).structured(
        system="s", user="u", output=ClaimJudgment, effort=effort)
    assert models.kwargs["config"].thinking_config.thinking_level.value == "LOW"


def test_default_low_on_gemini_2_budget(monkeypatch):
    monkeypatch.delenv("GEMINI_MAX_THINKING")
    models = FakeModels(resp())
    GeminiLLM(model="gemini-2.5-flash", client=SimpleNamespace(models=models)).structured(
        system="s", user="u", output=ClaimJudgment, effort="high")
    assert models.kwargs["config"].thinking_config.thinking_budget == 512


def test_default_low_survives_auto_switch(monkeypatch):
    monkeypatch.delenv("GEMINI_MAX_THINKING")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash,gemini-2.5-flash")
    models = QuotaModels(exhausted={"gemini-3.8-flash", "gemini-3.5-flash"})
    switching_llm(models).structured(system="s", user="u", output=ClaimJudgment, effort="high")
    levels = [(tc.thinking_level.value if tc.thinking_level else tc.thinking_budget) for _, tc in models.used]
    assert levels == ["LOW", "LOW", 512]


def test_slow_call_at_low_retries_with_same_settings(monkeypatch, caplog):
    monkeypatch.delenv("GEMINI_MAX_THINKING")
    llm, models, _ = flaky([err(genai_errors.ServerError, 504, "DEADLINE_EXCEEDED")])
    with caplog.at_level("INFO", logger="agent.llm"):
        llm.structured(system="s", user="u", output=ClaimJudgment, effort="high")
    assert models.calls == 2 and "retrying once with same settings" in caplog.text

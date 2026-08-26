from types import SimpleNamespace

import pytest
from langchain_openai.chat_models.base import OpenAIRateLimitError

from app.backoff import call_with_backoff


def _rate_limit():
    request = SimpleNamespace(
        method="POST",
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={},
    )
    response = SimpleNamespace(status_code=429, headers={}, request=request)
    return OpenAIRateLimitError(
        "Error code: 429 - rate limited",
        response=response,
        body={"error": {"code": 429}},
    )


def test_succeeds_after_transient_rate_limits(monkeypatch):
    monkeypatch.setattr("app.backoff.time.sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _rate_limit()
        return "ok"

    assert call_with_backoff(flaky, attempts=4, base_delay=0.0) == "ok"
    assert calls["n"] == 3


def test_raises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr("app.backoff.time.sleep", lambda s: None)

    def always_limited():
        raise _rate_limit()

    with pytest.raises(OpenAIRateLimitError):
        call_with_backoff(always_limited, attempts=3, base_delay=0.0)


def test_non_rate_limit_errors_raise_immediately():
    def broken():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        call_with_backoff(broken)


def test_passes_args_through():
    result = call_with_backoff(lambda a, b: a + b, 2, b=40)
    assert result == 42


def test_retry_on_predicate(monkeypatch):
    monkeypatch.setattr("app.backoff.time.sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("429 too many requests")
        return "ok"

    result = call_with_backoff(
        flaky,
        attempts=3,
        base_delay=0.0,
        retry_on=lambda exc: "429" in str(exc),
    )
    assert result == "ok"
    assert calls["n"] == 2


def test_retry_on_predicate_declines(monkeypatch):
    monkeypatch.setattr("app.backoff.time.sleep", lambda s: None)

    def broken():
        raise ValueError("404 not found")

    with pytest.raises(ValueError):
        call_with_backoff(
            broken, attempts=3, base_delay=0.0, retry_on=lambda exc: "429" in str(exc)
        )


def test_retry_on_tuple_of_types(monkeypatch):
    monkeypatch.setattr("app.backoff.time.sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise KeyError("transient")
        return "ok"

    assert call_with_backoff(flaky, attempts=3, base_delay=0.0, retry_on=(KeyError,)) == "ok"


def test_retry_on_bare_exception_class_is_treated_as_a_type(monkeypatch):
    monkeypatch.setattr("app.backoff.time.sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise KeyError("transient")
        return "ok"

    assert call_with_backoff(flaky, attempts=3, base_delay=0.0, retry_on=KeyError) == "ok"
    assert calls["n"] == 2


def test_bare_exception_class_does_not_retry_other_types(monkeypatch):
    monkeypatch.setattr("app.backoff.time.sleep", lambda s: None)
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        raise RuntimeError("not a KeyError")

    with pytest.raises(RuntimeError):
        call_with_backoff(broken, attempts=3, base_delay=0.0, retry_on=KeyError)
    assert calls["n"] == 1, "must not retry an exception the class does not cover"

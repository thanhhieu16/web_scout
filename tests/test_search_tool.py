import httpx
from langchain_core.messages import AIMessage, ToolMessage

from app.config import Settings
from app.nodes.parsing import collect_search_tool_sources
from app.tools.cache import TTLCache
from app.tools.search_tool import make_web_search


def _ok_response_body():
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "results",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": "https://x.dev",
                                "title": "X Title",
                                "content": "useful x content",
                            },
                        },
                        {
                            "type": "url_citation",
                            "url_citation": {"url": "https://x.dev", "title": "dup"},
                        },
                    ],
                }
            }
        ],
        "usage": {"server_tool_use_details": {"web_search_requests": 1}},
    }


def _empty_ddg_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text="<html>no results</html>")


# Passed to every test that expects the *primary* path to fail without a
# usable fallback, so the offline suite never falls through to a real
# network call against DuckDuckGo.
_no_fallback = httpx.MockTransport(_empty_ddg_handler)


def test_web_search_formats_annotations():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response_body())

    tool = make_web_search(Settings(_env_file=None), transport=httpx.MockTransport(handler))  # type: ignore[call-arg]
    out = tool.invoke({"query": "test query"})
    assert out.startswith("SEARCH_RESULTS (1 results, 1 search executed)")
    assert "[SRC] https://x.dev | X Title" in out
    assert "EXCERPT: useful x content" in out


def test_web_search_http_error_returns_search_error(monkeypatch):
    monkeypatch.setattr("app.backoff.time.sleep", lambda s: None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    tool = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(handler),
        fallback_transport=_no_fallback,
    )
    out = tool.invoke({"query": "q"})
    assert out.startswith("SEARCH_ERROR:")


def test_web_search_malformed_body_returns_search_error_not_raise():
    """A 200 with an HTML interstitial (proxy/Cloudflare block page) must not escape as
    a json.JSONDecodeError — response parsing must sit inside the same error boundary as
    the HTTP call, exactly like web_fetch's FETCH_ERROR."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>blocked</html>")

    tool = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(handler),
        fallback_transport=_no_fallback,
    )
    out = tool.invoke({"query": "q"})
    assert out.startswith("SEARCH_ERROR:")


def test_web_search_no_results_returns_search_error():
    body = {"choices": [{"message": {"role": "assistant", "content": "nothing"}}], "usage": {}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    tool = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(handler),
        fallback_transport=_no_fallback,
    )
    out = tool.invoke({"query": "q"})
    assert out == "SEARCH_ERROR: no results returned"


def test_collect_search_tool_sources_pairs_and_parses():
    content = (
        "SEARCH_RESULTS (2 results, 1 search executed):\n\n"
        "[SRC] https://a.dev | A Title\nEXCERPT: alpha excerpt\n\n"
        "[SRC] https://b.dev | B Title\nEXCERPT: beta excerpt"
    )
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "q"}, "id": "s1"}],
        ),
        ToolMessage(content=content, tool_call_id="s1"),
        AIMessage(content="## FINDINGS\n- [S1] claim | confidence: high\n"),
    ]
    sources = collect_search_tool_sources(messages)
    assert [s["url"] for s in sources] == ["https://a.dev", "https://b.dev"]
    assert sources[0]["title"] == "A Title"
    assert sources[0]["excerpt"] == "alpha excerpt"


def test_collect_search_tool_sources_ignores_unrelated_tools():
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "web_fetch", "args": {"url": "https://z"}, "id": "f1"}],
        ),
        ToolMessage(content="[SRC] https://noisy.dev | N\nEXCERPT: x", tool_call_id="f1"),
    ]
    assert collect_search_tool_sources(messages) == []


def test_web_search_records_usage():
    from app.usage import UsageCollector

    body = _ok_response_body()
    body["usage"]["total_tokens"] = 900
    body["usage"]["cost"] = 0.0042

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    collector = UsageCollector()
    tool = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(handler),
        usage=collector,
    )
    tool.invoke({"query": "q"})
    tokens, cost, searches = collector.drain()
    assert tokens == 900
    assert abs(cost - 0.0042) < 1e-9
    assert searches == 1


def test_web_search_records_usage_even_with_no_results():
    from app.usage import UsageCollector

    body = {
        "choices": [{"message": {"role": "assistant", "content": "nothing"}}],
        "usage": {"total_tokens": 120, "cost": 0.0008},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    collector = UsageCollector()
    tool = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(handler),
        fallback_transport=_no_fallback,
        usage=collector,
    )
    out = tool.invoke({"query": "q"})
    assert out == "SEARCH_ERROR: no results returned"
    tokens, cost, _ = collector.drain()
    assert tokens == 120
    assert abs(cost - 0.0008) < 1e-9


def test_web_search_body_requests_usage_accounting():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_ok_response_body())

    tool = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(handler),
    )
    tool.invoke({"query": "q"})
    assert captured["usage"] == {"include": True}


_DDG_HREF = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&amp;rut=abc"
_DDG_HTML = f"""
<div class="result results_links results_links_deep web-result">
  <div class="result__body">
    <h2 class="result__title">
      <a rel="nofollow" href="{_DDG_HREF}" class="result__a">Example Title</a>
    </h2>
    <a class="result__snippet" href="{_DDG_HREF}">Example snippet text here.</a>
  </div>
</div>
"""


def _failing_primary_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, text="boom")


def _ddg_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text=_DDG_HTML)


def test_web_search_falls_back_to_duckduckgo_on_hard_failure(monkeypatch):
    monkeypatch.setattr("app.backoff.time.sleep", lambda s: None)

    tool = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(_failing_primary_handler),
        fallback_transport=httpx.MockTransport(_ddg_handler),
    )
    out = tool.invoke({"query": "q"})
    assert out.startswith("SEARCH_RESULTS")
    assert "[SRC] https://example.com/page | Example Title" in out
    assert "EXCERPT: Example snippet text here." in out


def test_web_search_falls_back_to_duckduckgo_on_no_results():
    body = {"choices": [{"message": {"role": "assistant", "content": "nothing"}}], "usage": {}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    tool = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(handler),
        fallback_transport=httpx.MockTransport(_ddg_handler),
    )
    out = tool.invoke({"query": "q"})
    assert out.startswith("SEARCH_RESULTS")
    assert "[SRC] https://example.com/page | Example Title" in out


def test_web_search_returns_original_error_when_fallback_also_fails(monkeypatch):
    monkeypatch.setattr("app.backoff.time.sleep", lambda s: None)

    tool = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(_failing_primary_handler),
        fallback_transport=httpx.MockTransport(_empty_ddg_handler),
    )
    out = tool.invoke({"query": "q"})
    assert out.startswith("SEARCH_ERROR:")
    assert "HTTPStatusError" in out


def test_web_search_does_not_call_fallback_when_primary_succeeds():
    def fallback_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("fallback should not be called when primary succeeds")

    tool = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_ok_response_body())),
        fallback_transport=httpx.MockTransport(fallback_handler),
    )
    out = tool.invoke({"query": "q"})
    assert out.startswith("SEARCH_RESULTS")
    assert "x.dev" in out


def test_web_search_fallback_records_usage_as_one_search():
    from app.usage import UsageCollector

    collector = UsageCollector()
    tool = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(_failing_primary_handler),
        fallback_transport=httpx.MockTransport(_ddg_handler),
        usage=collector,
    )
    tool.invoke({"query": "q"})
    _, _, searches = collector.drain()
    assert searches == 1


def test_web_search_serves_repeat_query_from_cache_without_second_call():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_ok_response_body())

    cache = TTLCache()
    tool = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(handler),
        cache=cache,
    )
    first = tool.invoke({"query": "  Test Query  "})
    second = tool.invoke({"query": "test query"})
    assert first == second
    assert calls["n"] == 1


def test_web_search_does_not_cache_errors(monkeypatch):
    monkeypatch.setattr("app.backoff.time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, text="boom")

    cache = TTLCache()
    tool = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(handler),
        fallback_transport=_no_fallback,
        cache=cache,
    )
    tool.invoke({"query": "q"})
    tool.invoke({"query": "q"})
    # attempts=3 per invoke (no caching on failure) -> 6 calls across two invokes,
    # not 2 -- if a cached error ever slipped through this would drop to 3.
    assert calls["n"] == 6


def test_web_search_retries_transient_500(monkeypatch):
    monkeypatch.setattr("app.backoff.time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json=_ok_response_body())

    tool = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(handler),
    )
    out = tool.invoke({"query": "q"})
    assert out.startswith("SEARCH_RESULTS")
    assert calls["n"] == 2

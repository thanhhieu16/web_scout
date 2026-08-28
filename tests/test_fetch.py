from pathlib import Path

import httpx

from app.config import FetchConfig
from app.tools.cache import TTLCache
from app.tools.fetch import clean_html, make_web_fetch

FIXTURE = Path(__file__).parent / "fixtures" / "page.html"


def _public_resolver(host):
    return ["93.184.216.34"]


def test_clean_html_extracts_article_text():
    html = FIXTURE.read_text(encoding="utf-8")
    text = clean_html(html, max_chars=5000)
    assert "agent harness built on LangGraph" in text
    assert "window.tracker" not in text
    assert ".x{color:red}" not in text


def test_clean_html_truncates():
    html = "<p>" + ("word " * 5000) + "</p>"
    assert len(clean_html(html, max_chars=200)) <= 210


def test_tool_returns_error_for_bad_scheme():
    tool = make_web_fetch(FetchConfig())
    out = tool.invoke({"url": "ftp://example.com/file"})
    assert out.startswith("FETCH_ERROR")


def test_tool_returns_error_for_loopback_host():
    # 127.0.0.1 is rejected by the guard before any socket is opened, so this
    # exercises check_url's loopback rejection, not a real connection attempt.
    tool = make_web_fetch(FetchConfig(timeout_seconds=2.0))
    out = tool.invoke({"url": "http://127.0.0.1:9/nope"})
    assert out.startswith("FETCH_ERROR")


def test_tool_returns_error_for_connect_failure():
    # The guard lets a public address through; this covers the transport-error
    # path that the guard no longer exercises for 127.0.0.1.
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    tool = make_web_fetch(
        FetchConfig(), transport=httpx.MockTransport(handler), resolve=_public_resolver
    )
    out = tool.invoke({"url": "https://ok.example/page"})
    assert out.startswith("FETCH_ERROR")
    assert "ConnectError" in out


def test_tool_rejects_content_length_over_cap():
    def handler(request):
        return httpx.Response(
            200,
            headers={"content-length": "5000000", "content-type": "text/html"},
        )

    tool = make_web_fetch(
        FetchConfig(), transport=httpx.MockTransport(handler), resolve=_public_resolver
    )
    out = tool.invoke({"url": "https://hostile.example/big"})
    assert out.startswith("FETCH_ERROR")
    assert "max_download_bytes" in out
    assert "2000000" in out


def test_tool_rejects_streamed_body_over_cap():
    def handler(request):
        def gen():
            yield b"x" * 1_048_576
            yield b"x" * 1_048_576
            yield b"x"

        return httpx.Response(200, content=gen(), headers={"content-type": "text/html"})

    tool = make_web_fetch(
        FetchConfig(), transport=httpx.MockTransport(handler), resolve=_public_resolver
    )
    out = tool.invoke({"url": "https://hostile.example/stream"})
    assert out.startswith("FETCH_ERROR")
    assert "max_download_bytes" in out


def test_tool_extracts_small_page_via_mock_transport():
    html = (
        "<html><body>"
        "<p>WebScout fetch cap check with enough surrounding text.</p>"
        "<script>window.evil()</script>"
        "</body></html>"
    )

    def handler(request):
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    tool = make_web_fetch(
        FetchConfig(), transport=httpx.MockTransport(handler), resolve=_public_resolver
    )
    out = tool.invoke({"url": "https://ok.example/page"})
    assert out.startswith("FETCH_ERROR") is False
    assert "WebScout fetch cap check" in out
    assert "evil" not in out


def test_tool_serves_repeat_url_from_cache_without_second_network_call():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(
            200, text="<p>cached content here</p>", headers={"content-type": "text/html"}
        )

    cache = TTLCache()
    tool = make_web_fetch(
        FetchConfig(), transport=httpx.MockTransport(handler), resolve=_public_resolver, cache=cache
    )
    first = tool.invoke({"url": "https://ok.example/page"})
    second = tool.invoke({"url": "https://ok.example/page"})
    assert first == second
    assert calls["n"] == 1


def test_tool_does_not_cache_fetch_errors():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(500, text="boom")

    cache = TTLCache()
    tool = make_web_fetch(
        FetchConfig(), transport=httpx.MockTransport(handler), resolve=_public_resolver, cache=cache
    )
    tool.invoke({"url": "https://ok.example/page"})
    tool.invoke({"url": "https://ok.example/page"})
    assert calls["n"] == 2

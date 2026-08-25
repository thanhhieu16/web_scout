from pathlib import Path

from app.config import FetchConfig
from app.tools.fetch import clean_html, make_web_fetch

FIXTURE = Path(__file__).parent / "fixtures" / "page.html"


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


def test_tool_returns_error_for_unreachable_host():
    tool = make_web_fetch(FetchConfig(timeout_seconds=2.0))
    out = tool.invoke({"url": "http://127.0.0.1:9/nope"})
    assert out.startswith("FETCH_ERROR")

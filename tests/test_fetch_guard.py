import httpx

from app.config import FetchConfig
from app.tools.fetch import check_url, make_web_fetch

PUBLIC = ["93.184.216.34"]


def _public_resolver(host):
    return PUBLIC


def _private_resolver(host):
    return ["127.0.0.1"]


def test_check_url_rejects_non_http_scheme():
    assert "FETCH_ERROR" in (check_url("ftp://a.dev/x", FetchConfig(), _public_resolver) or "")


def test_check_url_rejects_loopback():
    msg = check_url("http://internal.example/x", FetchConfig(), _private_resolver)
    assert msg is not None
    assert "private" in msg


def test_check_url_rejects_link_local_metadata():
    msg = check_url(
        "http://metadata.example/latest",
        FetchConfig(),
        lambda host: ["169.254.169.254"],
    )
    assert msg is not None


def test_check_url_allows_public_host():
    assert check_url("https://ok.example/page", FetchConfig(), _public_resolver) is None


def test_check_url_allows_private_when_opted_in():
    cfg = FetchConfig(allow_private_hosts=True)
    assert check_url("http://localhost/x", cfg, _private_resolver) is None


def test_tool_blocks_private_host():
    def handler(request):
        raise AssertionError("request must not be sent to a private host")

    tool = make_web_fetch(
        FetchConfig(),
        transport=httpx.MockTransport(handler),
        resolve=_private_resolver,
    )
    out = tool.invoke({"url": "http://internal.example/x"})
    assert out.startswith("FETCH_ERROR")
    assert "private" in out


def test_tool_blocks_redirect_into_private_network():
    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "http://internal.example/x"})
        raise AssertionError("must not follow the redirect")

    def resolve(host):
        return ["127.0.0.1"] if host == "internal.example" else PUBLIC

    tool = make_web_fetch(
        FetchConfig(), transport=httpx.MockTransport(handler), resolve=resolve
    )
    out = tool.invoke({"url": "https://ok.example/start"})
    assert out.startswith("FETCH_ERROR")
    assert "private" in out


def test_tool_follows_public_redirect():
    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://ok.example/final"})
        return httpx.Response(
            200,
            text="<html><body><p>Final page body with enough text.</p></body></html>",
            headers={"content-type": "text/html"},
        )

    tool = make_web_fetch(
        FetchConfig(), transport=httpx.MockTransport(handler), resolve=_public_resolver
    )
    out = tool.invoke({"url": "https://ok.example/start"})
    assert "Final page body" in out


def test_tool_stops_at_redirect_limit():
    def handler(request):
        return httpx.Response(302, headers={"location": "https://ok.example/again"})

    tool = make_web_fetch(
        FetchConfig(max_redirects=2),
        transport=httpx.MockTransport(handler),
        resolve=_public_resolver,
    )
    out = tool.invoke({"url": "https://ok.example/start"})
    assert out.startswith("FETCH_ERROR")
    assert "redirect" in out

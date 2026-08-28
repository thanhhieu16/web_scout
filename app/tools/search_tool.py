import re
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from langchain_core.tools import tool

from app.backoff import call_with_backoff
from app.config import Settings
from app.tools.search import build_search_spec

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}

_TAG_RE = re.compile(r"<[^>]+>")
_DDG_LINK_RE = re.compile(r"<a\s+([^>]*)>(.*?)</a>", re.DOTALL)
_HREF_RE = re.compile(r'href="([^"]+)"')


def _search_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRY_STATUS
    return isinstance(exc, httpx.TransportError)


def _strip_tags(html: str) -> str:
    return _TAG_RE.sub("", html).strip()


def _decode_ddg_href(href: str) -> str:
    parsed = urlparse(href.replace("&amp;", "&"))
    qs = parse_qs(parsed.query)
    if "uddg" in qs:
        return unquote(qs["uddg"][0])
    return href


def _duckduckgo_fallback(query: str, settings: Settings, transport=None) -> list[str]:
    """Last-resort search when OpenRouter's web_search hard-fails. Scrapes the
    no-JS DuckDuckGo HTML endpoint via regex rather than adding an HTML-parsing
    dependency for a path that only fires on primary-search failure."""
    try:
        with httpx.Client(
            transport=transport, timeout=settings.search.timeout_seconds
        ) as client:
            resp = client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (compatible; WebScoutFallback/0.1)"},
            )
            resp.raise_for_status()
    except Exception:
        return []
    results: list[tuple[str, str]] = []
    snippets: list[str] = []
    for attrs, inner in _DDG_LINK_RE.findall(resp.text):
        href_match = _HREF_RE.search(attrs)
        if not href_match:
            continue
        if 'class="result__a"' in attrs:
            url = _decode_ddg_href(href_match.group(1))
            if url.startswith("http"):
                results.append((url, _strip_tags(inner) or url))
        elif 'class="result__snippet"' in attrs:
            snippets.append(_strip_tags(inner)[:300])
    lines = []
    for i, (url, title) in enumerate(results):
        excerpt = snippets[i] if i < len(snippets) else ""
        lines.append(f"[SRC] {url} | {title}\nEXCERPT: {excerpt}")
        if len(lines) >= settings.search.max_results:
            break
    return lines


def make_web_search(
    settings: Settings, transport=None, usage=None, fallback_transport=None, cache=None
):
    @tool
    def web_search(query: str) -> str:
        """Search the web for current information. Returns titled results with URLs and excerpts."""
        cache_key = query.strip().lower()
        if cache is not None:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        body = {
            "model": settings.researcher.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Call web_search now for this query: {query}. Then stop."
                    ),
                }
            ],
            "tools": [build_search_spec(settings.search)],
            "max_tokens": 800,
            "usage": {"include": True},
        }

        def _post() -> httpx.Response:
            with httpx.Client(
                transport=transport, timeout=settings.search.timeout_seconds
            ) as client:
                response = client.post(
                    settings.openrouter_base_url + "/chat/completions",
                    headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
                    json=body,
                )
                response.raise_for_status()
                return response

        try:
            resp = call_with_backoff(
                _post, attempts=3, base_delay=2.0, retry_on=_search_retryable
            )
            data = resp.json()
            message = (data.get("choices") or [{}])[0].get("message") or {}
            annotations = message.get("annotations") or []
            lines = []
            seen: set[str] = set()
            for ann in annotations:
                cite = ann.get("url_citation") if isinstance(ann, dict) else None
                url = (cite or {}).get("url")
                if not url or url in seen:
                    continue
                seen.add(url)
                title = cite.get("title") or url
                excerpt = (cite.get("content") or "").replace("\n", " ")[:300]
                lines.append(f"[SRC] {url} | {title}\nEXCERPT: {excerpt}")
            totals = data.get("usage") or {}
            details = (
                totals.get("server_tool_use_details") or totals.get("server_tool_use") or {}
            )
            searches = details.get("web_search_requests", 0) if isinstance(details, dict) else 0
        except Exception as exc:
            error = f"SEARCH_ERROR: {type(exc).__name__}: {exc}"
            fallback_lines = _duckduckgo_fallback(query, settings, fallback_transport)
            if not fallback_lines:
                return error
            if usage is not None:
                usage.add(searches=1)
            header = f"SEARCH_RESULTS ({len(fallback_lines)} results, fallback: duckduckgo):"
            result = header + "\n\n" + "\n\n".join(fallback_lines)
            if cache is not None:
                cache.set(cache_key, result)
            return result
        if usage is not None:
            usage.add(
                tokens=totals.get("total_tokens", 0) or 0,
                cost=totals.get("cost", 0.0) or 0.0,
                searches=searches,
            )
        if not lines:
            fallback_lines = _duckduckgo_fallback(query, settings, fallback_transport)
            if not fallback_lines:
                return "SEARCH_ERROR: no results returned"
            if usage is not None:
                usage.add(searches=1)
            header = f"SEARCH_RESULTS ({len(fallback_lines)} results, fallback: duckduckgo):"
            result = header + "\n\n" + "\n\n".join(fallback_lines)
            if cache is not None:
                cache.set(cache_key, result)
            return result
        header = f"SEARCH_RESULTS ({len(lines)} results, {searches} search executed):"
        result = header + "\n\n" + "\n\n".join(lines)
        if cache is not None:
            cache.set(cache_key, result)
        return result

    return web_search

import httpx
from langchain_core.tools import tool

from app.backoff import call_with_backoff
from app.config import Settings
from app.tools.search import build_search_spec

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _search_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRY_STATUS
    return isinstance(exc, httpx.TransportError)


def make_web_search(settings: Settings, transport=None, usage=None):
    @tool
    def web_search(query: str) -> str:
        """Search the web for current information. Returns titled results with URLs and excerpts."""
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
                transport=transport, timeout=settings.fetch.timeout_seconds * 8
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
        except Exception as exc:
            return f"SEARCH_ERROR: {type(exc).__name__}: {exc}"
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
        details = totals.get("server_tool_use_details") or totals.get("server_tool_use") or {}
        searches = details.get("web_search_requests", 0) if isinstance(details, dict) else 0
        if usage is not None:
            usage.add(
                tokens=totals.get("total_tokens", 0) or 0,
                cost=totals.get("cost", 0.0) or 0.0,
                searches=searches,
            )
        if not lines:
            return "SEARCH_ERROR: no results returned"
        header = f"SEARCH_RESULTS ({len(lines)} results, {searches} search executed):"
        return header + "\n\n" + "\n\n".join(lines)

    return web_search

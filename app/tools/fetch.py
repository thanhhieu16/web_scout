import re

import httpx
import trafilatura
from langchain_core.tools import tool

from app.config import FetchConfig


def clean_html(html: str, max_chars: int) -> str:
    extracted = trafilatura.extract(
        html, output_format="txt", include_links=False, favor_recall=True
    )
    if not extracted:
        stripped = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", html, flags=re.I)
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        extracted = re.sub(r"\s+", " ", stripped).strip()
    return extracted[:max_chars]


def make_web_fetch(cfg: FetchConfig):
    @tool
    def web_fetch(url: str) -> str:
        """Fetch a web page and return its readable main text."""
        try:
            resp = httpx.get(
                url,
                timeout=cfg.timeout_seconds,
                headers={"User-Agent": cfg.user_agent},
                follow_redirects=True,
            )
            resp.raise_for_status()
        except Exception as exc:
            return f"FETCH_ERROR: {type(exc).__name__}: {exc}"
        ctype = resp.headers.get("content-type", "")
        if "html" not in ctype and "text" not in ctype:
            return f"FETCH_ERROR: unsupported content-type {ctype}"
        return clean_html(resp.text, cfg.max_chars)

    return web_fetch

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


def make_web_fetch(cfg: FetchConfig, transport: httpx.BaseTransport | None = None):
    @tool
    def web_fetch(url: str) -> str:
        """Fetch a web page and return its readable main text."""
        too_large = (
            f"FETCH_ERROR: response exceeds max_download_bytes "
            f"({cfg.max_download_bytes})"
        )
        try:
            with httpx.Client(
                timeout=cfg.timeout_seconds,
                headers={"User-Agent": cfg.user_agent},
                follow_redirects=True,
                transport=transport,
            ) as client:
                with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    declared = resp.headers.get("content-length")
                    if declared and int(declared) > cfg.max_download_bytes:
                        return too_large
                    buf = bytearray()
                    for chunk in resp.iter_bytes():
                        room = cfg.max_download_bytes + 1 - len(buf)
                        buf.extend(chunk[:room])
                        if len(buf) > cfg.max_download_bytes:
                            return too_large
                    text = bytes(buf).decode(resp.encoding or "utf-8", errors="replace")
        except Exception as exc:
            return f"FETCH_ERROR: {type(exc).__name__}: {exc}"
        ctype = resp.headers.get("content-type", "")
        if "html" not in ctype and "text" not in ctype:
            return f"FETCH_ERROR: unsupported content-type {ctype}"
        return clean_html(text, cfg.max_chars)

    return web_fetch

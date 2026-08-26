import ipaddress
import re
import socket
from urllib.parse import urljoin, urlsplit

import httpx
import trafilatura
from langchain_core.tools import tool

from app.config import FetchConfig

_ALLOWED_SCHEMES = {"http", "https"}


def clean_html(html: str, max_chars: int) -> str:
    extracted = trafilatura.extract(
        html, output_format="txt", include_links=False, favor_recall=True
    )
    if not extracted:
        stripped = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", html, flags=re.I)
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        extracted = re.sub(r"\s+", " ", stripped).strip()
    return extracted[:max_chars]


def default_resolve(host: str) -> list[str]:
    """Resolve a hostname to every address it points at."""
    infos = socket.getaddrinfo(host, None)
    return [info[4][0] for info in infos]


def _is_blocked_ip(raw: str) -> bool:
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def check_url(url: str, cfg: FetchConfig, resolve=default_resolve) -> str | None:
    """Return a FETCH_ERROR message if the URL must not be fetched, else None.

    Called once per redirect hop, not just on the initial URL — a public host
    that 302s to 169.254.169.254 is the whole reason this exists.
    """
    parts = urlsplit(url)
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        return f"FETCH_ERROR: unsupported scheme {parts.scheme!r}"
    host = parts.hostname
    if not host:
        return f"FETCH_ERROR: no host in url {url!r}"
    if cfg.allow_private_hosts:
        return None
    try:
        addresses = resolve(host)
    except Exception as exc:
        return f"FETCH_ERROR: cannot resolve {host}: {type(exc).__name__}: {exc}"
    if not addresses:
        return f"FETCH_ERROR: cannot resolve {host}"
    for address in addresses:
        if _is_blocked_ip(address):
            return f"FETCH_ERROR: refusing to fetch private address {address} for host {host}"
    return None


def make_web_fetch(
    cfg: FetchConfig,
    transport: httpx.BaseTransport | None = None,
    resolve=default_resolve,
):
    @tool
    def web_fetch(url: str) -> str:
        """Fetch a web page and return its readable main text."""
        too_large = (
            f"FETCH_ERROR: response exceeds max_download_bytes "
            f"({cfg.max_download_bytes})"
        )
        current = url
        try:
            with httpx.Client(
                timeout=cfg.timeout_seconds,
                headers={"User-Agent": cfg.user_agent},
                follow_redirects=False,
                transport=transport,
            ) as client:
                for _ in range(cfg.max_redirects + 1):
                    blocked = check_url(current, cfg, resolve)
                    if blocked:
                        return blocked
                    with client.stream("GET", current) as resp:
                        if resp.is_redirect:
                            location = resp.headers.get("location")
                            if not location:
                                return "FETCH_ERROR: redirect without a location header"
                            current = urljoin(current, location)
                            continue
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
                        ctype = resp.headers.get("content-type", "")
                        if "html" not in ctype and "text" not in ctype:
                            return f"FETCH_ERROR: unsupported content-type {ctype}"
                        text = bytes(buf).decode(resp.encoding or "utf-8", errors="replace")
                        return clean_html(text, cfg.max_chars)
                return f"FETCH_ERROR: exceeded max_redirects ({cfg.max_redirects})"
        except Exception as exc:
            return f"FETCH_ERROR: {type(exc).__name__}: {exc}"

    return web_fetch

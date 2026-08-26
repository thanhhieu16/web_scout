import re
from typing import Any

from app.schemas import Finding, Source
from app.tools.search import count_web_searches

_LINE_RE = re.compile(
    r"^-\s*\[(?P<ref>S\d+)\]\s+(?P<claim>.+?)\s*\|\s*confidence:\s*"
    r"(?P<conf>high|medium|low)\s*$",
    re.IGNORECASE,
)
_BLOCK_RE = re.compile(r"^##\s*FINDINGS\s*$", re.IGNORECASE | re.MULTILINE)


def parse_findings_block(text: str) -> tuple[list[Finding], list[list[str]], str]:
    match = _BLOCK_RE.search(text)
    if not match:
        return [], [], text
    narrative = text[: match.start()].rstrip()
    findings: list[Finding] = []
    refs: list[list[str]] = []
    for raw in text[match.end() :].splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        findings.append(
            Finding(
                claim=m.group("claim").strip(),
                source_urls=[],
                confidence=m.group("conf").lower(),
            )
        )
        refs.append([m.group("ref")])
    return findings, refs, narrative


def map_refs_to_urls(
    findings: list[Finding], refs: list[list[str]], citations: list[dict]
) -> tuple[list[Finding], list[str]]:
    ordered_urls = [c.get("url") for c in citations if c.get("url")]
    ref_to_url = {f"S{i + 1}": u for i, u in enumerate(ordered_urls)}
    mapped: list[Finding] = []
    for finding, finding_refs in zip(findings, refs, strict=True):
        urls: list[str] = []
        for r in finding_refs:
            if r in ref_to_url:
                urls.append(ref_to_url[r])
            else:
                urls.append(f"unresolved:{r}")
        mapped.append(Finding(
            claim=finding["claim"],
            source_urls=list(dict.fromkeys(urls)),
            confidence=finding["confidence"],
        ))
    return mapped, []


def extract_url_citations(message: Any) -> list[dict]:
    containers = [
        getattr(message, "annotations", None),
        (getattr(message, "additional_kwargs", None) or {}).get("annotations"),
        (getattr(message, "response_metadata", None) or {}).get("annotations"),
    ]
    out: list[dict] = []
    for anns in containers:
        if not anns:
            continue
        if isinstance(anns, dict):
            anns = [anns]
        for entry in anns:
            if not isinstance(entry, dict):
                continue
            cite = entry.get("url_citation") or (
                entry if entry.get("url") else None
            )
            if isinstance(cite, dict) and cite.get("url"):
                out.append(cite)
    seen: set[str] = set()
    deduped = [c for c in out if not (c["url"] in seen or seen.add(c["url"]))]
    return deduped


def collect_citations(messages) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for message in messages:
        for cite in extract_url_citations(message):
            url = cite.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            out.append(cite)
    return out


def count_total_searches(messages) -> int:
    total = 0
    for message in messages:
        try:
            total += count_web_searches(message)
        except (TypeError, ValueError):
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str):
            match = re.search(r"(\d+) search executed", content)
            if match:
                total += int(match.group(1))
    return total


def sum_usage(messages) -> tuple[int, float]:
    tokens = 0
    cost = 0.0
    for message in messages:
        usage_meta = getattr(message, "usage_metadata", None)
        if isinstance(usage_meta, dict):
            tokens += int(usage_meta.get("total_tokens", 0) or 0)
        else:
            meta = getattr(message, "response_metadata", None) or {}
            token_usage = meta.get("token_usage") if isinstance(meta, dict) else None
            if isinstance(token_usage, dict):
                tokens += int(token_usage.get("total_tokens", 0) or 0)
        meta = getattr(message, "response_metadata", None) or {}
        token_usage = meta.get("token_usage") if isinstance(meta, dict) else None
        if isinstance(token_usage, dict):
            try:
                cost += float(token_usage.get("cost", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
    return tokens, round(cost, 6)


_SEARCH_ENGINE_MARKERS = (
    "duckduckgo.com/html",
    "duckduckgo.com/?q",
    "google.com/search",
    "bing.com/search",
    "search.brave.com/search",
)


def _is_search_result_page(url: str) -> bool:
    return any(marker in url for marker in _SEARCH_ENGINE_MARKERS)


def collect_fetched_sources(messages) -> "list[Source]":
    pending: dict[str, str] = {}
    out: list[Source] = []
    seen: set[str] = set()
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            if call.get("name") == "web_fetch":
                url = (call.get("args") or {}).get("url")
                if url and not url.startswith("FETCH_ERROR"):
                    pending[call.get("id")] = url
        content = getattr(message, "content", "")
        if isinstance(content, list):
            content = str(content)
        if getattr(message, "type", "") == "tool" and str(
            getattr(message, "tool_call_id", "")
        ) in pending:
            url = pending.pop(str(message.tool_call_id))
            text = str(content)
            if text.startswith("FETCH_ERROR") or url in seen:
                continue
            if _is_search_result_page(url):
                continue
            seen.add(url)
            out.append(Source(url=url, title=url, source_type="secondary", excerpt=text[:500]))
    return out


def collect_search_tool_sources(messages) -> "list[Source]":
    pending: set[str] = set()
    out: list[Source] = []
    seen: set[str] = set()
    block_re = re.compile(r"\[SRC\] (\S+) \| ([^\n]+)(?:\nEXCERPT: ([^\n]+))?")
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if call.get("name") == "web_search":
                pending.add(str(call.get("id")))
        content = getattr(message, "content", "")
        if isinstance(content, list):
            content = str(content)
        if getattr(message, "type", "") == "tool" and str(
            getattr(message, "tool_call_id", "")
        ) in pending:
            pending.discard(str(message.tool_call_id))
            for match in block_re.finditer(str(content)):
                url, title, excerpt = match.group(1), match.group(2), match.group(3)
                if url in seen:
                    continue
                seen.add(url)
                out.append(
                    Source(
                        url=url,
                        title=title.strip(),
                        source_type="secondary",
                        excerpt=(excerpt or "").strip()[:500],
                    )
                )
    return out


def build_sources(messages) -> "tuple[list[Source], list[dict]]":
    citations = collect_citations(messages)
    search_tool = collect_search_tool_sources(messages)
    fetched = collect_fetched_sources(messages)
    known: dict[str, Source] = {}
    ordered: list[Source] = []
    for cite in citations:
        url = cite.get("url")
        if not url or url in known:
            continue
        source = Source(
            url=url,
            title=cite.get("title") or url,
            source_type="secondary",
            excerpt=(cite.get("content") or "")[:500],
        )
        known[url] = source
        ordered.append(source)
    for source in search_tool + fetched:
        if source["url"] not in known:
            known[source["url"]] = source
            ordered.append(source)
    ref_order = [{"url": s["url"]} for s in ordered]
    return ordered, ref_order


def reconcile_sources(
    findings: list[Finding], citations: list[dict]
) -> tuple[list[Source], list[str]]:
    sources: list[Source] = []
    known: set[str] = set()
    for cite in citations:
        url = cite.get("url")
        if not url or url in known:
            continue
        known.add(url)
        sources.append(
            Source(
                url=url,
                title=cite.get("title") or url,
                source_type="secondary",
                excerpt=(cite.get("content") or "")[:500],
            )
        )
    unknown: list[str] = []
    for finding in findings:
        for url in finding.get("source_urls", []):
            resolvable = url.startswith("http") or url.startswith("unresolved:")
            if url not in known and url not in unknown and resolvable:
                unknown.append(url)
    return sources, unknown

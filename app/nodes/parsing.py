import re
import unicodedata
from typing import Any, NamedTuple

from app.schemas import Finding, Source
from app.tools.search import count_web_searches

_LINE_RE = re.compile(
    r"^-\s*(?P<refs>(?:\[S\d+\][\s,]*)+)\s*(?P<claim>.+?)\s*\|\s*confidence:\s*"
    r"(?P<conf>high|medium|low)(?:\s*\|\s*quote:\s*(?P<quote>.+?))?\s*$",
    re.IGNORECASE,
)
_QUOTE_MARKS = "\"'“”‘’«»"
_REF_RE = re.compile(r"S\d+", re.IGNORECASE)
_BLOCK_RE = re.compile(r"^##\s*FINDINGS\s*$", re.IGNORECASE | re.MULTILINE)
# A line that opens like a finding, or carries a confidence marker, but still fails
# _LINE_RE is a possibly-lost claim (e.g. a claim missing its [Sn] ref entirely).
# Anything else after the block is trailing prose, not evidence. Deliberately kept
# independent of _LINE_RE so a change to one doesn't silently narrow the other.
_ATTEMPTED_RE = re.compile(r"^-\s*\[S\d+\]|\|\s*confidence\s*:", re.IGNORECASE)


class FindingsParse(NamedTuple):
    findings: list[Finding]
    refs: list[list[str]]
    narrative: str
    dropped: list[str]
    block_found: bool
    # Parallel to `findings`; "" where the line carried no `| quote:` part.
    quotes: list[str]


def parse_findings_block(text: str) -> FindingsParse:
    match = _BLOCK_RE.search(text)
    if not match:
        return FindingsParse([], [], text, [], False, [])
    narrative = text[: match.start()].rstrip()
    findings: list[Finding] = []
    refs: list[list[str]] = []
    dropped: list[str] = []
    quotes: list[str] = []
    for raw in text[match.end() :].splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if not m:
            if _ATTEMPTED_RE.search(line):
                dropped.append(line)
            continue
        findings.append(
            Finding(
                claim=m.group("claim").strip(),
                source_urls=[],
                confidence=m.group("conf").lower(),
            )
        )
        refs.append([r.upper() for r in _REF_RE.findall(m.group("refs"))])
        quotes.append((m.group("quote") or "").strip().strip(_QUOTE_MARKS).strip())
    return FindingsParse(findings, refs, narrative, dropped, True, quotes)


def map_refs_to_urls(
    findings: list[Finding], refs: list[list[str]], ref_map: dict[str, str]
) -> list[Finding]:
    """Resolve each finding's `Sn` refs through `ref_map` (from `build_sources`);
    a ref missing from the map becomes `unresolved:Sn`."""
    mapped: list[Finding] = []
    for finding, finding_refs in zip(findings, refs, strict=True):
        urls = [ref_map.get(r, f"unresolved:{r}") for r in finding_refs]
        mapped.append(
            Finding(
                claim=finding["claim"],
                source_urls=list(dict.fromkeys(urls)),
                confidence=finding["confidence"],
            )
        )
    return mapped


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
    """Counts only server-tool usage reported in message metadata. Searches made
    by the client-side web_search tool arrive through UsageCollector instead."""
    total = 0
    for message in messages:
        try:
            total += count_web_searches(message)
        except (TypeError, ValueError):
            continue
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


# web_fetch output opens with "[Sn] <url>" then a blank line (app/tools/fetch.py).
_FETCH_HEADER_RE = re.compile(r"^\[(S\d+)\] \S+\n\n")
# web_search result blocks (app/tools/search_tool.py). "SRC" is the legacy
# unnumbered label, still accepted so an old transcript parses.
_SEARCH_BLOCK_RE = re.compile(r"\[(S\d+|SRC)\] (\S+) \| ([^\n]+)(?:\nEXCERPT: ([^\n]+))?")


def _fetched_entries(messages) -> "list[tuple[Source, str | None]]":
    pending: dict[str, str] = {}
    out: list[tuple[Source, str | None]] = []
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
            header = _FETCH_HEADER_RE.match(text)
            ref = header.group(1) if header else None
            body = text[header.end() :] if header else text
            out.append(
                (Source(url=url, title=url, source_type="secondary", excerpt=body[:500]), ref)
            )
    return out


def collect_fetched_sources(messages) -> "list[Source]":
    return [source for source, _ in _fetched_entries(messages)]


def _search_tool_entries(messages) -> "list[tuple[Source, str | None]]":
    pending: set[str] = set()
    out: list[tuple[Source, str | None]] = []
    seen: set[str] = set()
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
            for match in _SEARCH_BLOCK_RE.finditer(str(content)):
                label, url, title, excerpt = match.groups()
                if url in seen:
                    continue
                seen.add(url)
                source = Source(
                    url=url,
                    title=title.strip(),
                    source_type="secondary",
                    excerpt=(excerpt or "").strip()[:500],
                )
                out.append((source, None if label == "SRC" else label.upper()))
    return out


def collect_search_tool_sources(messages) -> "list[Source]":
    return [source for source, _ in _search_tool_entries(messages)]


def build_sources(messages) -> "tuple[list[Source], dict[str, str]]":
    """Return the ordered source list and the `Sn` -> url map findings resolve through.

    The map comes from the IDs the tools printed next to each URL (see
    app/tools/sources.py), so it no longer depends on the list order. Only a
    transcript where no tool printed an ID (annotation-only, or legacy
    unnumbered output) falls back to numbering sources by list position.
    """
    citations = collect_citations(messages)
    tool_entries = _search_tool_entries(messages) + _fetched_entries(messages)
    known: dict[str, Source] = {}
    ordered: list[Source] = []
    ref_map: dict[str, str] = {}
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
    for source, ref in tool_entries:
        if ref is not None:
            ref_map.setdefault(ref, source["url"])
        if source["url"] not in known:
            known[source["url"]] = source
            ordered.append(source)
    if not ref_map:
        ref_map = {f"S{i + 1}": s["url"] for i, s in enumerate(ordered)}
    return ordered, ref_map


def find_unknown_refs(findings: list[Finding], ref_map: dict[str, str]) -> list[str]:
    """URLs a finding cites that are not among the sources `ref_map` resolves to."""
    known = set(ref_map.values())
    unknown: list[str] = []
    for finding in findings:
        for url in finding.get("source_urls", []):
            resolvable = url.startswith("http") or url.startswith("unresolved:")
            if resolvable and url not in known and url not in unknown:
                unknown.append(url)
    return unknown


# Fewer words than this and a "quote" matches almost any page by accident.
_MIN_QUOTE_WORDS = 4
_ELLIPSIS_RE = re.compile(r"\.\.\.|…")


def _normalize_for_match(text: str) -> str:
    """Casefold and reduce to words, so a quote survives the punctuation,
    quote-mark and whitespace drift between extracted page text and what the
    model copies out of it."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[\W_]+", " ", text).split())


def collect_source_texts(messages) -> dict[str, str]:
    """Full text the agent actually saw per URL: fetched page bodies plus
    web_search excerpts. `build_sources` keeps only 500-char excerpts, too short
    to check a quote against."""
    texts: dict[str, list[str]] = {}
    for source, _ in _search_tool_entries(messages):
        texts.setdefault(source["url"], []).append(source["excerpt"])
    pending: dict[str, str] = {}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if call.get("name") == "web_fetch":
                url = (call.get("args") or {}).get("url")
                if url:
                    pending[str(call.get("id"))] = url
        if getattr(message, "type", "") != "tool":
            continue
        url = pending.pop(str(getattr(message, "tool_call_id", "")), None)
        content = str(getattr(message, "content", ""))
        if url is None or content.startswith("FETCH_ERROR"):
            continue
        header = _FETCH_HEADER_RE.match(content)
        texts.setdefault(url, []).append(content[header.end() :] if header else content)
    return {url: "\n".join(parts) for url, parts in texts.items()}


def _quote_in(quote: str, text: str) -> bool:
    haystack = _normalize_for_match(text)
    fragments = [_normalize_for_match(f) for f in _ELLIPSIS_RE.split(quote)]
    fragments = [f for f in fragments if f]
    return bool(fragments) and all(f in haystack for f in fragments)


def ground_findings(
    findings: list[Finding], quotes: list[str], texts: dict[str, str]
) -> tuple[list[Finding], list[str]]:
    """Check each finding's quote against the text of the sources it cites.

    - quote found in a cited source: `grounded=True`, confidence unchanged.
    - no usable quote: `grounded=False`; "high" drops to "medium", since
      nothing ties the claim to its source text.
    - quote given but in none of the cited sources (paraphrased, misattributed
      or invented): `grounded=False`, confidence drops to "low", and a weak
      claim is reported.
    """
    out: list[Finding] = []
    weak: list[str] = []
    for finding, quote in zip(findings, quotes, strict=True):
        grounded_finding = Finding(**finding, quote=quote, grounded=False)
        if len(_normalize_for_match(quote).split()) < _MIN_QUOTE_WORDS:
            if grounded_finding["confidence"] == "high":
                grounded_finding["confidence"] = "medium"
        elif any(
            _quote_in(quote, texts.get(url, "")) for url in finding.get("source_urls", [])
        ):
            grounded_finding["grounded"] = True
        else:
            grounded_finding["confidence"] = "low"
            weak.append(f"quote not found in cited source(s): {finding['claim']}")
        out.append(grounded_finding)
    return out, weak

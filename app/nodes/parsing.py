import re
from typing import Any

from app.schemas import Finding, Source

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
    for finding, finding_refs in zip(findings, refs):
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

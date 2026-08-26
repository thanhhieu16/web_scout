import operator
from typing import Annotated, TypedDict

_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def merge_sources(prior: list, new: list) -> list:
    """Dedupe by url, first occurrence wins — this is what keeps [Sn] numbering
    stable once a second research iteration runs."""
    out = list(prior)
    seen = {s.get("url") for s in out}
    for source in new:
        url = source.get("url")
        if url and url not in seen:
            seen.add(url)
            out.append(source)
    return out


def _claim_key(finding: dict) -> str:
    return " ".join(str(finding.get("claim", "")).split()).casefold()


def merge_findings(prior: list, new: list) -> list:
    """Dedupe by normalized claim text. A repeat is not independent confirmation —
    iteration 2 is instructed to research gaps only — so confidence never rises on
    a merge; it takes the more pessimistic of the two."""
    out = [dict(f) for f in prior]
    index = {_claim_key(f): i for i, f in enumerate(out)}
    for finding in new:
        key = _claim_key(finding)
        if key not in index:
            index[key] = len(out)
            out.append(dict(finding))
            continue
        existing = out[index[key]]
        urls = list(existing.get("source_urls") or [])
        for url in finding.get("source_urls") or []:
            if url not in urls:
                urls.append(url)
        existing["source_urls"] = urls
        old = _CONFIDENCE_ORDER.get(existing.get("confidence", "low"), 0)
        incoming = _CONFIDENCE_ORDER.get(finding.get("confidence", "low"), 0)
        if incoming < old:
            existing["confidence"] = finding["confidence"]
    return out


def merge_weak_claims(prior: list, new: list) -> list:
    return list(dict.fromkeys(list(prior) + list(new)))


class ResearchState(TypedDict, total=False):
    question: str
    max_iterations: int
    answer_language: str

    findings: Annotated[list, merge_findings]
    sources: Annotated[list, merge_sources]
    weak_claims: Annotated[list, merge_weak_claims]

    iteration: Annotated[int, operator.add]
    search_calls: Annotated[int, operator.add]
    total_tokens: Annotated[int, operator.add]
    total_cost: Annotated[float, operator.add]

    # Current verdict, not a ledger — last write wins on purpose.
    gaps: list
    contradictory_claims: list
    sufficient: bool
    answer: str

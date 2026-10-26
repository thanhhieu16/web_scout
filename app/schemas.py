from typing import NotRequired, TypedDict

from pydantic import BaseModel


class Source(TypedDict):
    url: str
    title: str
    source_type: str
    excerpt: str


class Finding(TypedDict):
    claim: str
    source_urls: list[str]
    confidence: str
    # Verbatim supporting text the agent copied from a cited source, and whether
    # ground_findings found it there. Absent on findings built before grounding.
    quote: NotRequired[str]
    grounded: NotRequired[bool]


class VerificationResult(BaseModel):
    sufficient: bool
    missing_information: list[str] = []
    weak_claims: list[str] = []
    contradictory_claims: list[str] = []

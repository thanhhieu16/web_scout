from typing import TypedDict

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


class VerificationResult(BaseModel):
    sufficient: bool
    missing_information: list[str] = []
    weak_claims: list[str] = []
    contradictory_claims: list[str] = []

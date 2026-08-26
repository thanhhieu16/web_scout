from typing import TypedDict


class ResearchState(TypedDict, total=False):
    question: str
    findings: list
    sources: list
    gaps: list
    weak_claims: list
    contradictory_claims: list
    sufficient: bool
    iteration: int
    max_iterations: int
    answer: str
    answer_language: str
    search_calls: int
    total_tokens: int
    total_cost: float

import pytest
from pydantic import ValidationError

from app.schemas import VerificationResult


def test_verification_result_defaults():
    v = VerificationResult(sufficient=True)
    assert v.missing_information == []
    assert v.weak_claims == []
    assert v.contradictory_claims == []


def test_verification_result_rejects_missing_sufficient():
    with pytest.raises(ValidationError):
        VerificationResult()  # type: ignore[call-arg]


def test_state_has_required_fields():
    from app.state import ResearchState

    hints = ResearchState.__annotations__
    for field in (
        "question",
        "findings",
        "sources",
        "gaps",
        "weak_claims",
        "contradictory_claims",
        "sufficient",
        "iteration",
        "max_iterations",
        "answer",
        "answer_language",
        "search_calls",
    ):
        assert field in hints, field

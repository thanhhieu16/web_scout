import pytest

from app.graph import route_after_verify


def test_route_sufficient_goes_answer():
    assert route_after_verify({"sufficient": True}) == "answer"


def test_route_budget_left_goes_research():
    assert (
        route_after_verify({"sufficient": False, "iteration": 1, "max_iterations": 3})
        == "research"
    )


def test_route_budget_exhausted_goes_answer():
    assert (
        route_after_verify({"sufficient": False, "iteration": 3, "max_iterations": 3})
        == "answer"
    )


def test_build_graph_compiles(monkeypatch):
    import app.graph as g

    monkeypatch.setattr(g, "build_research_agent", lambda s=None: object())
    compiled = g.build_graph()
    assert compiled is not None

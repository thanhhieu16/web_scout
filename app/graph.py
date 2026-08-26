from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.agent import build_research_agent
from app.config import get_settings
from app.nodes.answer import make_answer_node
from app.nodes.research import make_research_node
from app.nodes.verify import make_verify_node
from app.state import ResearchState
from app.usage import UsageCollector


def route_after_verify(
    state: ResearchState,
) -> Literal["research", "answer"]:
    if state.get("sufficient"):
        return "answer"
    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        return "answer"
    return "research"


def build_graph(settings=None):
    s = settings or get_settings()
    usage = UsageCollector()
    agent = build_research_agent(s, usage=usage)
    graph = StateGraph(ResearchState)
    graph.add_node("research", make_research_node(agent, s, usage=usage))
    graph.add_node("verify", make_verify_node(s))
    graph.add_node("answer", make_answer_node(s))
    graph.add_edge(START, "research")
    graph.add_edge("research", "verify")
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {"research": "research", "answer": "answer"},
    )
    graph.add_edge("answer", END)
    return graph.compile()

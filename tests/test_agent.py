from app.agent import RESEARCH_SYSTEM_PROMPT, build_research_agent
from app.config import Settings


def test_system_prompt_contains_contract():
    p = RESEARCH_SYSTEM_PROMPT
    assert "## FINDINGS" in p
    assert "[S1]" in p
    assert "confidence" in p
    assert "Never invent" in p


def test_build_research_agent_offline_construction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    s.skills_enabled = False
    agent = build_research_agent(s)
    assert agent is not None

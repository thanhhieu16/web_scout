import app.config
from app.agent import RESEARCH_SYSTEM_PROMPT, build_research_agent
from app.config import Settings


def test_system_prompt_contains_contract():
    p = RESEARCH_SYSTEM_PROMPT
    assert "## FINDINGS" in p
    assert "[S1]" in p
    assert "confidence" in p
    assert "Never invent" in p


def test_system_prompt_documents_multi_ref():
    assert "[S1][S2]" in RESEARCH_SYSTEM_PROMPT


def test_build_research_agent_offline_construction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    s.skills_enabled = False
    agent = build_research_agent(s)
    assert agent is not None


def test_skills_resolve_from_repo_root_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    s.skills_enabled = True
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return object()

    import app.agent as a

    monkeypatch.setattr(a, "_create_deep_agent", fake_create)
    a.build_research_agent(s)
    assert captured["skills"] == ["skills/"]
    assert "backend" in captured
    assert (app.config.REPO_ROOT / "skills").is_dir()

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
    assert captured["skills"] == ["/skills/"]
    assert "backend" in captured
    assert captured["permissions"]
    assert (app.config.REPO_ROOT / "skills").is_dir()


def test_skills_backend_exposes_only_skills_dir():
    from deepagents.middleware.filesystem import _check_fs_permission
    from deepagents.middleware.skills import _list_skills_with_errors

    from app.agent import _skills_backend

    backend, permissions = _skills_backend()
    skills, error = _list_skills_with_errors(backend, "/skills/")
    assert error is None
    assert [s["name"] for s in skills] == ["web-research"]
    # The repo root (and its .env) is unreachable from the skills route.
    for path in ("/skills/../.env", "/skills/../app/agent.py"):
        try:
            result = backend.read(path)
        except ValueError:
            continue
        assert result.file_data is None
    assert _check_fs_permission(permissions, "write", "/skills/web-research/SKILL.md") == "deny"
    assert _check_fs_permission(permissions, "read", "/skills/web-research/SKILL.md") == "allow"


def test_skills_backend_scratch_writes_stay_in_memory():
    from deepagents.backends import StateBackend

    from app.agent import _skills_backend

    backend, _ = _skills_backend()
    assert isinstance(backend.default, StateBackend)


def test_system_prompt_tells_agent_to_copy_tool_ids():
    assert "exact ID the tool" in RESEARCH_SYSTEM_PROMPT
    assert "order you first used" not in RESEARCH_SYSTEM_PROMPT


def test_research_agent_tools_share_one_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    s.skills_enabled = False
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return object()

    import app.agent as a
    from app.tools.sources import SourceRegistry

    monkeypatch.setattr(a, "_create_deep_agent", fake_create)
    registry = SourceRegistry()
    registry.id_for("https://first.dev")
    a.build_research_agent(s, registry=registry)
    assert captured["model"].rate_limit_attempts == a.RESEARCHER_RATE_LIMIT_ATTEMPTS
    assert registry.id_for("https://second.dev") == "S2"

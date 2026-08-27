from pathlib import Path

import pytest

from app.config import Settings


def test_yaml_values_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "answer:\n  model: other/model\n  temperature: 0.7\nmax_iterations: 2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.answer.model == "other/model"
    assert s.answer.temperature == 0.7
    assert s.max_iterations == 2
    assert s.openrouter_api_key == "sk-test"


def test_defaults_when_no_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.researcher.model == "z-ai/glm-5.3-flash"
    assert s.search.model_dump() == {
        "max_results": 5,
        "max_uses": 4,
        "max_characters": 4000,
        "timeout_seconds": 30.0,
    }


def test_load_env_file_populates_os_environ(tmp_path, monkeypatch):
    import os

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("LANGSMITH_PROJECT=webscout-test\n", encoding="utf-8")
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    from app.config import load_env_file

    load_env_file(str(tmp_path / '.env'))
    assert os.environ["LANGSMITH_PROJECT"] == "webscout-test"


def test_config_yaml_falls_back_to_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    # repo config.yaml sets skills_enabled: true; the code default is False
    assert s.skills_enabled is True


def test_override_model_sets_every_role(monkeypatch: pytest.MonkeyPatch):
    from app.config import MODEL_CHOICES, ROLE_NAMES, get_settings, override_model

    try:
        s = override_model("minimax/minimax-m3:free")
        assert [getattr(s, r).model for r in ROLE_NAMES] == [
            "minimax/minimax-m3:free"
        ] * len(ROLE_NAMES)
        # Same object every later call site reads.
        assert get_settings() is s
        # Temperatures are per-role and must survive a model swap.
        assert s.researcher.temperature != s.answer.temperature
    finally:
        get_settings.cache_clear()
    assert "minimax/minimax-m3:free" in MODEL_CHOICES


def test_model_choices_are_unique_and_nonempty():
    from app.config import DEFAULT_MODEL, MODEL_CHOICES

    assert len(set(MODEL_CHOICES)) == len(MODEL_CHOICES)
    assert MODEL_CHOICES[0] == DEFAULT_MODEL
    assert all(slug.count("/") == 1 for slug in MODEL_CHOICES)


def test_conversations_db_path_defaults_under_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONVERSATIONS_DB_PATH", raising=False)
    from app.config import REPO_ROOT

    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.conversations_db_path == str(REPO_ROOT / "data" / "webscout.db")

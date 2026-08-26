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
    assert s.researcher.model == "stealth/ox-alpha"
    assert s.search.model_dump() == {"max_results": 5, "max_uses": 4, "max_characters": 4000}


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

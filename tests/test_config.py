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

import pytest

from app.config import Settings
from evals.run_evals import load_dataset, target


def test_load_dataset_shape():
    rows = load_dataset()
    assert len(rows) >= 20
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    assert all({"id", "category", "question", "reference_notes"} <= set(r) for r in rows)


def test_target_maps_outputs(monkeypatch):
    class FakeCompiled:
        def invoke(self, state):
            return {"answer": "A", "sources": [{"url": "u"}], "search_calls": 5}

    import evals.run_evals as mod

    monkeypatch.setattr(mod, "_graph", lambda: FakeCompiled())
    out = target({"question": "q"})
    assert out == {"answer": "A", "sources": [{"url": "u"}], "search_calls": 5}


def test_run_evals_main_fails_fast_without_api_key(monkeypatch):
    import evals.run_evals as mod

    def forbidden_client(*args, **kwargs):
        raise AssertionError("Client must not be constructed without an API key")

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("WEBSCOUT_OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(mod, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(mod, "Client", forbidden_client)
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY is not set"):
        mod.main([])

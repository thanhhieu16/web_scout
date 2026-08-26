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

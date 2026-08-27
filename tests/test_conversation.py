from types import SimpleNamespace


def _fake_model(reply):
    def invoke(messages):
        return SimpleNamespace(content=reply)

    return SimpleNamespace(invoke=invoke)


class ExplodingModel:
    def invoke(self, messages):
        raise AssertionError("model must not be called when history is empty")


class RaisingModel:
    def invoke(self, messages):
        raise RuntimeError("rate limited")


class RecordingModel:
    def __init__(self, reply="Rewritten?"):
        self.reply = reply
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.reply)


def test_condense_question_skips_model_when_history_empty():
    from app.conversation import condense_question

    out = condense_question([], "What is LangGraph?", model=ExplodingModel())
    assert out == "What is LangGraph?"


def test_condense_question_rewrites_with_history():
    from app.conversation import condense_question

    history = [{"question": "What is LangGraph?", "answer": "A framework."}]
    out = condense_question(history, "and that?", model=_fake_model("What is LangChain?"))
    assert out == "What is LangChain?"


def test_condense_question_falls_back_on_model_error():
    from app.conversation import condense_question

    history = [{"question": "What is LangGraph?", "answer": "A framework."}]
    out = condense_question(history, "and that?", model=RaisingModel())
    assert out == "and that?"


def test_condense_question_blank_rewrite_falls_back_to_original():
    from app.conversation import condense_question

    history = [{"question": "What is LangGraph?", "answer": "A framework."}]
    out = condense_question(history, "and that?", model=_fake_model("   "))
    assert out == "and that?"


def test_condense_question_uses_last_three_turns_only():
    from app.conversation import condense_question

    history = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(5)]
    model = RecordingModel()
    condense_question(history, "and that?", model=model)
    sent = model.calls[0][1][1]  # messages == [("system", ...), ("human", text)]
    assert "q0" not in sent and "q1" not in sent
    assert "q2" in sent and "q3" in sent and "q4" in sent

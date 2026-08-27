import pytest

from web import store


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    store.init_db(path)
    return path


def test_create_conversation_returns_id_with_default_title(db_path):
    conv_id = store.create_conversation(db_path)
    conversations = store.list_conversations(db_path)
    assert len(conversations) == 1
    assert conversations[0]["id"] == conv_id
    assert conversations[0]["title"] == store.DEFAULT_TITLE
    assert conversations[0]["updated_at"]


def test_list_conversations_orders_by_updated_at_desc(db_path):
    first = store.create_conversation(db_path)
    second = store.create_conversation(db_path)
    store.append_message(db_path, first, "hi", {"answer": "hello"})
    ids = [c["id"] for c in store.list_conversations(db_path)]
    assert ids == [first, second]


def test_get_conversation_returns_messages_in_order(db_path):
    conv_id = store.create_conversation(db_path)
    store.append_message(db_path, conv_id, "q1", {"answer": "a1"})
    store.append_message(db_path, conv_id, "q2", {"answer": "a2"})
    conv = store.get_conversation(db_path, conv_id)
    assert conv["messages"] == [
        {"question": "q1", "out": {"answer": "a1"}},
        {"question": "q2", "out": {"answer": "a2"}},
    ]


def test_get_conversation_returns_none_for_missing_id(db_path):
    assert store.get_conversation(db_path, 999) is None


def test_append_message_sets_title_from_first_question(db_path):
    conv_id = store.create_conversation(db_path)
    store.append_message(db_path, conv_id, "What is LangGraph?", {"answer": "..."})
    conv = store.get_conversation(db_path, conv_id)
    assert conv["title"] == "What is LangGraph?"


def test_append_message_truncates_long_first_question(db_path):
    conv_id = store.create_conversation(db_path)
    long_q = "x" * 60
    store.append_message(db_path, conv_id, long_q, {"answer": "..."})
    conv = store.get_conversation(db_path, conv_id)
    assert conv["title"] == "x" * 40 + "…"


def test_append_message_does_not_overwrite_title_on_second_message(db_path):
    conv_id = store.create_conversation(db_path)
    store.append_message(db_path, conv_id, "first question", {"answer": "a1"})
    store.append_message(db_path, conv_id, "second question", {"answer": "a2"})
    conv = store.get_conversation(db_path, conv_id)
    assert conv["title"] == "first question"


def test_append_message_does_not_overwrite_a_manually_renamed_title(db_path):
    conv_id = store.create_conversation(db_path)
    store.rename_conversation(db_path, conv_id, "My custom title")
    store.append_message(db_path, conv_id, "first question", {"answer": "a1"})
    conv = store.get_conversation(db_path, conv_id)
    assert conv["title"] == "My custom title"


def test_append_message_raises_for_missing_conversation(db_path):
    with pytest.raises(KeyError):
        store.append_message(db_path, 999, "q", {"answer": "a"})


def test_rename_conversation_returns_true_on_success(db_path):
    conv_id = store.create_conversation(db_path)
    assert store.rename_conversation(db_path, conv_id, "New title") is True
    assert store.list_conversations(db_path)[0]["title"] == "New title"


def test_rename_conversation_returns_false_for_missing_id(db_path):
    assert store.rename_conversation(db_path, 999, "x") is False


def test_delete_conversation_returns_true_and_cascades_messages(db_path):
    conv_id = store.create_conversation(db_path)
    store.append_message(db_path, conv_id, "q", {"answer": "a"})
    assert store.delete_conversation(db_path, conv_id) is True
    assert store.list_conversations(db_path) == []
    assert store.get_conversation(db_path, conv_id) is None


def test_delete_conversation_returns_false_for_missing_id(db_path):
    assert store.delete_conversation(db_path, 999) is False

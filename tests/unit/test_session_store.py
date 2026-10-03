"""
Unit tests for app/session_store.py — SessionStore class.

Covers all acceptance criteria for task 3.3:
- create_session returns a valid UUID string.
- get_session with an unknown ID raises SessionNotFoundError.
- After appending 21 exchanges, get_history returns exactly 20 with the
  1st appended no longer present.
- delete_session removes the session; a subsequent get_session raises
  SessionNotFoundError.
- Multiple datasets can be stored and retrieved by filename within the
  same session.
"""

import uuid

import pandas as pd
import pytest

from app.exceptions import SessionNotFoundError
from app.models import ColumnSchema, Exchange
from app.session_store import SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_store() -> SessionStore:
    """Return a fresh SessionStore with a max history of 20 for all tests."""
    return SessionStore(max_history=20)


def make_exchange(index: int) -> Exchange:
    """Return a deterministic Exchange with a recognisable content string."""
    return Exchange(role="user", content=f"message-{index}")


def make_df_and_schema(
    rows: int = 5,
) -> tuple[pd.DataFrame, list[ColumnSchema]]:
    """Return a small DataFrame and matching ColumnSchema list."""
    df = pd.DataFrame({"value": range(rows), "label": [f"item-{i}" for i in range(rows)]})
    schema = [
        ColumnSchema(name="value", dtype="numeric", null_count=0),
        ColumnSchema(name="label", dtype="categorical", null_count=0),
    ]
    return df, schema


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_returns_string(self):
        store = make_store()
        session_id = store.create_session()
        assert isinstance(session_id, str)

    def test_returns_valid_uuid(self):
        store = make_store()
        session_id = store.create_session()
        # uuid.UUID raises ValueError for invalid strings.
        parsed = uuid.UUID(session_id)
        assert str(parsed) == session_id

    def test_each_call_returns_unique_id(self):
        store = make_store()
        ids = {store.create_session() for _ in range(10)}
        assert len(ids) == 10

    def test_created_session_is_retrievable(self):
        store = make_store()
        session_id = store.create_session()
        session = store.get_session(session_id)
        assert session.session_id == session_id

    def test_new_session_has_empty_datasets(self):
        store = make_store()
        session_id = store.create_session()
        session = store.get_session(session_id)
        assert session.datasets == {}
        assert session.dataframes == {}

    def test_new_session_has_empty_history(self):
        store = make_store()
        session_id = store.create_session()
        history = store.get_history(session_id)
        assert history == []


# ---------------------------------------------------------------------------
# get_session
# ---------------------------------------------------------------------------


class TestGetSession:
    def test_raises_for_unknown_id(self):
        store = make_store()
        with pytest.raises(SessionNotFoundError):
            store.get_session("nonexistent-id")

    def test_raises_for_random_uuid_that_was_never_created(self):
        store = make_store()
        with pytest.raises(SessionNotFoundError):
            store.get_session(str(uuid.uuid4()))

    def test_error_message_contains_session_id(self):
        store = make_store()
        bad_id = "missing-session-123"
        with pytest.raises(SessionNotFoundError) as exc_info:
            store.get_session(bad_id)
        assert bad_id in str(exc_info.value)

    def test_returns_correct_session(self):
        store = make_store()
        id_a = store.create_session()
        id_b = store.create_session()
        assert store.get_session(id_a).session_id == id_a
        assert store.get_session(id_b).session_id == id_b


# ---------------------------------------------------------------------------
# delete_session
# ---------------------------------------------------------------------------


class TestDeleteSession:
    def test_delete_existing_session(self):
        store = make_store()
        session_id = store.create_session()
        store.delete_session(session_id)
        with pytest.raises(SessionNotFoundError):
            store.get_session(session_id)

    def test_delete_unknown_session_raises(self):
        store = make_store()
        with pytest.raises(SessionNotFoundError):
            store.delete_session("never-existed")

    def test_delete_one_does_not_affect_other(self):
        store = make_store()
        id_a = store.create_session()
        id_b = store.create_session()
        store.delete_session(id_a)
        # id_b must still be accessible.
        session_b = store.get_session(id_b)
        assert session_b.session_id == id_b

    def test_subsequent_get_raises_after_delete(self):
        store = make_store()
        session_id = store.create_session()
        store.delete_session(session_id)
        with pytest.raises(SessionNotFoundError):
            store.get_session(session_id)


# ---------------------------------------------------------------------------
# add_dataset and multi-dataset retrieval
# ---------------------------------------------------------------------------


class TestAddDataset:
    def test_single_dataset_stored_and_retrievable(self):
        store = make_store()
        session_id = store.create_session()
        df, schema = make_df_and_schema(rows=5)
        store.add_dataset(session_id, "test.csv", df, schema)

        session = store.get_session(session_id)
        assert "test.csv" in session.datasets
        assert "test.csv" in session.dataframes

    def test_dataset_record_has_correct_row_count(self):
        store = make_store()
        session_id = store.create_session()
        df, schema = make_df_and_schema(rows=10)
        store.add_dataset(session_id, "data.csv", df, schema)

        record = store.get_session(session_id).datasets["data.csv"]
        assert record.row_count == 10

    def test_dataset_record_stores_schema(self):
        store = make_store()
        session_id = store.create_session()
        df, schema = make_df_and_schema()
        store.add_dataset(session_id, "data.csv", df, schema)

        record = store.get_session(session_id).datasets["data.csv"]
        assert len(record.column_schemas) == len(schema)
        assert record.column_schemas[0].name == schema[0].name

    def test_dataframe_shape_preserved(self):
        store = make_store()
        session_id = store.create_session()
        df, schema = make_df_and_schema(rows=7)
        store.add_dataset(session_id, "frame.csv", df, schema)

        stored_df = store.get_session(session_id).dataframes["frame.csv"]
        assert stored_df.shape == df.shape

    def test_multiple_datasets_stored_independently(self):
        store = make_store()
        session_id = store.create_session()

        df_a, schema_a = make_df_and_schema(rows=3)
        df_b, schema_b = make_df_and_schema(rows=8)
        df_c, schema_c = make_df_and_schema(rows=15)

        store.add_dataset(session_id, "a.csv", df_a, schema_a)
        store.add_dataset(session_id, "b.csv", df_b, schema_b)
        store.add_dataset(session_id, "c.csv", df_c, schema_c)

        session = store.get_session(session_id)
        assert set(session.datasets.keys()) == {"a.csv", "b.csv", "c.csv"}
        assert session.datasets["a.csv"].row_count == 3
        assert session.datasets["b.csv"].row_count == 8
        assert session.datasets["c.csv"].row_count == 15

    def test_multiple_datasets_dataframes_retrievable_by_filename(self):
        store = make_store()
        session_id = store.create_session()

        df_a = pd.DataFrame({"x": [1, 2, 3]})
        df_b = pd.DataFrame({"y": [10, 20]})
        schema_a = [ColumnSchema(name="x", dtype="numeric", null_count=0)]
        schema_b = [ColumnSchema(name="y", dtype="numeric", null_count=0)]

        store.add_dataset(session_id, "a.csv", df_a, schema_a)
        store.add_dataset(session_id, "b.csv", df_b, schema_b)

        session = store.get_session(session_id)
        assert session.dataframes["a.csv"].shape == (3, 1)
        assert session.dataframes["b.csv"].shape == (2, 1)

    def test_add_dataset_to_unknown_session_raises(self):
        store = make_store()
        df, schema = make_df_and_schema()
        with pytest.raises(SessionNotFoundError):
            store.add_dataset("no-such-session", "data.csv", df, schema)

    def test_overwrite_existing_filename(self):
        """Re-uploading the same filename replaces the previous dataset."""
        store = make_store()
        session_id = store.create_session()

        df_old, schema = make_df_and_schema(rows=3)
        df_new, schema = make_df_and_schema(rows=9)

        store.add_dataset(session_id, "same.csv", df_old, schema)
        store.add_dataset(session_id, "same.csv", df_new, schema)

        record = store.get_session(session_id).datasets["same.csv"]
        assert record.row_count == 9


# ---------------------------------------------------------------------------
# append_history and get_history — cap enforcement
# ---------------------------------------------------------------------------


class TestHistoryCap:
    def test_history_empty_initially(self):
        store = make_store()
        session_id = store.create_session()
        assert store.get_history(session_id) == []

    def test_append_single_exchange(self):
        store = make_store()
        session_id = store.create_session()
        exchange = make_exchange(0)
        store.append_history(session_id, exchange)
        history = store.get_history(session_id)
        assert len(history) == 1
        assert history[0].content == "message-0"

    def test_append_up_to_cap_stores_all(self):
        store = make_store()
        session_id = store.create_session()
        for i in range(20):
            store.append_history(session_id, make_exchange(i))
        history = store.get_history(session_id)
        assert len(history) == 20

    def test_append_21_returns_exactly_20(self):
        """Core acceptance criterion: 21 appends → 20 in history."""
        store = make_store()
        session_id = store.create_session()
        for i in range(21):
            store.append_history(session_id, make_exchange(i))
        history = store.get_history(session_id)
        assert len(history) == 20

    def test_append_21_first_exchange_is_gone(self):
        """The 1st-appended exchange (index 0) must not appear after 21 appends."""
        store = make_store()
        session_id = store.create_session()
        for i in range(21):
            store.append_history(session_id, make_exchange(i))
        history = store.get_history(session_id)
        contents = [e.content for e in history]
        assert "message-0" not in contents

    def test_append_21_most_recent_20_are_retained(self):
        """Exchanges 1–20 (indices 1..20) must all be present."""
        store = make_store()
        session_id = store.create_session()
        for i in range(21):
            store.append_history(session_id, make_exchange(i))
        history = store.get_history(session_id)
        contents = [e.content for e in history]
        for i in range(1, 21):
            assert f"message-{i}" in contents

    def test_large_overflow_still_caps_at_20(self):
        """Appending many more than 20 exchanges still yields exactly 20."""
        store = make_store()
        session_id = store.create_session()
        for i in range(100):
            store.append_history(session_id, make_exchange(i))
        history = store.get_history(session_id)
        assert len(history) == 20

    def test_large_overflow_retains_last_20(self):
        """After 100 appends, only the last 20 messages are kept."""
        store = make_store()
        session_id = store.create_session()
        for i in range(100):
            store.append_history(session_id, make_exchange(i))
        history = store.get_history(session_id)
        contents = [e.content for e in history]
        for i in range(80, 100):
            assert f"message-{i}" in contents

    def test_get_history_returns_chronological_order(self):
        store = make_store()
        session_id = store.create_session()
        for i in range(5):
            store.append_history(session_id, make_exchange(i))
        history = store.get_history(session_id)
        assert [e.content for e in history] == [f"message-{i}" for i in range(5)]

    def test_get_history_unknown_session_raises(self):
        store = make_store()
        with pytest.raises(SessionNotFoundError):
            store.get_history("nonexistent-id")

    def test_append_history_unknown_session_raises(self):
        store = make_store()
        with pytest.raises(SessionNotFoundError):
            store.append_history("nonexistent-id", make_exchange(0))

    def test_get_history_returns_copy(self):
        """Mutating the returned list must not affect the stored history."""
        store = make_store()
        session_id = store.create_session()
        store.append_history(session_id, make_exchange(0))
        history = store.get_history(session_id)
        history.clear()
        # Original history should still have 1 entry.
        assert len(store.get_history(session_id)) == 1


# ---------------------------------------------------------------------------
# build_message_history — task 4.1
# ---------------------------------------------------------------------------


class TestBuildMessageHistory:
    def test_empty_history_returns_empty_list(self):
        store = make_store()
        session_id = store.create_session()
        result = store.build_message_history(session_id)
        assert result == []

    def test_returns_list_of_dicts(self):
        store = make_store()
        session_id = store.create_session()
        store.append_history(session_id, Exchange(role="user", content="hello"))
        result = store.build_message_history(session_id)
        assert isinstance(result, list)
        assert isinstance(result[0], dict)

    def test_dict_has_role_and_content_keys(self):
        store = make_store()
        session_id = store.create_session()
        store.append_history(session_id, Exchange(role="user", content="hi"))
        msg = store.build_message_history(session_id)[0]
        assert set(msg.keys()) == {"role", "content"}

    def test_role_and_content_values_match_exchange(self):
        store = make_store()
        session_id = store.create_session()
        store.append_history(session_id, Exchange(role="assistant", content="hello back"))
        msg = store.build_message_history(session_id)[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "hello back"

    def test_returns_at_most_20_messages(self):
        """Acceptance criterion: build_message_history returns at most 20 message dicts."""
        store = make_store()
        session_id = store.create_session()
        for i in range(25):
            store.append_history(session_id, make_exchange(i))
        result = store.build_message_history(session_id)
        assert len(result) <= 20

    def test_returns_exactly_20_after_overflow(self):
        store = make_store()
        session_id = store.create_session()
        for i in range(25):
            store.append_history(session_id, make_exchange(i))
        result = store.build_message_history(session_id)
        assert len(result) == 20

    def test_chronological_order_oldest_first(self):
        """Acceptance criterion: message order is chronological (oldest first)."""
        store = make_store()
        session_id = store.create_session()
        for i in range(5):
            store.append_history(session_id, Exchange(role="user", content=f"msg-{i}"))
        result = store.build_message_history(session_id)
        contents = [m["content"] for m in result]
        assert contents == [f"msg-{i}" for i in range(5)]

    def test_chronological_order_preserved_after_overflow(self):
        """After overflow the retained messages are still in chronological order."""
        store = make_store()
        session_id = store.create_session()
        for i in range(25):
            store.append_history(session_id, make_exchange(i))
        result = store.build_message_history(session_id)
        # The 20 retained messages should be messages 5–24 in order.
        expected_contents = [f"message-{i}" for i in range(5, 25)]
        actual_contents = [m["content"] for m in result]
        assert actual_contents == expected_contents

    def test_maps_mixed_roles_correctly(self):
        store = make_store()
        session_id = store.create_session()
        store.append_history(session_id, Exchange(role="user", content="question"))
        store.append_history(session_id, Exchange(role="assistant", content="answer"))
        result = store.build_message_history(session_id)
        assert result[0] == {"role": "user", "content": "question"}
        assert result[1] == {"role": "assistant", "content": "answer"}

    def test_unknown_session_raises(self):
        store = make_store()
        with pytest.raises(SessionNotFoundError):
            store.build_message_history("nonexistent-id")

    def test_returns_independent_copy(self):
        """Mutating the returned list must not affect the stored history."""
        store = make_store()
        session_id = store.create_session()
        store.append_history(session_id, make_exchange(0))
        result = store.build_message_history(session_id)
        result.clear()
        assert len(store.build_message_history(session_id)) == 1


# ---------------------------------------------------------------------------
# reset_session_history — task 4.1
# ---------------------------------------------------------------------------


class TestResetSessionHistory:
    def test_reset_clears_history(self):
        """Acceptance criterion: after reset_session_history, build_message_history returns []."""
        store = make_store()
        session_id = store.create_session()
        for i in range(5):
            store.append_history(session_id, make_exchange(i))
        store.reset_session_history(session_id)
        assert store.build_message_history(session_id) == []

    def test_get_history_also_empty_after_reset(self):
        store = make_store()
        session_id = store.create_session()
        store.append_history(session_id, make_exchange(0))
        store.reset_session_history(session_id)
        assert store.get_history(session_id) == []

    def test_reset_on_already_empty_history_is_idempotent(self):
        store = make_store()
        session_id = store.create_session()
        store.reset_session_history(session_id)
        assert store.build_message_history(session_id) == []

    def test_reset_unknown_session_raises(self):
        store = make_store()
        with pytest.raises(SessionNotFoundError):
            store.reset_session_history("nonexistent-id")

    def test_new_appends_work_after_reset(self):
        """After a reset, the session can accumulate history again normally."""
        store = make_store()
        session_id = store.create_session()
        for i in range(5):
            store.append_history(session_id, make_exchange(i))
        store.reset_session_history(session_id)
        store.append_history(session_id, Exchange(role="user", content="fresh-start"))
        result = store.build_message_history(session_id)
        assert len(result) == 1
        assert result[0]["content"] == "fresh-start"

    def test_reset_does_not_affect_other_sessions(self):
        store = make_store()
        id_a = store.create_session()
        id_b = store.create_session()
        for i in range(3):
            store.append_history(id_a, make_exchange(i))
            store.append_history(id_b, make_exchange(i))
        store.reset_session_history(id_a)
        # Session A is empty; session B is untouched.
        assert store.build_message_history(id_a) == []
        assert len(store.build_message_history(id_b)) == 3

    def test_reset_does_not_delete_session(self):
        """reset_session_history clears history but keeps the session alive."""
        store = make_store()
        session_id = store.create_session()
        store.append_history(session_id, make_exchange(0))
        store.reset_session_history(session_id)
        # Session must still exist — no exception raised.
        session = store.get_session(session_id)
        assert session.session_id == session_id

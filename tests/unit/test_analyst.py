"""
Unit tests for app/analyst.py — Analyst Orchestrator (Task 7.1).

All LLM calls are mocked via unittest.mock.patch so no real OpenAI
API calls are made during testing.

Test cases
----------
1. Happy path — LLM returns a tool call; tool executes; QueryResponse is assembled.
2. No tool invoked — LLM returns only text content; reasoning_trace reflects
   that no computation was performed.
3. LLM error — error field is set; QueryResponse.answer contains the error;
   reasoning_trace.computation_description describes the failure.
4. Exchange appended — after every call, session history grows by 2 entries
   (user + assistant).
5. Schema context is prepended — system message is first in the message list
   forwarded to the LLM.
6. Raw DataFrame rows are never passed to the LLM — system message contains
   only schema summary text, not any row data.
7. Unknown tool returned by LLM — UnknownToolError is caught, not re-raised.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.analyst import analyse
from app.llm_client import LLMResponse, ToolCall
from app.models import ColumnSchema, Exchange, QueryResponse
from app.session_store import SessionStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store() -> SessionStore:
    """Return a fresh, isolated SessionStore (not the module-level singleton)."""
    return SessionStore(max_history=20)


@pytest.fixture()
def session_id(store: SessionStore) -> str:
    """Create a session in *store* and return its ID."""
    return store.create_session()


@pytest.fixture()
def session_with_dataset(store: SessionStore, session_id: str) -> str:
    """Add a small CSV-like DataFrame to the session and return the session ID."""
    df = pd.DataFrame(
        {
            "region": ["North", "South", "East"],
            "revenue": [100.0, 200.0, 150.0],
        }
    )
    schema = [
        ColumnSchema(name="region", dtype="categorical", null_count=0),
        ColumnSchema(name="revenue", dtype="numeric", null_count=0),
    ]
    store.add_dataset(session_id, "sales.csv", df, schema)
    return session_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(coro):
    """Run an async coroutine in a synchronous test context."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_tool_call(name: str, arguments: dict | None = None) -> ToolCall:
    return ToolCall(id="call_abc123", name=name, arguments=arguments or {})


# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------
# We patch `app.analyst.call_llm` (the name bound in analyst.py's namespace)
# and `app.analyst.session_store` so tests use an isolated store.
LLM_PATCH = "app.analyst.call_llm"
STORE_PATCH = "app.analyst.session_store"


# ---------------------------------------------------------------------------
# Test 1: Happy path — tool call dispatched, QueryResponse assembled
# ---------------------------------------------------------------------------


def test_happy_path_with_tool_call(store: SessionStore, session_with_dataset: str):
    """LLM returns a dataset_profile tool call; response includes non-empty answer."""
    session_id = session_with_dataset

    fake_profile_result = {
        "row_count": 3,
        "column_count": 2,
        "columns": [
            {"name": "region", "dtype": "object", "null_count": 0},
            {"name": "revenue", "dtype": "float64", "null_count": 0},
        ],
        "stats": {
            "revenue": {
                "mean": 150.0,
                "median": 150.0,
                "min": 100.0,
                "max": 200.0,
                "std": 50.0,
                "unavailable": False,
            }
        },
    }

    llm_resp = LLMResponse(
        content=None,
        tool_calls=[_make_tool_call("dataset_profile", {})],
    )

    with (
        patch(LLM_PATCH, return_value=llm_resp),
        patch(STORE_PATCH, store),
        patch("app.analyst.dispatch", return_value=fake_profile_result) as mock_dispatch,
    ):
        response: QueryResponse = run(
            analyse(session_id, "Profile the dataset", "sales.csv")
        )

    assert isinstance(response, QueryResponse)
    assert response.answer  # non-empty
    assert response.reasoning_trace is not None
    assert "dataset_profile" in response.reasoning_trace.tools_selected
    mock_dispatch.assert_called_once()


# ---------------------------------------------------------------------------
# Test 2: No tool invoked — LLM returns plain text only
# ---------------------------------------------------------------------------


def test_no_tool_invoked(store: SessionStore, session_id: str):
    """When no tool_calls are returned and the LLM content does not contain
    ambiguity signals, the answer is the unresolvable failure message and
    computation_description reflects that no resolution was possible."""
    llm_resp = LLMResponse(
        content="I need more information to answer that.",
        tool_calls=[],
    )

    with (
        patch(LLM_PATCH, return_value=llm_resp),
        patch(STORE_PATCH, store),
    ):
        response: QueryResponse = run(
            analyse(session_id, "What is the weather today?", None)
        )

    assert response.answer  # non-empty
    assert response.reasoning_trace.tools_selected == []
    assert "could not be resolved" in response.reasoning_trace.computation_description.lower()


# ---------------------------------------------------------------------------
# Test 3: LLM error — error QueryResponse returned
# ---------------------------------------------------------------------------


def test_llm_error_returns_error_response(store: SessionStore, session_id: str):
    """When LLMResponse.error is set, analyse returns a QueryResponse
    with an answer describing the error and an appropriate reasoning_trace."""
    llm_resp = LLMResponse(
        content=None,
        tool_calls=[],
        error="LLM API error: service unavailable",
    )

    with (
        patch(LLM_PATCH, return_value=llm_resp),
        patch(STORE_PATCH, store),
    ):
        response: QueryResponse = run(
            analyse(session_id, "Summarise the data", None)
        )

    assert response.answer  # non-empty
    assert "unable" in response.answer.lower() or "error" in response.answer.lower()
    assert response.reasoning_trace is not None
    assert "LLM call failed" in response.reasoning_trace.computation_description


# ---------------------------------------------------------------------------
# Test 4: Exchange appended after every call
# ---------------------------------------------------------------------------


def test_exchange_appended_to_history(store: SessionStore, session_id: str):
    """After analyse returns, two exchanges (user + assistant) are in history.
    When the LLM returns non-empty content and no tool call, Fix C preserves
    that content as the answer rather than replacing it with a failure message.
    """
    llm_resp = LLMResponse(content="Here is my answer.", tool_calls=[])

    with (
        patch(LLM_PATCH, return_value=llm_resp),
        patch(STORE_PATCH, store),
    ):
        run(analyse(session_id, "Hello", None))

    history = store.get_history(session_id)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "Hello"
    assert history[1].role == "assistant"
    # Fix C: LLM content is preserved, not replaced with failure message
    assert history[1].content == "Here is my answer."


# ---------------------------------------------------------------------------
# Test 5: Schema context is the first (system) message sent to the LLM
# ---------------------------------------------------------------------------


def test_schema_context_prepended_as_system_message(
    store: SessionStore, session_with_dataset: str
):
    """The system message is the schema context; raw DataFrame rows are absent."""
    session_id = session_with_dataset
    captured_messages: list[list[dict]] = []

    def fake_llm(messages, tools=None):
        captured_messages.append(messages)
        return LLMResponse(content="ok", tool_calls=[])

    with (
        patch(LLM_PATCH, side_effect=fake_llm),
        patch(STORE_PATCH, store),
    ):
        run(analyse(session_id, "What columns exist?", "sales.csv"))

    assert captured_messages, "call_llm was never called"
    messages = captured_messages[0]
    assert messages[0]["role"] == "system"
    # Schema context mentions the filename and column names
    system_content = messages[0]["content"]
    assert "sales.csv" in system_content
    assert "revenue" in system_content


# ---------------------------------------------------------------------------
# Test 6: Raw DataFrame rows are never passed to the LLM
# ---------------------------------------------------------------------------


def test_raw_dataframe_rows_never_in_messages(
    store: SessionStore, session_with_dataset: str
):
    """Verify that no raw row values (e.g., 'North', '100.0') appear in the
    system message — only schema-level information is included."""
    session_id = session_with_dataset
    captured_messages: list[list[dict]] = []

    def fake_llm(messages, tools=None):
        captured_messages.append(messages)
        return LLMResponse(content="ok", tool_calls=[])

    with (
        patch(LLM_PATCH, side_effect=fake_llm),
        patch(STORE_PATCH, store),
    ):
        run(analyse(session_id, "Profile the data", "sales.csv"))

    messages = captured_messages[0]
    system_content = messages[0]["content"]

    # Row-level values should NOT appear in the system (schema) message.
    # "North", "South", "East" are row values, not column names.
    assert "North" not in system_content
    assert "South" not in system_content
    assert "East" not in system_content


# ---------------------------------------------------------------------------
# Test 7: Unknown tool returned by LLM — error is caught, not re-raised
# ---------------------------------------------------------------------------


def test_unknown_tool_is_caught_gracefully(store: SessionStore, session_id: str):
    """UnknownToolError for a tool name not in the registry is caught;
    analyse returns a QueryResponse rather than raising."""
    from app.exceptions import UnknownToolError

    llm_resp = LLMResponse(
        content=None,
        tool_calls=[_make_tool_call("nonexistent_tool", {})],
    )

    with (
        patch(LLM_PATCH, return_value=llm_resp),
        patch(STORE_PATCH, store),
        patch(
            "app.analyst.dispatch",
            side_effect=UnknownToolError("Unknown tool 'nonexistent_tool'"),
        ),
    ):
        # Should not raise; should return a QueryResponse
        response: QueryResponse = run(
            analyse(session_id, "Do something weird", None)
        )

    assert isinstance(response, QueryResponse)
    assert response.answer  # non-empty even when the tool failed
    assert response.reasoning_trace is not None


# ---------------------------------------------------------------------------
# Test 8: reasoning_trace.tools_selected lists all dispatched tools
# ---------------------------------------------------------------------------


def test_tools_selected_lists_all_dispatched_tools(
    store: SessionStore, session_with_dataset: str
):
    """tools_selected in reasoning_trace contains every tool that was invoked."""
    session_id = session_with_dataset

    llm_resp = LLMResponse(
        content=None,
        tool_calls=[
            _make_tool_call("dataset_profile", {}),
            _make_tool_call("aggregate_data", {"group_by": "region", "agg_col": "revenue", "agg_fn": "sum"}),
        ],
    )

    fake_results = [{"row_count": 3}, {"results": [{"group_value": "North", "result": 100}]}]
    call_count = 0

    def fake_dispatch(name, args, session):
        nonlocal call_count
        result = fake_results[call_count]
        call_count += 1
        return result

    with (
        patch(LLM_PATCH, return_value=llm_resp),
        patch(STORE_PATCH, store),
        patch("app.analyst.dispatch", side_effect=fake_dispatch),
    ):
        response: QueryResponse = run(
            analyse(session_id, "Profile and aggregate", "sales.csv")
        )

    assert "dataset_profile" in response.reasoning_trace.tools_selected
    assert "aggregate_data" in response.reasoning_trace.tools_selected
    assert len(response.reasoning_trace.tools_selected) == 2


# ---------------------------------------------------------------------------
# Test 9: Conversation history is included in the LLM messages
# ---------------------------------------------------------------------------


def test_conversation_history_included_in_messages(
    store: SessionStore, session_id: str
):
    """Prior exchanges are included in the message list before the current query."""
    # Pre-populate history
    store.append_history(session_id, Exchange(role="user", content="First question"))
    store.append_history(session_id, Exchange(role="assistant", content="First answer"))

    captured_messages: list[list[dict]] = []

    def fake_llm(messages, tools=None):
        captured_messages.append(messages)
        return LLMResponse(content="Second answer", tool_calls=[])

    with (
        patch(LLM_PATCH, side_effect=fake_llm),
        patch(STORE_PATCH, store),
    ):
        run(analyse(session_id, "Second question", None))

    messages = captured_messages[0]
    roles = [m["role"] for m in messages]
    # system + user (history) + assistant (history) + user (current)
    assert roles == ["system", "user", "assistant", "user"]
    assert messages[-1]["content"] == "Second question"


# ---------------------------------------------------------------------------
# Task 7.3 — Missing Column and Ambiguous Query Handling
# ---------------------------------------------------------------------------


# Test 10: Missing column — error listing available columns, tools_selected=[]
def test_missing_column_returns_error_with_available_columns(
    store: SessionStore, session_with_dataset: str
):
    """A query referencing a quoted column that does not exist returns an error
    message listing available columns; tools_selected is empty."""
    session_id = session_with_dataset

    with patch(STORE_PATCH, store):
        response: QueryResponse = run(
            analyse(session_id, 'Show me the "nonexistent_column" values', "sales.csv")
        )

    assert isinstance(response, QueryResponse)
    # Answer should mention the missing column situation
    assert "Column not found" in response.answer
    # Available columns should be listed in the answer
    assert "region" in response.answer
    assert "revenue" in response.answer
    # tools_selected must be empty
    assert response.reasoning_trace.tools_selected == []


# Test 11: Ambiguous query — LLM returns a clarifying question, not empty string
def test_ambiguous_query_returns_clarifying_question(
    store: SessionStore, session_id: str
):
    """When the LLM returns no tool_calls and the content contains an ambiguity
    signal, the answer is the LLM's clarifying question (not an empty string)."""
    clarifying_question = "Could you clarify which metric you would like to analyse?"
    llm_resp = LLMResponse(
        content=clarifying_question,
        tool_calls=[],
    )

    with (
        patch(LLM_PATCH, return_value=llm_resp),
        patch(STORE_PATCH, store),
    ):
        response: QueryResponse = run(
            analyse(session_id, "Show me something interesting", None)
        )

    assert response.answer == clarifying_question
    assert response.answer != ""
    assert response.reasoning_trace.tools_selected == []
    assert "ambiguous" in response.reasoning_trace.computation_description.lower()


# Test 12: Unresolvable query — failure message, not a clarifying question
def test_unresolvable_query_returns_failure_message(
    store: SessionStore, session_id: str
):
    """Fix C: when the LLM returns non-empty content and no tool call, that
    content is now passed through as the answer.  The hardcoded failure message
    only fires when content is also empty."""
    llm_resp = LLMResponse(
        content="I cannot find any relevant data for that request.",
        tool_calls=[],
    )

    with (
        patch(LLM_PATCH, return_value=llm_resp),
        patch(STORE_PATCH, store),
    ):
        response: QueryResponse = run(
            analyse(session_id, "What is the GDP of Mars?", None)
        )

    # Fix C: LLM content is preserved
    assert response.answer == "I cannot find any relevant data for that request."
    assert response.reasoning_trace.tools_selected == []


# ---------------------------------------------------------------------------
# Regression tests: tool context prefix in session history
# ---------------------------------------------------------------------------


class TestToolContextInHistory:
    """
    Verify that when a tool is invoked the assistant message stored in
    session history begins with '[Used tool(s): <name>]', giving the LLM
    explicit tool context on subsequent turns.
    """

    def test_history_contains_tool_prefix_when_tool_ran(
        self, store: SessionStore, session_with_dataset: str
    ):
        """When a tool is dispatched, the history assistant message must carry
        the '[Used tool(s): ...]' prefix so follow-up turns can resolve references."""
        session_id = session_with_dataset

        fake_agg_result = {
            "results": [{"group_value": "North", "result": 100.0}],
            "group_by": "region",
            "agg_col": "revenue",
            "agg_fn": "sum",
        }

        llm_resp = LLMResponse(
            content=None,
            tool_calls=[_make_tool_call("aggregate_data", {})],
        )

        with (
            patch(LLM_PATCH, return_value=llm_resp),
            patch(STORE_PATCH, store),
            patch("app.analyst.dispatch", return_value=fake_agg_result),
        ):
            run(analyse(session_id, "Revenue by region?", "sales.csv"))

        history = store.get_history(session_id)
        assert len(history) == 2
        assistant_msg = history[1].content
        # Must carry tool prefix
        assert assistant_msg.startswith("[Used tool(s): aggregate_data]"), (
            f"Expected tool prefix in history, got: {assistant_msg!r}"
        )
        # Must also contain the human-readable answer
        assert "Aggregation results" in assistant_msg or "North" in assistant_msg

    def test_history_no_prefix_when_no_tool_ran(
        self, store: SessionStore, session_id: str
    ):
        """When no tool is invoked the stored assistant message is plain text
        (no '[Used tool(s):]' prefix)."""
        llm_resp = LLMResponse(
            content="Could you clarify which column you mean?",
            tool_calls=[],
        )

        with (
            patch(LLM_PATCH, return_value=llm_resp),
            patch(STORE_PATCH, store),
        ):
            run(analyse(session_id, "What is interesting?", None))

        history = store.get_history(session_id)
        assistant_msg = history[1].content
        assert not assistant_msg.startswith("[Used tool(s):]"), (
            "No-tool response must NOT carry a tool prefix"
        )
        assert "[Used tool(s):" not in assistant_msg

    def test_api_response_answer_is_unchanged(
        self, store: SessionStore, session_with_dataset: str
    ):
        """The QueryResponse.answer returned to the API/UI must NOT contain the
        tool prefix — it is only added to the history copy."""
        session_id = session_with_dataset

        fake_profile_result = {
            "row_count": 3,
            "column_count": 2,
            "columns": [{"name": "region", "dtype": "categorical", "null_count": 0}],
            "stats": {},
        }

        llm_resp = LLMResponse(
            content=None,
            tool_calls=[_make_tool_call("dataset_profile", {})],
        )

        with (
            patch(LLM_PATCH, return_value=llm_resp),
            patch(STORE_PATCH, store),
            patch("app.analyst.dispatch", return_value=fake_profile_result),
        ):
            response: QueryResponse = run(
                analyse(session_id, "Profile the data", "sales.csv")
            )

        # API response must NOT contain the prefix
        assert not response.answer.startswith("[Used tool(s):"), (
            "QueryResponse.answer must not start with the tool prefix"
        )

    def test_multiple_tools_prefix_lists_all(
        self, store: SessionStore, session_with_dataset: str
    ):
        """When multiple tools run in one turn, all their names appear in the prefix."""
        session_id = session_with_dataset

        llm_resp = LLMResponse(
            content=None,
            tool_calls=[
                _make_tool_call("dataset_profile", {}),
                _make_tool_call("aggregate_data", {"group_by": "region", "agg_col": "revenue", "agg_fn": "sum"}),
            ],
        )

        fake_results = [
            {"row_count": 3, "column_count": 2, "columns": [], "stats": {}},
            {"results": [{"group_value": "North", "result": 100}], "group_by": "region", "agg_col": "revenue", "agg_fn": "sum"},
        ]
        call_count = 0

        def fake_dispatch(name, args, session):
            nonlocal call_count
            r = fake_results[call_count]
            call_count += 1
            return r

        with (
            patch(LLM_PATCH, return_value=llm_resp),
            patch(STORE_PATCH, store),
            patch("app.analyst.dispatch", side_effect=fake_dispatch),
        ):
            run(analyse(session_id, "Profile and aggregate", "sales.csv"))

        history = store.get_history(session_id)
        assistant_msg = history[1].content
        assert "dataset_profile" in assistant_msg
        assert "aggregate_data" in assistant_msg
        assert assistant_msg.startswith("[Used tool(s):")

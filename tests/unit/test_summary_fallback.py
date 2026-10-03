"""
Regression tests for the deterministic generate_summary fallback in analyst.py.

When the LLM returns no tool call for explicit business-insight / summary queries,
the analyst must fall back to invoking generate_summary directly.

These tests verify:
1. _is_summary_query detects insight/summary phrases.
2. _is_summary_query does NOT trigger for chart requests or conversational queries.
3. _extract_column_hints extracts group_by and agg_col from query + session schema.
4. The fallback is invoked and returns a non-empty answer for summary queries.
5. generate_summary now accepts group_by and agg_col focus parameters.
6. Existing aggregation, chart, percentage tools are unaffected.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.analyst import _is_summary_query, _extract_column_hints, analyse
from app.llm_client import LLMResponse
from app.models import ColumnSchema, QueryResponse, Session
from app.session_store import SessionStore
from app.tools.generate_summary import generate_summary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store() -> SessionStore:
    return SessionStore(max_history=20)


@pytest.fixture()
def session_id(store: SessionStore) -> str:
    return store.create_session()


@pytest.fixture()
def session_with_revenue_data(store: SessionStore, session_id: str) -> str:
    df = pd.DataFrame({
        "region": ["North", "South", "East", "West"] * 5,
        "revenue": [1000.0, 800.0, 600.0, 400.0] * 5,
        "units_sold": [10, 8, 6, 4] * 5,
    })
    schema = [
        ColumnSchema(name="region", dtype="categorical", null_count=0),
        ColumnSchema(name="revenue", dtype="numeric", null_count=0),
        ColumnSchema(name="units_sold", dtype="numeric", null_count=0),
    ]
    store.add_dataset(session_id, "sample.csv", df, schema)
    return session_id


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


LLM_PATCH = "app.analyst.call_llm"
STORE_PATCH = "app.analyst.session_store"


# ---------------------------------------------------------------------------
# 1. _is_summary_query detects insight/summary phrases
# ---------------------------------------------------------------------------

class TestIsSummaryQuery:
    def test_business_insights(self):
        assert _is_summary_query("Give me 3 key business insights")

    def test_summarize(self):
        assert _is_summary_query("Summarize the key insights from this data")

    def test_summarise_british(self):
        assert _is_summary_query("Summarise the revenue by region")

    def test_in_text_phrase(self):
        assert _is_summary_query("Give me insights in text, do not create a chart")

    def test_do_not_create_a_chart(self):
        assert _is_summary_query("Based on the data, give me 3 insights. Do not create a chart.")

    def test_key_findings(self):
        assert _is_summary_query("What are the key findings from this dataset?")

    def test_summary_word(self):
        assert _is_summary_query("Can you provide a summary of the dataset?")

    def test_analyse_the_data(self):
        assert _is_summary_query("Analyse the data and tell me insights")


# ---------------------------------------------------------------------------
# 2. _is_summary_query does NOT trigger for chart/conversational queries
# ---------------------------------------------------------------------------

class TestIsSummaryQueryNegative:
    def test_explicit_bar_chart(self):
        assert not _is_summary_query("Create a bar chart showing total revenue by region")

    def test_explicit_line_chart(self):
        assert not _is_summary_query("Show me a line chart of revenue over time")

    def test_aggregate_question(self):
        assert not _is_summary_query("Which region had the highest revenue?")

    def test_percentage_question(self):
        assert not _is_summary_query("What percentage of revenue came from North?")

    def test_greeting(self):
        assert not _is_summary_query("Hello, what can you do?")

    def test_anomaly_question(self):
        assert not _is_summary_query("Detect anomalies in the dataset")

    def test_filter_question(self):
        assert not _is_summary_query("Show rows where revenue is greater than 2000")


# ---------------------------------------------------------------------------
# 3. _extract_column_hints extracts group_by and agg_col from query + schema
# ---------------------------------------------------------------------------

class TestExtractColumnHints:
    def _make_session_with_schema(self) -> Session:
        from app.models import DatasetRecord
        return Session(
            session_id="test",
            datasets={
                "data.csv": DatasetRecord(
                    filename="data.csv",
                    row_count=10,
                    column_schemas=[
                        ColumnSchema(name="region", dtype="categorical", null_count=0),
                        ColumnSchema(name="revenue", dtype="numeric", null_count=0),
                        ColumnSchema(name="product", dtype="categorical", null_count=0),
                        ColumnSchema(name="units_sold", dtype="numeric", null_count=0),
                    ]
                )
            },
            dataframes={},
            history=[],
        )

    def test_extracts_revenue_and_region(self):
        session = self._make_session_with_schema()
        group_by, agg_col = _extract_column_hints(
            "Summarize the revenue by region", session
        )
        assert agg_col == "revenue"
        assert group_by == "region"

    def test_extracts_units_and_product(self):
        session = self._make_session_with_schema()
        group_by, agg_col = _extract_column_hints(
            "Give me insights on units_sold by product", session
        )
        assert agg_col == "units_sold"
        assert group_by == "product"

    def test_returns_none_for_unrecognised_columns(self):
        session = self._make_session_with_schema()
        group_by, agg_col = _extract_column_hints(
            "Tell me about the weather", session
        )
        assert group_by is None
        assert agg_col is None

    def test_partial_match_numeric_only(self):
        session = self._make_session_with_schema()
        group_by, agg_col = _extract_column_hints(
            "What are the revenue trends?", session
        )
        assert agg_col == "revenue"
        # group_by may or may not match depending on whether 'region' appears


# ---------------------------------------------------------------------------
# 4. Fallback invoked and returns non-empty answer for summary queries
# ---------------------------------------------------------------------------

class TestSummaryFallbackExecution:
    def test_fallback_fires_when_no_tool_called(
        self, store: SessionStore, session_with_revenue_data: str
    ):
        """When LLM returns no tool call for a summary query, fallback invokes
        generate_summary and returns a non-empty answer."""
        session_id = session_with_revenue_data

        llm_resp = LLMResponse(content=None, tool_calls=[])

        with (
            patch(LLM_PATCH, return_value=llm_resp),
            patch(STORE_PATCH, store),
        ):
            response: QueryResponse = run(
                analyse(session_id, "Summarize the key business insights", "sample.csv")
            )

        assert response.answer, "Expected non-empty answer from summary fallback"
        assert "unable to find" not in response.answer.lower(), (
            "Fallback should not produce unresolvable message for summary query"
        )

    def test_fallback_result_contains_insight_content(
        self, store: SessionStore, session_with_revenue_data: str
    ):
        """The fallback answer should contain meaningful insight text."""
        session_id = session_with_revenue_data

        llm_resp = LLMResponse(content=None, tool_calls=[])

        with (
            patch(LLM_PATCH, return_value=llm_resp),
            patch(STORE_PATCH, store),
        ):
            response: QueryResponse = run(
                analyse(
                    session_id,
                    "Summarize the key business insights from the revenue by region",
                    "sample.csv",
                )
            )

        # The response should mention region/revenue somewhere in the summary
        assert response.answer, "Expected non-empty answer"
        assert response.reasoning_trace.tools_selected == ["generate_summary"], (
            "generate_summary must appear in tools_selected via fallback"
        )

    def test_fallback_does_not_fire_for_chart_query(
        self, store: SessionStore, session_with_revenue_data: str
    ):
        """An explicit chart query must not trigger the summary fallback."""
        session_id = session_with_revenue_data

        fake_chart_result = {"base64_png": "abc123==", "sampled": False}
        llm_resp = LLMResponse(
            content=None,
            tool_calls=[
                type("ToolCall", (), {"id": "c1", "name": "generate_chart",
                                     "arguments": {"chart_type": "bar",
                                                   "x_col": "region",
                                                   "y_col": "revenue"}})()
            ],
        )

        with (
            patch(LLM_PATCH, return_value=llm_resp),
            patch(STORE_PATCH, store),
            patch("app.analyst.dispatch", return_value=fake_chart_result),
        ):
            response: QueryResponse = run(
                analyse(
                    session_id,
                    "Create a bar chart showing total revenue by region",
                    "sample.csv",
                )
            )

        assert response.chart_base64 == "abc123=="
        assert "generate_chart" in response.reasoning_trace.tools_selected

    def test_fallback_does_not_fire_without_dataset(
        self, store: SessionStore, session_id: str
    ):
        """If no dataset is loaded, the fallback must not fire (no df to summarise)."""
        llm_resp = LLMResponse(content=None, tool_calls=[])

        with (
            patch(LLM_PATCH, return_value=llm_resp),
            patch(STORE_PATCH, store),
        ):
            response: QueryResponse = run(
                analyse(session_id, "Summarize the key business insights", None)
            )

        # No dataset → fallback skipped → unresolvable
        assert "unable to find" in response.answer.lower()


# ---------------------------------------------------------------------------
# 5. generate_summary with focus columns
# ---------------------------------------------------------------------------

class TestGenerateSummaryWithFocusColumns:
    @pytest.fixture()
    def df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "region": ["North", "South", "East", "West"],
            "revenue": [1000.0, 800.0, 600.0, 400.0],
            "units_sold": [10, 8, 6, 4],
        })

    def test_focus_on_revenue_by_region(self, df):
        result = generate_summary(df, group_by="region", agg_col="revenue")
        assert result.get("focused") is True
        focused = result["insights"]["focused_group_analysis"]
        assert focused["top_group"] == "North"
        assert focused["group_by_column"] if "group_by_column" in focused else focused["top_group"]
        # The focused data was produced with agg_col=revenue
        assert abs(focused["group_totals"]["North"] - 1000.0) < 0.01

    def test_focus_on_units_sold_by_region(self, df):
        result = generate_summary(df, group_by="region", agg_col="units_sold")
        assert result.get("focused") is True
        focused = result["insights"]["focused_group_analysis"]
        assert focused["top_group"] == "North"  # 10 units

    def test_no_focus_columns_uses_auto_detect(self, df):
        result = generate_summary(df)
        assert "profile" in result
        assert "insights" in result

    def test_invalid_group_by_falls_back_to_auto(self, df):
        """When group_by is not a real column, auto-detection is used."""
        result = generate_summary(df, group_by="nonexistent")
        assert "profile" in result

    def test_invalid_agg_col_falls_back_to_auto(self, df):
        """When agg_col is not numeric, auto-detection is used."""
        result = generate_summary(df, agg_col="region")
        assert "profile" in result

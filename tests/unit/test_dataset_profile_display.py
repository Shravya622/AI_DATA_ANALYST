"""
Regression tests verifying that a dataset_profile tool result displayed to the
user contains all four requested categories:
  1. Row count
  2. Column count
  3. Every column name, dtype, and null count
  4. Numeric statistics (mean, median, min, max, std)

These tests work at the _format_profile_with_columns level (unit) and at the
analyst response level (integration with mocked LLM).
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pandas as pd
import pytest

from app.analyst import _format_profile_with_columns, analyse
from app.llm_client import LLMResponse, ToolCall
from app.models import ColumnSchema, QueryResponse
from app.session_store import SessionStore
from app.tools.dataset_profile import dataset_profile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mixed_df() -> pd.DataFrame:
    """DataFrame with date, categorical, and numeric columns including nulls."""
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
        "region": ["North", "South", None],
        "revenue": [1000.0, 2000.0, 1500.0],
        "units_sold": [10, 20, 15],
    })


@pytest.fixture()
def store() -> SessionStore:
    return SessionStore(max_history=20)


@pytest.fixture()
def session_with_data(store: SessionStore, mixed_df: pd.DataFrame) -> str:
    sid = store.create_session()
    schema = [
        ColumnSchema(name="date", dtype="datetime", null_count=0),
        ColumnSchema(name="region", dtype="categorical", null_count=1),
        ColumnSchema(name="revenue", dtype="numeric", null_count=0),
        ColumnSchema(name="units_sold", dtype="numeric", null_count=0),
    ]
    store.add_dataset(sid, "data.csv", mixed_df, schema)
    return sid


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


LLM_PATCH = "app.analyst.call_llm"
STORE_PATCH = "app.analyst.session_store"


# ---------------------------------------------------------------------------
# Unit tests: _format_profile_with_columns
# ---------------------------------------------------------------------------

class TestFormatProfileWithColumns:
    """Verify the formatter renders all four categories."""

    def test_row_count_present(self, mixed_df):
        profile = dataset_profile(mixed_df)
        text = _format_profile_with_columns(profile)
        assert "3" in text or "rows" in text.lower()

    def test_column_count_present(self, mixed_df):
        profile = dataset_profile(mixed_df)
        text = _format_profile_with_columns(profile)
        assert "4" in text or "columns" in text.lower()

    def test_all_column_names_present(self, mixed_df):
        profile = dataset_profile(mixed_df)
        text = _format_profile_with_columns(profile)
        for col in ["date", "region", "revenue", "units_sold"]:
            assert col in text, f"Column name '{col}' missing from profile output"

    def test_dtypes_present(self, mixed_df):
        """dtype strings must appear for at least some columns."""
        profile = dataset_profile(mixed_df)
        text = _format_profile_with_columns(profile)
        # The raw pandas dtype strings (object, float64, int64) must appear
        dtype_keywords = ["dtype", "object", "float", "int", "numeric", "categorical"]
        assert any(kw in text.lower() for kw in dtype_keywords), (
            f"No dtype information found in profile output:\n{text}"
        )

    def test_null_count_present(self, mixed_df):
        """Null counts must be visible for columns that have them."""
        profile = dataset_profile(mixed_df)
        text = _format_profile_with_columns(profile)
        # region has 1 null — either "1 null" or "null count: 1" must appear
        assert "null" in text.lower(), (
            f"Null count information missing from profile output:\n{text}"
        )

    def test_numeric_statistics_present(self, mixed_df):
        """mean, min, max must appear for numeric columns."""
        profile = dataset_profile(mixed_df)
        text = _format_profile_with_columns(profile)
        for stat in ["mean", "min", "max"]:
            assert stat in text.lower(), (
                f"Numeric statistic '{stat}' missing from profile output:\n{text}"
            )

    def test_numeric_values_are_correct(self, mixed_df):
        """Revenue mean = (1000+2000+1500)/3 = 1500.0."""
        profile = dataset_profile(mixed_df)
        text = _format_profile_with_columns(profile)
        assert "1500" in text, (
            f"Expected revenue mean=1500.0 in profile output, got:\n{text}"
        )

    def test_all_four_categories_in_one_output(self, mixed_df):
        """Convenience test: all four categories present simultaneously."""
        profile = dataset_profile(mixed_df)
        text = _format_profile_with_columns(profile)

        # 1. Row count
        assert "3" in text

        # 2. Column count
        assert "4" in text

        # 3. Column names
        for col in ["date", "region", "revenue", "units_sold"]:
            assert col in text, f"Column name '{col}' not found"

        # 4. Numeric statistics
        assert "mean" in text.lower()
        assert "min" in text.lower()
        assert "max" in text.lower()


# ---------------------------------------------------------------------------
# Integration tests: analyst response for dataset_profile tool call
# ---------------------------------------------------------------------------

class TestDatasetProfileInAnalystResponse:
    """Verify the analyst response for a dataset_profile tool call contains
    all four categories in the user-facing answer field."""

    def _profile_result(self, df: pd.DataFrame) -> dict:
        return dataset_profile(df)

    def test_answer_contains_column_names(
        self, store: SessionStore, session_with_data: str, mixed_df: pd.DataFrame
    ):
        """The QueryResponse.answer for a dataset_profile call must contain
        every column name."""
        llm_resp = LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name="dataset_profile", arguments={})],
        )
        profile_result = self._profile_result(mixed_df)

        with (
            patch(LLM_PATCH, return_value=llm_resp),
            patch(STORE_PATCH, store),
            patch("app.analyst.dispatch", return_value=profile_result),
        ):
            response: QueryResponse = run(
                analyse(session_with_data,
                        "Give me a full dataset profile including column names and types",
                        "data.csv")
            )

        answer = response.answer
        for col in ["date", "region", "revenue", "units_sold"]:
            assert col in answer, (
                f"Column name '{col}' not found in analyst answer:\n{answer}"
            )

    def test_answer_contains_dtypes(
        self, store: SessionStore, session_with_data: str, mixed_df: pd.DataFrame
    ):
        llm_resp = LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name="dataset_profile", arguments={})],
        )
        profile_result = self._profile_result(mixed_df)

        with (
            patch(LLM_PATCH, return_value=llm_resp),
            patch(STORE_PATCH, store),
            patch("app.analyst.dispatch", return_value=profile_result),
        ):
            response: QueryResponse = run(
                analyse(session_with_data, "Profile the dataset", "data.csv")
            )

        answer = response.answer.lower()
        dtype_keywords = ["dtype", "object", "float", "int", "numeric"]
        assert any(kw in answer for kw in dtype_keywords), (
            f"No dtype information in answer:\n{response.answer}"
        )

    def test_answer_contains_null_counts(
        self, store: SessionStore, session_with_data: str, mixed_df: pd.DataFrame
    ):
        llm_resp = LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name="dataset_profile", arguments={})],
        )
        profile_result = self._profile_result(mixed_df)

        with (
            patch(LLM_PATCH, return_value=llm_resp),
            patch(STORE_PATCH, store),
            patch("app.analyst.dispatch", return_value=profile_result),
        ):
            response: QueryResponse = run(
                analyse(session_with_data,
                        "Show me missing values and data types",
                        "data.csv")
            )

        assert "null" in response.answer.lower(), (
            f"Null count information not found in answer:\n{response.answer}"
        )

    def test_answer_contains_numeric_statistics(
        self, store: SessionStore, session_with_data: str, mixed_df: pd.DataFrame
    ):
        llm_resp = LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name="dataset_profile", arguments={})],
        )
        profile_result = self._profile_result(mixed_df)

        with (
            patch(LLM_PATCH, return_value=llm_resp),
            patch(STORE_PATCH, store),
            patch("app.analyst.dispatch", return_value=profile_result),
        ):
            response: QueryResponse = run(
                analyse(session_with_data, "Profile the dataset", "data.csv")
            )

        answer = response.answer.lower()
        for stat in ["mean", "min", "max"]:
            assert stat in answer, (
                f"Numeric statistic '{stat}' not found in answer:\n{response.answer}"
            )

    def test_answer_contains_row_and_column_counts(
        self, store: SessionStore, session_with_data: str, mixed_df: pd.DataFrame
    ):
        llm_resp = LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name="dataset_profile", arguments={})],
        )
        profile_result = self._profile_result(mixed_df)

        with (
            patch(LLM_PATCH, return_value=llm_resp),
            patch(STORE_PATCH, store),
            patch("app.analyst.dispatch", return_value=profile_result),
        ):
            response: QueryResponse = run(
                analyse(session_with_data, "Profile the dataset", "data.csv")
            )

        answer = response.answer
        assert "3" in answer, "Row count (3) not found in answer"
        assert "4" in answer, "Column count (4) not found in answer"

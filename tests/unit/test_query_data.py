"""
Unit tests for app/tools/query_data.py

Covers all acceptance criteria for task 5.2:
  - Filtering on a numeric column with > returns only matching rows.
  - Selecting specific columns returns only those columns.
  - limit caps the number of returned rows.
  - An unknown column name in filters or columns returns {"error": "column X not found"}.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.tools.query_data import query_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """A small mixed DataFrame used across most tests."""
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [25, 30, 22, 35, 28],
            "score": [88.5, 72.0, 95.0, 60.0, 81.5],
            "region": ["North", "South", "North", "East", "West"],
        }
    )


# ---------------------------------------------------------------------------
# Acceptance criterion 1 — numeric filter with > returns only matching rows
# ---------------------------------------------------------------------------

class TestFiltering:
    def test_gt_filter_returns_matching_rows(self, sample_df):
        result = query_data(sample_df, filters=[{"column": "age", "operator": ">", "value": 28}])
        assert "rows" in result
        assert all(row["age"] > 28 for row in result["rows"])
        assert len(result["rows"]) == 2  # Bob (30) and Diana (35)

    def test_gt_filter_excludes_equal_value(self, sample_df):
        result = query_data(sample_df, filters=[{"column": "age", "operator": ">", "value": 30}])
        assert all(row["age"] > 30 for row in result["rows"])

    def test_gte_filter_includes_equal_value(self, sample_df):
        result = query_data(sample_df, filters=[{"column": "age", "operator": ">=", "value": 30}])
        assert all(row["age"] >= 30 for row in result["rows"])
        assert len(result["rows"]) == 2  # Bob (30) and Diana (35)

    def test_lt_filter(self, sample_df):
        result = query_data(sample_df, filters=[{"column": "score", "operator": "<", "value": 80.0}])
        assert all(row["score"] < 80.0 for row in result["rows"])

    def test_lte_filter(self, sample_df):
        result = query_data(sample_df, filters=[{"column": "score", "operator": "<=", "value": 72.0}])
        assert all(row["score"] <= 72.0 for row in result["rows"])

    def test_eq_filter(self, sample_df):
        result = query_data(sample_df, filters=[{"column": "region", "operator": "==", "value": "North"}])
        assert all(row["region"] == "North" for row in result["rows"])
        assert len(result["rows"]) == 2

    def test_ne_filter(self, sample_df):
        result = query_data(sample_df, filters=[{"column": "region", "operator": "!=", "value": "North"}])
        assert all(row["region"] != "North" for row in result["rows"])
        assert len(result["rows"]) == 3

    def test_multiple_filters_are_applied_sequentially(self, sample_df):
        """Both filters must hold simultaneously for a row to be included."""
        result = query_data(
            sample_df,
            filters=[
                {"column": "age", "operator": ">", "value": 24},
                {"column": "score", "operator": ">=", "value": 80.0},
            ],
        )
        for row in result["rows"]:
            assert row["age"] > 24
            assert row["score"] >= 80.0

    def test_filter_no_matches_returns_empty_rows(self, sample_df):
        result = query_data(sample_df, filters=[{"column": "age", "operator": ">", "value": 100}])
        assert result["rows"] == []
        assert result["total_rows"] == 0

    def test_no_filters_returns_all_rows(self, sample_df):
        result = query_data(sample_df)
        assert len(result["rows"]) == len(sample_df)

    def test_none_filters_is_treated_as_no_filter(self, sample_df):
        result = query_data(sample_df, filters=None)
        assert len(result["rows"]) == len(sample_df)


# ---------------------------------------------------------------------------
# Acceptance criterion 2 — selecting specific columns
# ---------------------------------------------------------------------------

class TestColumnSelection:
    def test_select_single_column(self, sample_df):
        result = query_data(sample_df, columns=["name"])
        assert "rows" in result
        for row in result["rows"]:
            assert list(row.keys()) == ["name"]

    def test_select_multiple_columns(self, sample_df):
        result = query_data(sample_df, columns=["name", "age"])
        for row in result["rows"]:
            assert set(row.keys()) == {"name", "age"}

    def test_column_order_is_preserved(self, sample_df):
        result = query_data(sample_df, columns=["score", "name"])
        for row in result["rows"]:
            assert list(row.keys()) == ["score", "name"]

    def test_no_columns_returns_all_columns(self, sample_df):
        result = query_data(sample_df)
        for row in result["rows"]:
            assert set(row.keys()) == set(sample_df.columns)

    def test_none_columns_returns_all_columns(self, sample_df):
        result = query_data(sample_df, columns=None)
        for row in result["rows"]:
            assert set(row.keys()) == set(sample_df.columns)

    def test_filter_and_column_selection_combined(self, sample_df):
        result = query_data(
            sample_df,
            filters=[{"column": "age", "operator": ">", "value": 28}],
            columns=["name", "age"],
        )
        assert all(row["age"] > 28 for row in result["rows"])
        for row in result["rows"]:
            assert set(row.keys()) == {"name", "age"}


# ---------------------------------------------------------------------------
# Acceptance criterion 3 — limit caps returned rows
# ---------------------------------------------------------------------------

class TestLimit:
    def test_limit_caps_rows(self, sample_df):
        result = query_data(sample_df, limit=2)
        assert len(result["rows"]) == 2

    def test_limit_does_not_affect_total_rows(self, sample_df):
        """total_rows should reflect the count before limit is applied."""
        result = query_data(sample_df, limit=2)
        assert result["total_rows"] == len(sample_df)

    def test_limit_larger_than_dataset_returns_all(self, sample_df):
        result = query_data(sample_df, limit=1000)
        assert len(result["rows"]) == len(sample_df)

    def test_default_limit_is_100(self):
        """Default limit of 100 — dataset of 150 rows should yield 100 rows."""
        large_df = pd.DataFrame({"x": range(150)})
        result = query_data(large_df)
        assert len(result["rows"]) == 100
        assert result["total_rows"] == 150

    def test_limit_one_returns_first_row(self, sample_df):
        result = query_data(sample_df, limit=1)
        assert len(result["rows"]) == 1
        assert result["rows"][0]["name"] == "Alice"

    def test_limit_zero_returns_no_rows(self, sample_df):
        result = query_data(sample_df, limit=0)
        assert result["rows"] == []
        assert result["total_rows"] == len(sample_df)


# ---------------------------------------------------------------------------
# Acceptance criterion 4 — unknown column returns error dict
# ---------------------------------------------------------------------------

class TestUnknownColumn:
    def test_unknown_filter_column_returns_error(self, sample_df):
        result = query_data(
            sample_df,
            filters=[{"column": "salary", "operator": ">", "value": 50000}],
        )
        assert "error" in result
        assert "salary" in result["error"]

    def test_unknown_select_column_returns_error(self, sample_df):
        result = query_data(sample_df, columns=["name", "nonexistent"])
        assert "error" in result
        assert "nonexistent" in result["error"]

    def test_error_format_matches_spec(self, sample_df):
        """Error message must follow: column '<name>' not found."""
        result = query_data(
            sample_df,
            filters=[{"column": "missing_col", "operator": "==", "value": 1}],
        )
        assert result == {"error": "column 'missing_col' not found"}

    def test_error_format_for_columns_matches_spec(self, sample_df):
        result = query_data(sample_df, columns=["bad_col"])
        assert result == {"error": "column 'bad_col' not found"}

    def test_first_bad_filter_column_stops_execution(self, sample_df):
        """Only the first error is surfaced — no partial results."""
        result = query_data(
            sample_df,
            filters=[
                {"column": "bad1", "operator": "==", "value": 1},
                {"column": "bad2", "operator": "==", "value": 2},
            ],
        )
        assert "error" in result
        assert "bad1" in result["error"]


# ---------------------------------------------------------------------------
# Return structure sanity checks
# ---------------------------------------------------------------------------

class TestReturnStructure:
    def test_successful_call_has_rows_and_total_rows(self, sample_df):
        result = query_data(sample_df)
        assert "rows" in result
        assert "total_rows" in result

    def test_rows_are_list_of_dicts(self, sample_df):
        result = query_data(sample_df)
        assert isinstance(result["rows"], list)
        assert all(isinstance(row, dict) for row in result["rows"])

    def test_total_rows_matches_filtered_count(self, sample_df):
        result = query_data(
            sample_df,
            filters=[{"column": "age", "operator": ">", "value": 24}],
        )
        # Alice(25), Bob(30), Diana(35), Eve(28) → 4 rows
        assert result["total_rows"] == 4

    def test_empty_dataframe_returns_empty_rows(self):
        empty_df = pd.DataFrame({"a": pd.Series([], dtype=int), "b": pd.Series([], dtype=str)})
        result = query_data(empty_df)
        assert result["rows"] == []
        assert result["total_rows"] == 0

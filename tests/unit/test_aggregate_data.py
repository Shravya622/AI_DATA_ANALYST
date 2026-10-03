"""
Unit tests for app/tools/aggregate_data.py

Covers all acceptance criteria for task 5.3:
  - aggregate_data(df, "region", "revenue", "sum") returns correct per-region sums.
  - An unknown group_by column returns {"error": "column X not found"}.
  - An unknown agg_fn returns {"error": "unsupported aggregation function X"}.
  - Calling the function twice on the same unchanged DataFrame with the same
    parameters returns identical results (idempotence).
"""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from app.tools.aggregate_data import aggregate_data, SUPPORTED_AGG_FNS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sales_df() -> pd.DataFrame:
    """Small sales DataFrame that mirrors the sample fixture columns."""
    return pd.DataFrame(
        {
            "region": ["North", "South", "North", "East", "South", "East", "North"],
            "product": ["A", "B", "A", "C", "A", "B", "C"],
            "revenue": [100.0, 200.0, 150.0, 300.0, 250.0, 175.0, 125.0],
            "units_sold": [10, 20, 15, 30, 25, 17, 12],
        }
    )


# ---------------------------------------------------------------------------
# Acceptance criterion 1 — correct per-group aggregation results
# ---------------------------------------------------------------------------

class TestAggregationResults:
    def test_sum_per_region(self, sales_df):
        """aggregate_data(df, "region", "revenue", "sum") returns correct sums."""
        result = aggregate_data(sales_df, "region", "revenue", "sum")
        assert "results" in result
        # Build a lookup for easy assertion
        by_region = {r["group_value"]: r["result"] for r in result["results"]}
        assert by_region["North"] == pytest.approx(100.0 + 150.0 + 125.0)
        assert by_region["South"] == pytest.approx(200.0 + 250.0)
        assert by_region["East"] == pytest.approx(300.0 + 175.0)

    def test_mean_per_region(self, sales_df):
        result = aggregate_data(sales_df, "region", "revenue", "mean")
        by_region = {r["group_value"]: r["result"] for r in result["results"]}
        assert by_region["North"] == pytest.approx((100.0 + 150.0 + 125.0) / 3)
        assert by_region["South"] == pytest.approx((200.0 + 250.0) / 2)
        assert by_region["East"] == pytest.approx((300.0 + 175.0) / 2)

    def test_count_per_region(self, sales_df):
        result = aggregate_data(sales_df, "region", "revenue", "count")
        by_region = {r["group_value"]: r["result"] for r in result["results"]}
        assert by_region["North"] == 3
        assert by_region["South"] == 2
        assert by_region["East"] == 2

    def test_min_per_region(self, sales_df):
        result = aggregate_data(sales_df, "region", "revenue", "min")
        by_region = {r["group_value"]: r["result"] for r in result["results"]}
        assert by_region["North"] == pytest.approx(100.0)
        assert by_region["South"] == pytest.approx(200.0)
        assert by_region["East"] == pytest.approx(175.0)

    def test_max_per_region(self, sales_df):
        result = aggregate_data(sales_df, "region", "revenue", "max")
        by_region = {r["group_value"]: r["result"] for r in result["results"]}
        assert by_region["North"] == pytest.approx(150.0)
        assert by_region["South"] == pytest.approx(250.0)
        assert by_region["East"] == pytest.approx(300.0)

    def test_result_metadata_fields_present(self, sales_df):
        """Returned dict includes group_by, agg_col, agg_fn metadata."""
        result = aggregate_data(sales_df, "region", "revenue", "sum")
        assert result["group_by"] == "region"
        assert result["agg_col"] == "revenue"
        assert result["agg_fn"] == "sum"

    def test_results_is_list_of_dicts(self, sales_df):
        result = aggregate_data(sales_df, "region", "revenue", "sum")
        assert isinstance(result["results"], list)
        for item in result["results"]:
            assert "group_value" in item
            assert "result" in item

    def test_group_values_are_strings(self, sales_df):
        """group_value must always be a string even if source dtype is numeric."""
        df = pd.DataFrame({"year": [2021, 2022, 2021, 2022], "sales": [10, 20, 30, 40]})
        result = aggregate_data(df, "year", "sales", "sum")
        for item in result["results"]:
            assert isinstance(item["group_value"], str)

    def test_result_values_are_python_native(self, sales_df):
        """result values must be JSON-serialisable Python scalars, not numpy types."""
        import json
        result = aggregate_data(sales_df, "region", "revenue", "sum")
        # Should not raise
        json.dumps(result)

    def test_aggregation_on_integer_column(self, sales_df):
        result = aggregate_data(sales_df, "region", "units_sold", "sum")
        assert "results" in result
        by_region = {r["group_value"]: r["result"] for r in result["results"]}
        assert by_region["North"] == 10 + 15 + 12
        assert by_region["South"] == 20 + 25
        assert by_region["East"] == 30 + 17

    def test_single_group_returns_single_result(self):
        df = pd.DataFrame({"cat": ["A", "A", "A"], "val": [1.0, 2.0, 3.0]})
        result = aggregate_data(df, "cat", "val", "sum")
        assert len(result["results"]) == 1
        assert result["results"][0]["result"] == pytest.approx(6.0)

    def test_all_supported_agg_fns_succeed(self, sales_df):
        """Every supported aggregation function returns a results dict."""
        for fn in SUPPORTED_AGG_FNS:
            result = aggregate_data(sales_df, "region", "revenue", fn)
            assert "results" in result, f"agg_fn={fn!r} did not return 'results'"
            assert "error" not in result


# ---------------------------------------------------------------------------
# Acceptance criterion 2 — unknown group_by column returns error
# ---------------------------------------------------------------------------

class TestUnknownGroupByColumn:
    def test_unknown_group_by_returns_error(self, sales_df):
        result = aggregate_data(sales_df, "country", "revenue", "sum")
        assert "error" in result
        assert "country" in result["error"]

    def test_unknown_group_by_error_format(self, sales_df):
        result = aggregate_data(sales_df, "nonexistent", "revenue", "sum")
        assert result == {"error": "column nonexistent not found"}

    def test_no_results_key_on_group_by_error(self, sales_df):
        result = aggregate_data(sales_df, "missing_col", "revenue", "sum")
        assert "results" not in result

    def test_unknown_agg_col_returns_error(self, sales_df):
        result = aggregate_data(sales_df, "region", "profit", "sum")
        assert "error" in result
        assert "profit" in result["error"]

    def test_unknown_agg_col_error_format(self, sales_df):
        result = aggregate_data(sales_df, "region", "nonexistent_col", "sum")
        assert result == {"error": "column nonexistent_col not found"}


# ---------------------------------------------------------------------------
# Acceptance criterion 3 — unknown agg_fn returns error
# ---------------------------------------------------------------------------

class TestUnsupportedAggFn:
    def test_unknown_agg_fn_returns_error(self, sales_df):
        result = aggregate_data(sales_df, "region", "revenue", "median")
        assert "error" in result
        assert "median" in result["error"]

    def test_unknown_agg_fn_error_format(self, sales_df):
        result = aggregate_data(sales_df, "region", "revenue", "variance")
        assert result == {"error": "unsupported aggregation function variance"}

    def test_empty_string_agg_fn_returns_error(self, sales_df):
        result = aggregate_data(sales_df, "region", "revenue", "")
        assert "error" in result

    def test_case_sensitive_agg_fn(self, sales_df):
        """Aggregation function names are case-sensitive; 'Sum' is not valid."""
        result = aggregate_data(sales_df, "region", "revenue", "Sum")
        assert "error" in result

    def test_no_results_key_on_agg_fn_error(self, sales_df):
        result = aggregate_data(sales_df, "region", "revenue", "bad_fn")
        assert "results" not in result


# ---------------------------------------------------------------------------
# Acceptance criterion 4 — idempotence
# ---------------------------------------------------------------------------

class TestIdempotence:
    def test_same_call_twice_returns_identical_results(self, sales_df):
        """Calling aggregate_data twice with identical args produces identical output."""
        result1 = aggregate_data(sales_df, "region", "revenue", "sum")
        result2 = aggregate_data(sales_df, "region", "revenue", "sum")
        assert result1 == result2

    def test_idempotent_for_all_supported_fns(self, sales_df):
        for fn in SUPPORTED_AGG_FNS:
            r1 = aggregate_data(sales_df, "region", "revenue", fn)
            r2 = aggregate_data(sales_df, "region", "revenue", fn)
            assert r1 == r2, f"Not idempotent for agg_fn={fn!r}"

    def test_dataframe_not_mutated_by_call(self, sales_df):
        """The function must not alter the input DataFrame."""
        original_shape = sales_df.shape
        original_values = sales_df["revenue"].tolist()
        aggregate_data(sales_df, "region", "revenue", "sum")
        assert sales_df.shape == original_shape
        assert sales_df["revenue"].tolist() == original_values

    def test_repeated_calls_on_copy_are_consistent(self, sales_df):
        """Even when called on a deep copy, the results match."""
        df_copy = copy.deepcopy(sales_df)
        result1 = aggregate_data(sales_df, "region", "revenue", "mean")
        result2 = aggregate_data(df_copy, "region", "revenue", "mean")
        # Normalise result ordering before comparing
        def sort_results(r):
            return sorted(r["results"], key=lambda x: x["group_value"])
        assert sort_results(result1) == sort_results(result2)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_row_dataframe(self):
        df = pd.DataFrame({"cat": ["A"], "val": [42.0]})
        result = aggregate_data(df, "cat", "val", "sum")
        assert result["results"][0]["result"] == pytest.approx(42.0)

    def test_empty_dataframe_returns_empty_results(self):
        df = pd.DataFrame({"region": pd.Series([], dtype=str), "revenue": pd.Series([], dtype=float)})
        result = aggregate_data(df, "region", "revenue", "sum")
        assert "results" in result
        assert result["results"] == []

    def test_group_by_and_agg_col_same_column(self):
        """group_by and agg_col can reference the same column (count use-case)."""
        df = pd.DataFrame({"region": ["A", "A", "B", "B", "B"]})
        result = aggregate_data(df, "region", "region", "count")
        assert "results" in result
        by_group = {r["group_value"]: r["result"] for r in result["results"]}
        assert by_group["A"] == 2
        assert by_group["B"] == 3

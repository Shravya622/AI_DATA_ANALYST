"""
Unit tests for app/tools/dataset_profile.py

Acceptance criteria covered:
  AC-1  Returns correct row_count and column_count for any non-empty DataFrame.
  AC-2  All numeric columns have mean, median, min, max, std populated.
  AC-3  An all-null numeric column has each stat marked as "unavailable": True
        rather than omitted.
  AC-4  No statistic is derived from LLM output (verified structurally — the
        function is pure pandas computation with no external calls).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.tools.dataset_profile import dataset_profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_numeric_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "salary": [50_000.0, 60_000.0, 70_000.0, 80_000.0, 90_000.0],
        }
    )


def _make_mixed_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Carol"],
            "score": [88.5, 92.0, 76.3],
            "grade": ["A", "A", "B"],
        }
    )


# ---------------------------------------------------------------------------
# AC-1: row_count and column_count
# ---------------------------------------------------------------------------

class TestRowAndColumnCounts:
    def test_basic_counts(self) -> None:
        df = _make_numeric_df()
        result = dataset_profile(df)

        assert result["row_count"] == 5
        assert result["column_count"] == 2

    def test_single_row_single_column(self) -> None:
        df = pd.DataFrame({"value": [42]})
        result = dataset_profile(df)

        assert result["row_count"] == 1
        assert result["column_count"] == 1

    def test_single_row_many_columns(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4], "e": [5]})
        result = dataset_profile(df)

        assert result["row_count"] == 1
        assert result["column_count"] == 5

    def test_many_rows_single_column(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        result = dataset_profile(df)

        assert result["row_count"] == 100
        assert result["column_count"] == 1

    def test_mixed_type_df_counts(self) -> None:
        df = _make_mixed_df()
        result = dataset_profile(df)

        assert result["row_count"] == 3
        assert result["column_count"] == 3


# ---------------------------------------------------------------------------
# AC-1 continued: columns list structure
# ---------------------------------------------------------------------------

class TestColumnsMetadata:
    def test_columns_list_length_matches_dataframe(self) -> None:
        df = _make_mixed_df()
        result = dataset_profile(df)

        assert len(result["columns"]) == len(df.columns)

    def test_column_names_match(self) -> None:
        df = _make_numeric_df()
        result = dataset_profile(df)
        names = [c["name"] for c in result["columns"]]

        assert names == list(df.columns)

    def test_column_dtype_present(self) -> None:
        df = _make_numeric_df()
        result = dataset_profile(df)

        for col_meta in result["columns"]:
            assert "dtype" in col_meta
            assert isinstance(col_meta["dtype"], str)
            assert len(col_meta["dtype"]) > 0

    def test_null_count_zero_when_no_nulls(self) -> None:
        df = _make_numeric_df()
        result = dataset_profile(df)

        for col_meta in result["columns"]:
            assert col_meta["null_count"] == 0

    def test_null_count_reflects_actual_nulls(self) -> None:
        df = pd.DataFrame({"a": [1.0, None, 3.0, None, 5.0]})
        result = dataset_profile(df)

        assert result["columns"][0]["null_count"] == 2


# ---------------------------------------------------------------------------
# AC-2: numeric stats populated for all numeric columns
# ---------------------------------------------------------------------------

class TestNumericStats:
    STAT_KEYS = {"mean", "median", "min", "max", "std"}

    def test_all_stat_keys_present(self) -> None:
        df = _make_numeric_df()
        result = dataset_profile(df)

        for col in ["age", "salary"]:
            col_stats = result["stats"][col]
            assert self.STAT_KEYS.issubset(col_stats.keys()), (
                f"Missing stat keys for column '{col}': "
                f"{self.STAT_KEYS - col_stats.keys()}"
            )

    def test_stats_values_are_not_none_for_valid_data(self) -> None:
        df = _make_numeric_df()
        result = dataset_profile(df)

        for col in ["age", "salary"]:
            col_stats = result["stats"][col]
            for key in self.STAT_KEYS:
                assert col_stats[key] is not None, f"{key} is None for column '{col}'"

    def test_mean_correct(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = dataset_profile(df)

        assert result["stats"]["x"]["mean"] == pytest.approx(3.0, abs=1e-4)

    def test_median_correct(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = dataset_profile(df)

        assert result["stats"]["x"]["median"] == pytest.approx(3.0, abs=1e-4)

    def test_min_correct(self) -> None:
        df = pd.DataFrame({"x": [10.0, 20.0, 5.0, 15.0]})
        result = dataset_profile(df)

        assert result["stats"]["x"]["min"] == pytest.approx(5.0, abs=1e-4)

    def test_max_correct(self) -> None:
        df = pd.DataFrame({"x": [10.0, 20.0, 5.0, 15.0]})
        result = dataset_profile(df)

        assert result["stats"]["x"]["max"] == pytest.approx(20.0, abs=1e-4)

    def test_std_correct(self) -> None:
        df = pd.DataFrame({"x": [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]})
        result = dataset_profile(df)
        expected_std = float(pd.Series([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).std())

        assert result["stats"]["x"]["std"] == pytest.approx(expected_std, rel=1e-4)

    def test_stats_rounded_to_4_decimal_places(self) -> None:
        df = pd.DataFrame({"x": [1.0, 3.0]})
        result = dataset_profile(df)

        for key in self.STAT_KEYS:
            val = result["stats"]["x"][key]
            if val is not None:
                # Rounding to 4 dp means at most 4 decimal places
                rounded = round(val, 4)
                assert val == rounded, f"{key}={val} is not rounded to 4 dp"

    def test_non_numeric_columns_excluded_from_stats(self) -> None:
        df = _make_mixed_df()
        result = dataset_profile(df)

        # 'name' and 'grade' are categorical — should not appear in stats
        assert "name" not in result["stats"]
        assert "grade" not in result["stats"]
        # 'score' is numeric — should appear
        assert "score" in result["stats"]

    def test_integer_column_profiled_as_numeric(self) -> None:
        df = pd.DataFrame({"count": [1, 2, 3, 4, 5]})
        result = dataset_profile(df)

        assert "count" in result["stats"]
        assert result["stats"]["count"]["mean"] == pytest.approx(3.0, abs=1e-4)

    def test_unavailable_false_for_valid_data(self) -> None:
        df = _make_numeric_df()
        result = dataset_profile(df)

        for col in ["age", "salary"]:
            assert result["stats"][col]["unavailable"] is False


# ---------------------------------------------------------------------------
# AC-3: all-null numeric column → unavailable: True, stats are None
# ---------------------------------------------------------------------------

class TestAllNullColumn:
    def test_all_null_stats_are_none(self) -> None:
        df = pd.DataFrame({"null_col": [None, None, None, None]})
        # Force numeric dtype
        df["null_col"] = pd.to_numeric(df["null_col"])
        result = dataset_profile(df)

        col_stats = result["stats"]["null_col"]
        for key in ("mean", "median", "min", "max", "std"):
            assert col_stats[key] is None, f"Expected None for {key}, got {col_stats[key]}"

    def test_all_null_unavailable_is_true(self) -> None:
        df = pd.DataFrame({"null_col": pd.array([None, None, None], dtype=pd.Float64Dtype())})
        result = dataset_profile(df)

        assert result["stats"]["null_col"]["unavailable"] is True

    def test_all_null_column_null_count_matches_row_count(self) -> None:
        df = pd.DataFrame({"null_col": pd.array([None, None, None], dtype=pd.Float64Dtype())})
        result = dataset_profile(df)

        col_meta = next(c for c in result["columns"] if c["name"] == "null_col")
        assert col_meta["null_count"] == 3

    def test_mixed_null_partial_column_has_valid_stats(self) -> None:
        """A column with some nulls but not all-null should have valid stats."""
        df = pd.DataFrame({"x": [1.0, None, 3.0, None, 5.0]})
        result = dataset_profile(df)

        col_stats = result["stats"]["x"]
        # mean of [1, 3, 5] = 3.0
        assert col_stats["mean"] == pytest.approx(3.0, abs=1e-4)
        assert col_stats["unavailable"] is False

    def test_nan_float_column_treated_as_unavailable(self) -> None:
        df = pd.DataFrame({"x": [float("nan"), float("nan"), float("nan")]})
        result = dataset_profile(df)

        assert result["stats"]["x"]["unavailable"] is True
        for key in ("mean", "median", "min", "max", "std"):
            assert result["stats"]["x"][key] is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_numeric_columns_stats_is_empty(self) -> None:
        df = pd.DataFrame({"name": ["Alice", "Bob"], "city": ["NY", "LA"]})
        result = dataset_profile(df)

        assert result["stats"] == {}

    def test_single_value_column_std_is_nan_or_zero(self) -> None:
        """A single-row numeric column: std is NaN (pandas ddof=1) → unavailable."""
        df = pd.DataFrame({"x": [42.0]})
        result = dataset_profile(df)

        # pandas std with ddof=1 on a single element returns NaN
        col_stats = result["stats"]["x"]
        assert col_stats["std"] is None
        assert col_stats["unavailable"] is True

    def test_constant_column_std_is_zero(self) -> None:
        """A column with all identical values has std=0, which is valid."""
        df = pd.DataFrame({"x": [7.0, 7.0, 7.0, 7.0]})
        result = dataset_profile(df)

        col_stats = result["stats"]["x"]
        assert col_stats["std"] == pytest.approx(0.0, abs=1e-4)
        assert col_stats["unavailable"] is False

    def test_large_dataframe_counts(self) -> None:
        df = pd.DataFrame({"v": range(10_000)})
        result = dataset_profile(df)

        assert result["row_count"] == 10_000
        assert result["column_count"] == 1

    def test_return_keys_present(self) -> None:
        df = _make_numeric_df()
        result = dataset_profile(df)

        assert set(result.keys()) == {"row_count", "column_count", "columns", "stats"}

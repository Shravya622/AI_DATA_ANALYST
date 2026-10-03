"""
Unit tests for app.type_inferrer.infer_schema

Covers every acceptance criterion for task 3.2:
  - integer column → "numeric"
  - ISO date-string column → "datetime"
  - arbitrary string column → "categorical"
  - every column has exactly one ColumnSchema entry
  - null_count accurately reflects NaN values
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.type_inferrer import infer_schema
from app.models import ColumnSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _schema_map(df: pd.DataFrame) -> dict[str, ColumnSchema]:
    """Return infer_schema result keyed by column name for easy lookup."""
    return {s.name: s for s in infer_schema(df)}


# ---------------------------------------------------------------------------
# AC: integer column → "numeric"
# ---------------------------------------------------------------------------

class TestNumericClassification:
    def test_integer_column_is_numeric(self):
        df = pd.DataFrame({"count": [1, 2, 3, 4, 5]})
        schemas = _schema_map(df)
        assert schemas["count"].dtype == "numeric"

    def test_float_column_is_numeric(self):
        df = pd.DataFrame({"price": [1.1, 2.2, 3.3]})
        schemas = _schema_map(df)
        assert schemas["price"].dtype == "numeric"

    def test_mixed_int_float_column_is_numeric(self):
        df = pd.DataFrame({"val": pd.array([1, 2, 3], dtype="Int64")})
        schemas = _schema_map(df)
        assert schemas["val"].dtype == "numeric"

    def test_numpy_int_array_is_numeric(self):
        df = pd.DataFrame({"x": np.array([10, 20, 30], dtype=np.int32)})
        schemas = _schema_map(df)
        assert schemas["x"].dtype == "numeric"


# ---------------------------------------------------------------------------
# AC: ISO date-string column → "datetime"
# ---------------------------------------------------------------------------

class TestDatetimeClassification:
    def test_iso_date_strings_are_datetime(self):
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-02-15", "2024-03-31"]
        })
        schemas = _schema_map(df)
        assert schemas["date"].dtype == "datetime"

    def test_iso_datetime_strings_are_datetime(self):
        df = pd.DataFrame({
            "ts": ["2024-01-01T10:00:00", "2024-06-15T12:30:00", "2024-12-31T23:59:59"]
        })
        schemas = _schema_map(df)
        assert schemas["ts"].dtype == "datetime"

    def test_native_datetime_dtype_is_datetime(self):
        df = pd.DataFrame({
            "dt": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])
        })
        schemas = _schema_map(df)
        assert schemas["dt"].dtype == "datetime"

    def test_mostly_parseable_dates_with_few_nulls(self):
        """90% valid ISO dates (above 80% threshold) → datetime."""
        dates = ["2024-01-" + str(i).zfill(2) for i in range(1, 10)]
        dates.append(None)  # 10% null
        df = pd.DataFrame({"d": dates})
        schemas = _schema_map(df)
        assert schemas["d"].dtype == "datetime"

    def test_below_threshold_not_datetime(self):
        """Only 50% valid ISO dates → should NOT be datetime (falls to categorical)."""
        values = ["2024-01-01", "2024-01-02", "not-a-date", "also-not", "nope"]
        df = pd.DataFrame({"mixed": values})
        schemas = _schema_map(df)
        assert schemas["mixed"].dtype == "categorical"


# ---------------------------------------------------------------------------
# AC: arbitrary string column → "categorical"
# ---------------------------------------------------------------------------

class TestCategoricalClassification:
    def test_arbitrary_strings_are_categorical(self):
        df = pd.DataFrame({"region": ["North", "South", "East", "West"]})
        schemas = _schema_map(df)
        assert schemas["region"].dtype == "categorical"

    def test_mixed_non_date_strings_are_categorical(self):
        df = pd.DataFrame({"label": ["foo", "bar", "baz", "qux"]})
        schemas = _schema_map(df)
        assert schemas["label"].dtype == "categorical"

    def test_boolean_like_strings_are_categorical(self):
        df = pd.DataFrame({"flag": ["True", "False", "True", "False"]})
        schemas = _schema_map(df)
        assert schemas["flag"].dtype == "categorical"

    def test_object_dtype_random_words_categorical(self):
        df = pd.DataFrame({"words": ["apple", "banana", "cherry", "date", "elderberry"]})
        schemas = _schema_map(df)
        # "date" alone should not flip the whole column to datetime (only 20%)
        assert schemas["words"].dtype == "categorical"


# ---------------------------------------------------------------------------
# AC: every column has exactly one ColumnSchema entry
# ---------------------------------------------------------------------------

class TestSchemaCoverage:
    def test_single_column_returns_one_entry(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        schemas = infer_schema(df)
        assert len(schemas) == 1

    def test_multiple_columns_each_get_one_entry(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
        })
        schemas = infer_schema(df)
        assert len(schemas) == 3

    def test_schema_names_match_dataframe_columns(self):
        df = pd.DataFrame({
            "revenue": [100.0, 200.0],
            "region": ["North", "South"],
            "date": ["2024-01-01", "2024-06-01"],
        })
        schemas = infer_schema(df)
        schema_names = [s.name for s in schemas]
        assert schema_names == list(df.columns)

    def test_empty_dataframe_returns_schema_per_column(self):
        df = pd.DataFrame({"x": pd.Series([], dtype=float), "y": pd.Series([], dtype=str)})
        schemas = infer_schema(df)
        assert len(schemas) == 2
        assert [s.name for s in schemas] == ["x", "y"]

    def test_order_preserved(self):
        cols = ["z", "a", "m", "b"]
        df = pd.DataFrame({c: [1, 2] for c in cols})
        schemas = infer_schema(df)
        assert [s.name for s in schemas] == cols

    def test_dtype_always_one_of_three_values(self):
        df = pd.DataFrame({
            "num": [1, 2, 3],
            "cat": ["a", "b", "c"],
            "dt": ["2024-01-01", "2024-02-01", "2024-03-01"],
        })
        schemas = infer_schema(df)
        allowed = {"numeric", "datetime", "categorical"}
        for s in schemas:
            assert s.dtype in allowed, f"Unexpected dtype '{s.dtype}' for column '{s.name}'"


# ---------------------------------------------------------------------------
# AC: null_count accurately reflects NaN values
# ---------------------------------------------------------------------------

class TestNullCount:
    def test_no_nulls_gives_zero(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        schemas = _schema_map(df)
        assert schemas["a"].null_count == 0

    def test_all_nulls_counted(self):
        df = pd.DataFrame({"a": [None, None, None]})
        schemas = _schema_map(df)
        assert schemas["a"].null_count == 3

    def test_partial_nulls_counted_correctly(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0, None, 5.0]})
        schemas = _schema_map(df)
        assert schemas["a"].null_count == 2

    def test_null_count_is_int(self):
        df = pd.DataFrame({"a": [1, None, 3]})
        schemas = _schema_map(df)
        assert isinstance(schemas["a"].null_count, int)

    def test_null_count_independent_per_column(self):
        df = pd.DataFrame({
            "x": [1, None, 3, None, None],   # 3 nulls
            "y": [None, 2, 3, 4, 5],          # 1 null
            "z": [1, 2, 3, 4, 5],             # 0 nulls
        })
        schemas = _schema_map(df)
        assert schemas["x"].null_count == 3
        assert schemas["y"].null_count == 1
        assert schemas["z"].null_count == 0

    def test_string_column_nan_counted(self):
        df = pd.DataFrame({"name": ["Alice", None, "Charlie", None]})
        schemas = _schema_map(df)
        assert schemas["name"].null_count == 2

    def test_date_column_with_nulls(self):
        df = pd.DataFrame({
            "date": ["2024-01-01", None, "2024-03-01", None, "2024-05-01"]
        })
        schemas = _schema_map(df)
        assert schemas["date"].null_count == 2
        assert schemas["date"].dtype == "datetime"


# ---------------------------------------------------------------------------
# Integration: multi-type DataFrame (mirrors sample_data fixture)
# ---------------------------------------------------------------------------

class TestMixedDataFrame:
    def test_sample_data_like_dataframe(self):
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "region": ["North", "South", "East"],
            "product": ["Widget", "Gadget", "Doohickey"],
            "revenue": [1000.0, 1500.0, 800.0],
            "units_sold": [10, 15, 8],
            "discount_pct": [0.05, 0.10, 0.0],
        })
        schemas = _schema_map(df)

        assert schemas["date"].dtype == "datetime"
        assert schemas["region"].dtype == "categorical"
        assert schemas["product"].dtype == "categorical"
        assert schemas["revenue"].dtype == "numeric"
        assert schemas["units_sold"].dtype == "numeric"
        assert schemas["discount_pct"].dtype == "numeric"

        # All null counts should be 0 for this clean fixture
        for name, s in schemas.items():
            assert s.null_count == 0, f"Expected 0 nulls for '{name}'"

    def test_returns_column_schema_instances(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        schemas = infer_schema(df)
        for s in schemas:
            assert isinstance(s, ColumnSchema)

"""
Unit tests for the generate_sql and generate_pandas tools (task 11.2).

Covers all acceptance criteria:
- generate_sql returns language="sql" and a non-null snippet for valid inputs.
- generate_pandas returns language="python" and a non-null snippet for valid inputs.
- Both are registered in TOOLS.
- snippet=None (not an exception) is returned when generation fails.
"""

import pandas as pd
import pytest

from app.tools.generate_pandas import generate_pandas
from app.tools.generate_sql import generate_sql
from app.tools.registry import TOOLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "region": ["North", "South", "East"],
        "revenue": [1000, 2000, 1500],
    })


# ---------------------------------------------------------------------------
# generate_sql — valid inputs
# ---------------------------------------------------------------------------

class TestGenerateSql:
    def test_aggregate_returns_sql_snippet(self, sample_df):
        result = generate_sql(
            sample_df,
            operation="aggregate",
            table_name="sales",
            params={"group_by": "region", "agg_col": "revenue", "agg_fn": "SUM"},
        )
        assert result["language"] == "sql"
        assert result["snippet"] is not None
        assert "region" in result["snippet"].lower() or "sum" in result["snippet"].lower()

    def test_filter_returns_sql_snippet(self, sample_df):
        result = generate_sql(
            sample_df,
            operation="filter",
            table_name="sales",
            params={"column": "revenue", "operator": ">", "value": 500},
        )
        assert result["language"] == "sql"
        assert result["snippet"] is not None
        assert "WHERE" in result["snippet"] or "where" in result["snippet"]

    def test_select_returns_sql_snippet(self, sample_df):
        result = generate_sql(
            sample_df,
            operation="select",
            table_name="sales",
            params={"columns": ["region", "revenue"]},
        )
        assert result["language"] == "sql"
        assert result["snippet"] is not None

    def test_default_table_name_is_data(self, sample_df):
        """table_name defaults to 'data' when omitted."""
        result = generate_sql(sample_df, operation="select")
        assert result["language"] == "sql"
        assert result["snippet"] is not None
        assert "data" in result["snippet"]

    def test_invalid_operation_returns_null_snippet(self, sample_df):
        """An unknown operation should return snippet=None, not raise."""
        result = generate_sql(sample_df, operation="pivot")
        assert result["snippet"] is None
        assert "error" in result
        assert result["language"] == "sql"

    def test_no_exception_on_failure(self, sample_df):
        """Failure path must return a dict, never raise."""
        try:
            result = generate_sql(sample_df, operation="nonexistent_op")
            assert isinstance(result, dict)
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"generate_sql raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# generate_pandas — valid inputs
# ---------------------------------------------------------------------------

class TestGeneratePandas:
    def test_aggregate_returns_python_snippet(self, sample_df):
        result = generate_pandas(
            sample_df,
            operation="aggregate",
            params={"group_by": "region", "agg_col": "revenue", "agg_fn": "sum"},
        )
        assert result["language"] == "python"
        assert result["snippet"] is not None
        assert "groupby" in result["snippet"]

    def test_filter_returns_python_snippet(self, sample_df):
        result = generate_pandas(
            sample_df,
            operation="filter",
            params={"column": "revenue", "operator": ">", "value": 500},
        )
        assert result["language"] == "python"
        assert result["snippet"] is not None
        assert "revenue" in result["snippet"]

    def test_select_returns_python_snippet(self, sample_df):
        result = generate_pandas(
            sample_df,
            operation="select",
            params={"columns": ["region", "revenue"]},
        )
        assert result["language"] == "python"
        assert result["snippet"] is not None

    def test_invalid_operation_returns_null_snippet(self, sample_df):
        """An unknown operation should return snippet=None, not raise."""
        result = generate_pandas(sample_df, operation="pivot")
        assert result["snippet"] is None
        assert "error" in result
        assert result["language"] == "python"

    def test_no_exception_on_failure(self, sample_df):
        """Failure path must return a dict, never raise."""
        try:
            result = generate_pandas(sample_df, operation="nonexistent_op")
            assert isinstance(result, dict)
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"generate_pandas raised unexpectedly: {exc}")

    def test_params_defaults_to_empty_dict(self, sample_df):
        """Calling without params should not raise."""
        result = generate_pandas(sample_df, operation="select")
        assert result["language"] == "python"
        # snippet may or may not be None depending on default params — just no exception


# ---------------------------------------------------------------------------
# Tool Registry — both tools must be registered
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_generate_sql_registered(self):
        assert "generate_sql" in TOOLS

    def test_generate_pandas_registered(self):
        assert "generate_pandas" in TOOLS

    def test_generate_sql_has_json_schema(self):
        tool = TOOLS["generate_sql"]
        schema = tool.json_schema
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "generate_sql"
        props = schema["function"]["parameters"]["properties"]
        assert "operation" in props
        assert props["operation"]["enum"] == ["aggregate", "filter", "select"]

    def test_generate_pandas_has_json_schema(self):
        tool = TOOLS["generate_pandas"]
        schema = tool.json_schema
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "generate_pandas"
        props = schema["function"]["parameters"]["properties"]
        assert "operation" in props
        assert props["operation"]["enum"] == ["aggregate", "filter", "select"]

    def test_generate_sql_callable(self):
        assert callable(TOOLS["generate_sql"].callable)

    def test_generate_pandas_callable(self):
        assert callable(TOOLS["generate_pandas"].callable)


# ---------------------------------------------------------------------------
# Registry dispatch integration
# ---------------------------------------------------------------------------

class TestRegistryDispatch:
    def test_dispatch_generate_sql(self, sample_df):
        from unittest.mock import MagicMock
        from app.tools.registry import dispatch

        session = MagicMock()
        session.dataframes = {"sales.csv": sample_df}

        result = dispatch(
            "generate_sql",
            {"operation": "select", "table_name": "sales"},
            session,
        )
        assert result["language"] == "sql"
        assert result["snippet"] is not None

    def test_dispatch_generate_pandas(self, sample_df):
        from unittest.mock import MagicMock
        from app.tools.registry import dispatch

        session = MagicMock()
        session.dataframes = {"sales.csv": sample_df}

        result = dispatch(
            "generate_pandas",
            {"operation": "select", "params": {"columns": ["region"]}},
            session,
        )
        assert result["language"] == "python"
        assert result["snippet"] is not None

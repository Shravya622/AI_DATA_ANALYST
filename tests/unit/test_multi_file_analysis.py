"""
Regression tests for multi-file analysis via combine_all_datasets=True.

Tests verify:
1. Two compatible datasets + combine_all_datasets=True → combined result
2. combine_all_datasets=False → original single-dataset behavior preserved
3. Explicit dataset_filename + combine_all_datasets=False → only that file
4. Incompatible schemas (no shared columns) → clear ValueError
5. Original session DataFrames unchanged after combining
6. Single dataset + combine_all_datasets=True → works normally (no-op)
7. Tool schema exposes combine_all_datasets parameter
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.exceptions import UnknownToolError
from app.models import Session
from app.tools.registry import TOOLS, _resolve_dataframe, dispatch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def df_north() -> pd.DataFrame:
    """Sample dataset 1: North and South regions."""
    return pd.DataFrame({
        "region": ["North", "South", "North"],
        "revenue": [1000.0, 2000.0, 500.0],
    })


@pytest.fixture()
def df_east() -> pd.DataFrame:
    """Sample dataset 2: East and West regions."""
    return pd.DataFrame({
        "region": ["East", "West", "East"],
        "revenue": [300.0, 400.0, 200.0],
    })


@pytest.fixture()
def df_incompatible() -> pd.DataFrame:
    """Dataset with completely different columns — no overlap with df_north."""
    return pd.DataFrame({
        "product": ["A", "B"],
        "units": [10, 20],
    })


@pytest.fixture()
def multi_session(df_north: pd.DataFrame, df_east: pd.DataFrame) -> Session:
    """Session with two compatible DataFrames loaded."""
    return Session(
        session_id="multi-test",
        datasets={},
        dataframes={
            "north_south.csv": df_north,
            "east_west.csv": df_east,
        },
        history=[],
    )


@pytest.fixture()
def single_session(df_north: pd.DataFrame) -> Session:
    """Session with only one DataFrame."""
    return Session(
        session_id="single-test",
        datasets={},
        dataframes={"north_south.csv": df_north},
        history=[],
    )


@pytest.fixture()
def incompatible_session(df_north: pd.DataFrame, df_incompatible: pd.DataFrame) -> Session:
    """Session with two DataFrames that share no columns."""
    return Session(
        session_id="incompatible-test",
        datasets={},
        dataframes={
            "north_south.csv": df_north,
            "products.csv": df_incompatible,
        },
        history=[],
    )


# ---------------------------------------------------------------------------
# _resolve_dataframe: combine_all_datasets=True
# ---------------------------------------------------------------------------

class TestResolveDataframeCombine:

    def test_combine_true_returns_concatenated_df(self, multi_session, df_north, df_east):
        """Two compatible datasets combined → result has rows from both."""
        combined = _resolve_dataframe({"combine_all_datasets": True}, multi_session)
        assert combined is not None
        assert len(combined) == len(df_north) + len(df_east)

    def test_combine_true_contains_all_regions(self, multi_session):
        """Combined DataFrame must contain regions from both files."""
        combined = _resolve_dataframe({"combine_all_datasets": True}, multi_session)
        regions = set(combined["region"].tolist())
        assert "North" in regions
        assert "East" in regions
        assert "West" in regions

    def test_combine_true_total_revenue_correct(self, multi_session, df_north, df_east):
        """Total revenue in combined DataFrame equals sum across both files."""
        combined = _resolve_dataframe({"combine_all_datasets": True}, multi_session)
        expected = float(df_north["revenue"].sum() + df_east["revenue"].sum())
        assert abs(float(combined["revenue"].sum()) - expected) < 0.01

    def test_combine_true_uses_shared_columns_only(self, multi_session):
        """Combined DataFrame should only contain columns shared by all datasets."""
        combined = _resolve_dataframe({"combine_all_datasets": True}, multi_session)
        shared = set(multi_session.dataframes["north_south.csv"].columns) & \
                 set(multi_session.dataframes["east_west.csv"].columns)
        assert set(combined.columns) == shared

    def test_combine_true_single_dataset_still_works(self, single_session, df_north):
        """combine_all_datasets=True with only one dataset → returns that dataset."""
        result = _resolve_dataframe({"combine_all_datasets": True}, single_session)
        assert len(result) == len(df_north)

    def test_combine_true_incompatible_schemas_raises_valueerror(self, incompatible_session):
        """No shared columns between datasets → clear ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _resolve_dataframe({"combine_all_datasets": True}, incompatible_session)
        assert "no shared columns" in str(exc_info.value).lower()
        assert "north_south.csv" in str(exc_info.value) or "products.csv" in str(exc_info.value)

    def test_combine_does_not_mutate_original_dataframes(self, multi_session, df_north):
        """Original DataFrames in the session must not be modified by combining."""
        original_north_len = len(multi_session.dataframes["north_south.csv"])
        original_north_cols = list(multi_session.dataframes["north_south.csv"].columns)

        _resolve_dataframe({"combine_all_datasets": True}, multi_session)

        assert len(multi_session.dataframes["north_south.csv"]) == original_north_len
        assert list(multi_session.dataframes["north_south.csv"].columns) == original_north_cols

    def test_combine_false_returns_first_df(self, multi_session):
        """combine_all_datasets=False → returns first available DataFrame (original behavior)."""
        result = _resolve_dataframe({"combine_all_datasets": False}, multi_session)
        assert len(result) in (3, 3)  # either df_north or df_east — both have 3 rows

    def test_combine_absent_returns_first_df(self, multi_session):
        """Missing combine_all_datasets key → original behavior preserved."""
        result = _resolve_dataframe({}, multi_session)
        assert result is not None
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _resolve_dataframe: explicit filename overrides
# ---------------------------------------------------------------------------

class TestResolveDataframeExplicitFilename:

    def test_explicit_filename_returns_only_that_file(self, multi_session, df_north):
        """Explicit dataset_filename → only that DataFrame returned."""
        result = _resolve_dataframe(
            {"dataset_filename": "north_south.csv", "combine_all_datasets": False},
            multi_session,
        )
        assert len(result) == len(df_north)
        assert set(result["region"].tolist()) == {"North", "South"}

    def test_explicit_filename_east_file(self, multi_session, df_east):
        result = _resolve_dataframe(
            {"dataset_filename": "east_west.csv", "combine_all_datasets": False},
            multi_session,
        )
        assert len(result) == len(df_east)
        assert "East" in result["region"].tolist()


# ---------------------------------------------------------------------------
# dispatch integration: combine_all_datasets through aggregate_data
# ---------------------------------------------------------------------------

class TestDispatchWithCombine:

    def test_aggregate_combine_true_sums_both_files(self, multi_session, df_north, df_east):
        """aggregate_data with combine_all_datasets=True sums revenue across both files."""
        result = dispatch(
            "aggregate_data",
            {
                "group_by": "region",
                "agg_col": "revenue",
                "agg_fn": "sum",
                "combine_all_datasets": True,
            },
            multi_session,
        )
        assert "results" in result
        # Should have 4 regions: North, South (from file 1) + East, West (from file 2)
        group_values = {r["group_value"] for r in result["results"]}
        assert "North" in group_values
        assert "East" in group_values

    def test_aggregate_combine_false_uses_single_file(self, multi_session):
        """aggregate_data with combine_all_datasets=False uses only first file."""
        result = dispatch(
            "aggregate_data",
            {
                "group_by": "region",
                "agg_col": "revenue",
                "agg_fn": "sum",
                "combine_all_datasets": False,
            },
            multi_session,
        )
        assert "results" in result
        # Must NOT have all 4 regions (only the single file's 2 regions)
        group_values = {r["group_value"] for r in result["results"]}
        assert len(group_values) <= 3  # at most one file's regions

    def test_aggregate_combine_true_total_revenue_correct(self, multi_session, df_north, df_east):
        """Combined total revenue must equal sum across both files."""
        result = dispatch(
            "aggregate_data",
            {
                "group_by": "region",
                "agg_col": "revenue",
                "agg_fn": "sum",
                "combine_all_datasets": True,
            },
            multi_session,
        )
        total = sum(r["result"] for r in result["results"])
        expected = float(df_north["revenue"].sum() + df_east["revenue"].sum())
        assert abs(total - expected) < 0.01

    def test_incompatible_schemas_dispatch_returns_error(self, incompatible_session):
        """Incompatible schemas produce an error result, not a crash."""
        result = dispatch(
            "aggregate_data",
            {
                "group_by": "region",
                "agg_col": "revenue",
                "agg_fn": "sum",
                "combine_all_datasets": True,
            },
            incompatible_session,
        )
        # dispatch catches the ValueError and returns an error dict
        assert "error" in result


# ---------------------------------------------------------------------------
# Tool schema verification
# ---------------------------------------------------------------------------

class TestToolSchemas:

    def test_aggregate_data_schema_has_combine_all_datasets(self):
        schema = TOOLS["aggregate_data"].json_schema
        props = schema["function"]["parameters"]["properties"]
        assert "combine_all_datasets" in props
        assert props["combine_all_datasets"]["type"] == "boolean"

    def test_query_data_schema_has_combine_all_datasets(self):
        schema = TOOLS["query_data"].json_schema
        props = schema["function"]["parameters"]["properties"]
        assert "combine_all_datasets" in props

    def test_dataset_profile_schema_has_combine_all_datasets(self):
        schema = TOOLS["dataset_profile"].json_schema
        props = schema["function"]["parameters"]["properties"]
        assert "combine_all_datasets" in props

    def test_generate_summary_schema_has_combine_all_datasets(self):
        schema = TOOLS["generate_summary"].json_schema
        props = schema["function"]["parameters"]["properties"]
        assert "combine_all_datasets" in props

    def test_detect_anomalies_schema_has_combine_all_datasets(self):
        schema = TOOLS["detect_anomalies"].json_schema
        props = schema["function"]["parameters"]["properties"]
        assert "combine_all_datasets" in props

    def test_generate_chart_schema_does_not_have_combine_all_datasets(self):
        """generate_chart intentionally does NOT get combine_all_datasets."""
        schema = TOOLS["generate_chart"].json_schema
        props = schema["function"]["parameters"]["properties"]
        assert "combine_all_datasets" not in props

"""
Unit tests for app/tools/percentage_of_total.py

Covers:
- Correct percentage calculation for all groups
- Highlighted (specific) group percentage
- Follow-up scenario: top group from previous aggregation
- Unknown group_value returns error (not exception)
- Unknown column returns error
- Non-numeric agg_col returns error
- Division-by-zero (zero total) returns error
- Existing aggregate_data behaviour unchanged
- Tool is registered in TOOLS registry
"""

from __future__ import annotations

import pytest
import pandas as pd

from app.tools.percentage_of_total import percentage_of_total
from app.tools.aggregate_data import aggregate_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def revenue_df() -> pd.DataFrame:
    """Sample-data-like fixture: 4 regions, revenue values match sample_data sums."""
    return pd.DataFrame({
        "region": ["East", "North", "South", "West"],
        "revenue": [94291.88, 127184.87, 97483.33, 126194.57],
    })


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    """Minimal DataFrame: 3 groups with easy-to-verify percentages."""
    return pd.DataFrame({
        "product": ["A", "A", "B", "B", "C"],
        "units":   [50,  50,  25,  25,  50],
    })


# ---------------------------------------------------------------------------
# Correct percentage calculation — all groups
# ---------------------------------------------------------------------------

class TestPercentageCalculation:
    def test_returns_results_list(self, revenue_df):
        result = percentage_of_total(revenue_df, group_by="region", agg_col="revenue")
        assert "results" in result
        assert isinstance(result["results"], list)
        assert len(result["results"]) == 4

    def test_percentages_sum_to_100(self, revenue_df):
        result = percentage_of_total(revenue_df, group_by="region", agg_col="revenue")
        total_pct = sum(r["percentage"] for r in result["results"])
        assert abs(total_pct - 100.0) < 0.1, f"Percentages sum to {total_pct}, expected ~100"

    def test_north_is_approx_28_57_percent(self, revenue_df):
        """North's share of total revenue in sample_data is ≈ 28.57%."""
        result = percentage_of_total(revenue_df, group_by="region", agg_col="revenue")
        north = next(r for r in result["results"] if r["group_value"] == "North")
        assert abs(north["percentage"] - 28.57) < 0.1, (
            f"Expected ~28.57%, got {north['percentage']}"
        )

    def test_simple_df_group_a_is_50_percent(self, simple_df):
        """Group A = 100 out of 200 total units → 50%."""
        result = percentage_of_total(simple_df, group_by="product", agg_col="units")
        a = next(r for r in result["results"] if r["group_value"] == "A")
        assert abs(a["percentage"] - 50.0) < 0.01

    def test_results_sorted_descending(self, revenue_df):
        result = percentage_of_total(revenue_df, group_by="region", agg_col="revenue")
        pcts = [r["percentage"] for r in result["results"]]
        assert pcts == sorted(pcts, reverse=True)

    def test_metadata_fields_present(self, revenue_df):
        result = percentage_of_total(revenue_df, group_by="region", agg_col="revenue")
        assert result["group_by"] == "region"
        assert result["agg_col"] == "revenue"


# ---------------------------------------------------------------------------
# Highlighted (specific) group percentage — follow-up scenario
# ---------------------------------------------------------------------------

class TestHighlightedGroup:
    def test_highlighted_north_returns_correct_percentage(self, revenue_df):
        """Simulates: 'What % came from that region?' where that region = North."""
        result = percentage_of_total(
            revenue_df, group_by="region", agg_col="revenue", group_value="North"
        )
        assert "highlighted" in result
        assert result["highlighted"] is not None
        assert result["highlighted"]["group_value"] == "North"
        assert abs(result["highlighted"]["percentage"] - 28.57) < 0.1

    def test_highlighted_is_none_when_not_specified(self, revenue_df):
        result = percentage_of_total(revenue_df, group_by="region", agg_col="revenue")
        assert result.get("highlighted") is None

    def test_highlighted_case_insensitive(self, revenue_df):
        """group_value matching is case-insensitive."""
        result = percentage_of_total(
            revenue_df, group_by="region", agg_col="revenue", group_value="north"
        )
        assert result.get("highlighted") is not None
        assert result["highlighted"]["group_value"] == "North"

    def test_unknown_group_value_returns_error(self, revenue_df):
        result = percentage_of_total(
            revenue_df, group_by="region", agg_col="revenue", group_value="Mars"
        )
        assert "error" in result
        assert "Mars" in result["error"]

    def test_unknown_group_value_no_exception(self, revenue_df):
        """Must not raise — returns error dict."""
        try:
            result = percentage_of_total(
                revenue_df, group_by="region", agg_col="revenue", group_value="ZZZ"
            )
            assert "error" in result
        except Exception as exc:
            pytest.fail(f"percentage_of_total raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestErrorCases:
    def test_unknown_group_by_column_returns_error(self, revenue_df):
        result = percentage_of_total(revenue_df, group_by="customer", agg_col="revenue")
        assert "error" in result
        assert "customer" in result["error"]

    def test_unknown_agg_col_returns_error(self, revenue_df):
        result = percentage_of_total(revenue_df, group_by="region", agg_col="profit")
        assert "error" in result
        assert "profit" in result["error"]

    def test_non_numeric_agg_col_returns_error(self, revenue_df):
        result = percentage_of_total(revenue_df, group_by="revenue", agg_col="region")
        assert "error" in result

    def test_zero_total_returns_error(self):
        """If all values in agg_col are 0, division-by-zero must return error."""
        df = pd.DataFrame({"cat": ["A", "B", "C"], "val": [0, 0, 0]})
        result = percentage_of_total(df, group_by="cat", agg_col="val")
        assert "error" in result
        assert "zero" in result["error"].lower()

    def test_no_exception_on_empty_dataframe(self):
        df = pd.DataFrame({"cat": pd.Series([], dtype=str), "val": pd.Series([], dtype=float)})
        try:
            result = percentage_of_total(df, group_by="cat", agg_col="val")
            # Empty → zero total → should return error
            assert "error" in result
        except Exception as exc:
            pytest.fail(f"Raised unexpectedly on empty DataFrame: {exc}")


# ---------------------------------------------------------------------------
# Existing aggregate_data behaviour unchanged
# ---------------------------------------------------------------------------

class TestAggregateDataUnchanged:
    def test_aggregate_sum_still_works(self, revenue_df):
        result = aggregate_data(revenue_df, group_by="region", agg_col="revenue", agg_fn="sum")
        assert "results" in result
        assert len(result["results"]) == 4

    def test_aggregate_result_not_affected_by_new_tool(self, revenue_df):
        """percentage_of_total must not mutate the DataFrame used by aggregate_data."""
        _ = percentage_of_total(revenue_df, group_by="region", agg_col="revenue")
        agg = aggregate_data(revenue_df, group_by="region", agg_col="revenue", agg_fn="sum")
        north = next(r for r in agg["results"] if r["group_value"] == "North")
        assert abs(north["result"] - 127184.87) < 0.01


# ---------------------------------------------------------------------------
# Tool is registered in TOOLS registry
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_percentage_of_total_in_tools(self):
        from app.tools.registry import TOOLS
        assert "percentage_of_total" in TOOLS

    def test_registry_now_has_nine_tools(self):
        from app.tools.registry import TOOLS
        assert len(TOOLS) == 9

    def test_schema_requires_group_by_and_agg_col(self):
        from app.tools.registry import TOOLS
        schema = TOOLS["percentage_of_total"].json_schema
        required = schema["function"]["parameters"].get("required", [])
        assert "group_by" in required
        assert "agg_col" in required

    def test_group_value_is_optional_in_schema(self):
        from app.tools.registry import TOOLS
        schema = TOOLS["percentage_of_total"].json_schema
        required = schema["function"]["parameters"].get("required", [])
        assert "group_value" not in required

    def test_dispatch_via_registry(self, revenue_df):
        from app.models import Session
        from app.tools.registry import dispatch
        session = Session(
            session_id="test",
            datasets={},
            dataframes={"data.csv": revenue_df},
            history=[],
        )
        result = dispatch(
            "percentage_of_total",
            {"group_by": "region", "agg_col": "revenue", "group_value": "North"},
            session,
        )
        assert "results" in result
        assert result["highlighted"]["group_value"] == "North"
        assert abs(result["highlighted"]["percentage"] - 28.57) < 0.1

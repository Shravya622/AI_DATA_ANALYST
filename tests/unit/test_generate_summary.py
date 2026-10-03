"""
Unit tests for app/tools/generate_summary.py

Covers all acceptance criteria for task 5.4:
  - Returns a profile section and at least three insight sections.
  - All numeric figures come from Pandas computation, not LLM generation.
  - Each insight is labelled (e.g. "top_group_by_revenue", "trend_revenue",
    "distribution_region").
  - If fewer than three insights can be computed, available insights are
    returned without error.
"""

from __future__ import annotations

import pytest
import pandas as pd

from app.tools.generate_summary import generate_summary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def full_df() -> pd.DataFrame:
    """DataFrame with both numeric and categorical columns (≥ 10 rows)."""
    return pd.DataFrame(
        {
            "region": ["North", "South", "East", "West", "North",
                        "South", "East", "West", "North", "South",
                        "East", "West"],
            "product": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"],
            "revenue": [100.0, 200.0, 150.0, 300.0, 120.0,
                        180.0, 160.0, 280.0, 110.0, 220.0, 140.0, 260.0],
            "units": [10, 20, 15, 30, 12, 18, 16, 28, 11, 22, 14, 26],
        }
    )


@pytest.fixture()
def numeric_only_df() -> pd.DataFrame:
    """DataFrame with only numeric columns (no categorical)."""
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0,
                   6.0, 7.0, 8.0, 9.0, 10.0],
            "b": [10.0, 9.0, 8.0, 7.0, 6.0,
                   5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )


@pytest.fixture()
def categorical_only_df() -> pd.DataFrame:
    """DataFrame with only categorical columns (no numeric)."""
    return pd.DataFrame(
        {
            "color": ["red", "blue", "green", "red", "blue"],
            "shape": ["circle", "square", "triangle", "circle", "square"],
        }
    )


@pytest.fixture()
def single_row_df() -> pd.DataFrame:
    """DataFrame with a single row — trend insight cannot be computed."""
    return pd.DataFrame({"region": ["North"], "revenue": [500.0]})


@pytest.fixture()
def small_df_under_10_rows() -> pd.DataFrame:
    """DataFrame with fewer than 10 rows — trend insight should be skipped."""
    return pd.DataFrame(
        {
            "region": ["North", "South", "East"],
            "revenue": [100.0, 200.0, 150.0],
        }
    )


# ---------------------------------------------------------------------------
# AC: Returns a profile section
# ---------------------------------------------------------------------------


class TestProfileSection:
    def test_profile_key_present(self, full_df):
        result = generate_summary(full_df)
        assert "profile" in result

    def test_profile_has_row_count(self, full_df):
        result = generate_summary(full_df)
        assert result["profile"]["row_count"] == len(full_df)

    def test_profile_has_column_count(self, full_df):
        result = generate_summary(full_df)
        assert result["profile"]["column_count"] == len(full_df.columns)

    def test_profile_has_stats_for_numeric_cols(self, full_df):
        result = generate_summary(full_df)
        stats = result["profile"]["stats"]
        assert "revenue" in stats
        assert "units" in stats

    def test_profile_stats_contain_mean(self, full_df):
        result = generate_summary(full_df)
        assert result["profile"]["stats"]["revenue"]["mean"] is not None


# ---------------------------------------------------------------------------
# AC: Returns at least three insight sections (when data allows)
# ---------------------------------------------------------------------------


class TestInsightSections:
    def test_insights_key_present(self, full_df):
        result = generate_summary(full_df)
        assert "insights" in result

    def test_at_least_three_insights_for_full_df(self, full_df):
        result = generate_summary(full_df)
        assert len(result["insights"]) >= 3

    def test_top_group_insight_present(self, full_df):
        result = generate_summary(full_df)
        labels = list(result["insights"].keys())
        assert any(lbl.startswith("top_group_by_") for lbl in labels)

    def test_trend_insight_present(self, full_df):
        result = generate_summary(full_df)
        labels = list(result["insights"].keys())
        assert any(lbl.startswith("trend_") for lbl in labels)

    def test_distribution_insight_present(self, full_df):
        result = generate_summary(full_df)
        labels = list(result["insights"].keys())
        assert any(lbl.startswith("distribution_") for lbl in labels)


# ---------------------------------------------------------------------------
# AC: Each insight is labelled with a descriptive key
# ---------------------------------------------------------------------------


class TestInsightLabelling:
    def test_top_group_label_includes_numeric_col(self, full_df):
        result = generate_summary(full_df)
        # First numeric col is "revenue"
        assert "top_group_by_revenue" in result["insights"]

    def test_trend_label_includes_numeric_col(self, full_df):
        result = generate_summary(full_df)
        assert "trend_revenue" in result["insights"]

    def test_distribution_label_includes_categorical_col(self, full_df):
        result = generate_summary(full_df)
        assert "distribution_region" in result["insights"]

    def test_insight_dict_has_label_key(self, full_df):
        result = generate_summary(full_df)
        for key, insight in result["insights"].items():
            assert "label" in insight, f"Insight '{key}' missing 'label' key"
            assert insight["label"] == key

    def test_top_group_insight_structure(self, full_df):
        result = generate_summary(full_df)
        insight = result["insights"]["top_group_by_revenue"]
        assert "top_group" in insight
        assert "top_value" in insight
        assert "text" in insight

    def test_trend_insight_structure(self, full_df):
        result = generate_summary(full_df)
        insight = result["insights"]["trend_revenue"]
        assert "first_period_mean" in insight
        assert "last_period_mean" in insight
        assert "direction" in insight
        assert "text" in insight

    def test_distribution_insight_structure(self, full_df):
        result = generate_summary(full_df)
        insight = result["insights"]["distribution_region"]
        assert "distribution" in insight
        assert "top_value" in insight
        assert "unique_count" in insight


# ---------------------------------------------------------------------------
# AC: All numeric figures come from Pandas computation
# ---------------------------------------------------------------------------


class TestNumericAccuracy:
    def test_top_group_value_matches_pandas(self, full_df):
        result = generate_summary(full_df)
        expected_top_value = float(full_df.groupby("region")["revenue"].sum().max())
        actual = result["insights"]["top_group_by_revenue"]["top_value"]
        assert abs(actual - expected_top_value) < 1e-6

    def test_top_group_name_matches_pandas(self, full_df):
        result = generate_summary(full_df)
        expected_top_group = str(full_df.groupby("region")["revenue"].sum().idxmax())
        actual = result["insights"]["top_group_by_revenue"]["top_group"]
        assert actual == expected_top_group

    def test_trend_first_mean_matches_pandas(self, full_df):
        result = generate_summary(full_df)
        slice_size = max(1, int(len(full_df) * 0.1))
        expected_first_mean = float(full_df["revenue"].iloc[:slice_size].mean())
        actual = result["insights"]["trend_revenue"]["first_period_mean"]
        assert abs(actual - round(expected_first_mean, 4)) < 1e-6

    def test_trend_last_mean_matches_pandas(self, full_df):
        result = generate_summary(full_df)
        slice_size = max(1, int(len(full_df) * 0.1))
        expected_last_mean = float(full_df["revenue"].iloc[-slice_size:].mean())
        actual = result["insights"]["trend_revenue"]["last_period_mean"]
        assert abs(actual - round(expected_last_mean, 4)) < 1e-6

    def test_distribution_counts_match_pandas(self, full_df):
        result = generate_summary(full_df)
        expected = {str(k): int(v) for k, v in full_df["region"].value_counts().items()}
        actual = result["insights"]["distribution_region"]["distribution"]
        assert actual == expected

    def test_profile_row_count_is_correct(self, full_df):
        result = generate_summary(full_df)
        assert result["profile"]["row_count"] == 12


# ---------------------------------------------------------------------------
# AC: Fewer than three insights returned without error
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_numeric_only_df_no_error(self, numeric_only_df):
        """No categorical column → top_group and distribution insights absent."""
        result = generate_summary(numeric_only_df)
        assert "profile" in result
        assert "insights" in result
        # Only trend insight possible (no categorical col for top_group/distribution)
        labels = list(result["insights"].keys())
        assert not any(lbl.startswith("top_group_by_") for lbl in labels)
        assert not any(lbl.startswith("distribution_") for lbl in labels)
        # Trend insight should still be present (≥10 rows)
        assert any(lbl.startswith("trend_") for lbl in labels)

    def test_categorical_only_df_no_error(self, categorical_only_df):
        """No numeric column → only distribution insight possible."""
        result = generate_summary(categorical_only_df)
        assert "profile" in result
        assert "insights" in result
        labels = list(result["insights"].keys())
        # No numeric col means no top_group or trend
        assert not any(lbl.startswith("top_group_by_") for lbl in labels)
        assert not any(lbl.startswith("trend_") for lbl in labels)
        # Distribution should be present
        assert any(lbl.startswith("distribution_") for lbl in labels)

    def test_small_df_under_10_rows_no_trend_insight(self, small_df_under_10_rows):
        """Fewer than 10 rows → trend insight is skipped (not enough data)."""
        result = generate_summary(small_df_under_10_rows)
        assert "profile" in result
        labels = list(result["insights"].keys())
        assert not any(lbl.startswith("trend_") for lbl in labels)

    def test_single_row_no_error(self, single_row_df):
        result = generate_summary(single_row_df)
        assert "profile" in result
        assert "insights" in result

    def test_empty_insights_is_valid_when_no_useful_cols(self):
        """A DF with no numeric and no object/category cols yields empty insights."""
        import numpy as np
        df = pd.DataFrame({"dt": pd.to_datetime(["2024-01-01", "2024-01-02"])})
        result = generate_summary(df)
        assert "profile" in result
        assert isinstance(result["insights"], dict)

    def test_return_type_is_dict(self, full_df):
        result = generate_summary(full_df)
        assert isinstance(result, dict)

    def test_no_exception_on_single_category_value(self):
        """All rows have same category value — should still work."""
        df = pd.DataFrame(
            {
                "region": ["North"] * 12,
                "revenue": [float(i * 10) for i in range(1, 13)],
            }
        )
        result = generate_summary(df)
        assert "profile" in result
        assert "insights" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_trend_direction_increasing(self):
        """Trend should be 'increasing' when last period > first period."""
        # first 10%: [1.0], last 10%: [10.0]
        df = pd.DataFrame(
            {
                "cat": list("abcdefghij"),
                "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            }
        )
        result = generate_summary(df)
        trend = result["insights"]["trend_val"]
        assert trend["direction"] == "increasing"

    def test_trend_direction_decreasing(self):
        """Trend should be 'decreasing' when last period < first period."""
        df = pd.DataFrame(
            {
                "cat": list("abcdefghij"),
                "val": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            }
        )
        result = generate_summary(df)
        trend = result["insights"]["trend_val"]
        assert trend["direction"] == "decreasing"

    def test_distribution_unique_count_correct(self, full_df):
        result = generate_summary(full_df)
        dist = result["insights"]["distribution_region"]
        expected_unique = int(full_df["region"].nunique())
        assert dist["unique_count"] == expected_unique


# ---------------------------------------------------------------------------
# Focused grouped mode — group_by + agg_col provided
# ---------------------------------------------------------------------------


class TestFocusedGroupedMode:
    """
    When group_by and agg_col are provided, generate_summary must produce
    a focused, concise insight about that specific grouping — NOT a generic
    dataset overview.
    """

    @pytest.fixture()
    def region_revenue_df(self) -> pd.DataFrame:
        """Four-region revenue fixture matching sample_data aggregated totals."""
        return pd.DataFrame({
            "region": ["East", "North", "South", "West"],
            "revenue": [94291.88, 127184.87, 97483.33, 126194.57],
        })

    def test_focused_mode_flag_set(self, region_revenue_df):
        result = generate_summary(region_revenue_df, group_by="region", agg_col="revenue")
        assert result.get("focused") is True

    def test_focused_mode_has_no_generic_profile(self, region_revenue_df):
        """Focused mode must NOT include the generic dataset overview."""
        result = generate_summary(region_revenue_df, group_by="region", agg_col="revenue")
        assert result.get("profile") is None, (
            "Focused mode must not include a generic dataset profile"
        )

    def test_focused_mode_has_insights_text(self, region_revenue_df):
        result = generate_summary(region_revenue_df, group_by="region", agg_col="revenue")
        focused_data = result["insights"].get("focused_group_analysis", {})
        assert "insights_text" in focused_data
        assert len(focused_data["insights_text"]) > 0

    def test_focused_mode_identifies_top_group_as_north(self, region_revenue_df):
        result = generate_summary(region_revenue_df, group_by="region", agg_col="revenue")
        focused_data = result["insights"]["focused_group_analysis"]
        assert focused_data["top_group"] == "North", (
            f"Expected top group=North, got {focused_data['top_group']}"
        )

    def test_focused_mode_identifies_bottom_group_as_east(self, region_revenue_df):
        result = generate_summary(region_revenue_df, group_by="region", agg_col="revenue")
        focused_data = result["insights"]["focused_group_analysis"]
        assert focused_data["bottom_group"] == "East", (
            f"Expected bottom group=East, got {focused_data['bottom_group']}"
        )

    def test_focused_mode_north_value_correct(self, region_revenue_df):
        result = generate_summary(region_revenue_df, group_by="region", agg_col="revenue")
        focused_data = result["insights"]["focused_group_analysis"]
        assert abs(focused_data["group_totals"]["North"] - 127184.87) < 0.01

    def test_focused_mode_east_value_correct(self, region_revenue_df):
        result = generate_summary(region_revenue_df, group_by="region", agg_col="revenue")
        focused_data = result["insights"]["focused_group_analysis"]
        assert abs(focused_data["group_totals"]["East"] - 94291.88) < 0.01

    def test_focused_mode_north_percentage_approx_28_57(self, region_revenue_df):
        """North's share of total revenue ≈ 28.57%."""
        result = generate_summary(region_revenue_df, group_by="region", agg_col="revenue")
        focused_data = result["insights"]["focused_group_analysis"]
        north_pct = focused_data["percentages"]["North"]
        assert abs(north_pct - 28.57) < 0.1, (
            f"Expected North ≈ 28.57%, got {north_pct}"
        )

    def test_focused_mode_percentages_sum_to_100(self, region_revenue_df):
        result = generate_summary(region_revenue_df, group_by="region", agg_col="revenue")
        focused_data = result["insights"]["focused_group_analysis"]
        total_pct = sum(focused_data["percentages"].values())
        assert abs(total_pct - 100.0) < 0.1

    def test_focused_mode_insights_text_mentions_north(self, region_revenue_df):
        result = generate_summary(region_revenue_df, group_by="region", agg_col="revenue")
        text = result["insights"]["focused_group_analysis"]["insights_text"]
        assert "North" in text

    def test_focused_mode_insights_text_mentions_east(self, region_revenue_df):
        result = generate_summary(region_revenue_df, group_by="region", agg_col="revenue")
        text = result["insights"]["focused_group_analysis"]["insights_text"]
        assert "East" in text

    def test_focused_mode_insights_text_mentions_percentage(self, region_revenue_df):
        result = generate_summary(region_revenue_df, group_by="region", agg_col="revenue")
        text = result["insights"]["focused_group_analysis"]["insights_text"]
        assert "%" in text

    def test_focused_mode_no_hardcoded_values(self):
        """Focused mode must work for any dataset, not just sample_data."""
        df = pd.DataFrame({
            "product": ["A", "B", "C"],
            "units": [300, 100, 200],
        })
        result = generate_summary(df, group_by="product", agg_col="units")
        assert result.get("focused") is True
        focused_data = result["insights"]["focused_group_analysis"]
        assert focused_data["top_group"] == "A"
        assert focused_data["bottom_group"] == "B"
        assert abs(focused_data["percentages"]["A"] - 50.0) < 0.1


# ---------------------------------------------------------------------------
# Generic mode unchanged when no focus columns provided
# ---------------------------------------------------------------------------


class TestGenericModeUnchanged:
    """When group_by/agg_col are not provided, generate_summary must still
    return the generic profile + three insight sections."""

    @pytest.fixture()
    def full_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "region": ["North", "South", "East", "West", "North",
                       "South", "East", "West", "North", "South",
                       "East", "West"],
            "product": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"],
            "revenue": [100.0, 200.0, 150.0, 300.0, 120.0,
                        180.0, 160.0, 280.0, 110.0, 220.0, 140.0, 260.0],
            "units": [10, 20, 15, 30, 12, 18, 16, 28, 11, 22, 14, 26],
        })

    def test_generic_mode_has_profile(self, full_df):
        result = generate_summary(full_df)
        assert result.get("profile") is not None

    def test_generic_mode_has_insights(self, full_df):
        result = generate_summary(full_df)
        assert "insights" in result
        assert len(result["insights"]) >= 1

    def test_generic_mode_not_focused(self, full_df):
        result = generate_summary(full_df)
        assert not result.get("focused", False)

    def test_generic_mode_profile_row_count_correct(self, full_df):
        result = generate_summary(full_df)
        assert result["profile"]["row_count"] == 12

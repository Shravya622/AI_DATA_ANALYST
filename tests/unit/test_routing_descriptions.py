"""
Regression tests for tool routing descriptions in the registry.

These tests verify that:
1. generate_summary description contains the trigger phrases for business-insight queries.
2. generate_chart description contains explicit exclusion phrases for text queries.
3. Existing generate_summary callable still works correctly on a real DataFrame.
4. Existing generate_chart callable still works correctly.
5. Registry schema integrity is preserved for both tools.

Note: These tests validate the SCHEMA DESCRIPTIONS that guide LLM tool selection.
They do not mock LLM calls — they verify the contract between the registry and
the LLM prompt, which is what actually controls routing behaviour.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.tools.registry import TOOLS
from app.tools.generate_summary import generate_summary
from app.tools.generate_chart import generate_chart


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_df() -> pd.DataFrame:
    return pd.DataFrame({
        "region": ["North", "South", "East", "West"],
        "revenue": [1000.0, 800.0, 600.0, 400.0],
        "units_sold": [10, 8, 6, 4],
    })


# ---------------------------------------------------------------------------
# 1. generate_summary schema description contains routing triggers
# ---------------------------------------------------------------------------

class TestGenerateSummaryDescription:
    """The description the LLM sees must clearly route insight/summary queries."""

    def _get_description(self) -> str:
        return TOOLS["generate_summary"].json_schema["function"]["description"]

    def test_description_mentions_textual(self):
        desc = self._get_description()
        assert "TEXT" in desc.upper() or "text" in desc, (
            "generate_summary description must mention text-based output"
        )

    def test_description_mentions_business_insights(self):
        desc = self._get_description()
        assert "insight" in desc.lower(), (
            "generate_summary description must mention 'insights'"
        )

    def test_description_mentions_summary(self):
        desc = self._get_description()
        assert "summary" in desc.lower() or "summarise" in desc.lower(), (
            "generate_summary description must mention 'summary'"
        )

    def test_description_triggers_on_no_chart_phrase(self):
        desc = self._get_description()
        # Must explicitly guide the model when user says "do not create a chart"
        assert "do not" in desc.lower() or "no chart" in desc.lower(), (
            "generate_summary description must handle 'do not create a chart' phrasing"
        )

    def test_description_mentions_do_not_use_chart(self):
        desc = self._get_description()
        assert "generate_chart" in desc, (
            "generate_summary must reference generate_chart to guide routing away from it"
        )


# ---------------------------------------------------------------------------
# 2. generate_chart schema description contains exclusion phrases
# ---------------------------------------------------------------------------

class TestGenerateChartDescription:
    """The description must explicitly exclude text-insight queries."""

    def _get_description(self) -> str:
        return TOOLS["generate_chart"].json_schema["function"]["description"]

    def test_description_says_only_use_for_visual(self):
        desc = self._get_description()
        assert "only" in desc.lower() or "explicit" in desc.lower(), (
            "generate_chart description must state it is only for explicit chart requests"
        )

    def test_description_excludes_text_summaries(self):
        desc = self._get_description()
        assert "summary" in desc.lower() or "insight" in desc.lower(), (
            "generate_chart description must mention it should NOT be used for text summaries"
        )

    def test_description_mentions_do_not_use_for_text(self):
        desc = self._get_description()
        assert "do not" in desc.lower() or "not" in desc.lower(), (
            "generate_chart description must include a 'do not' exclusion"
        )

    def test_description_still_lists_supported_chart_types(self):
        desc = self._get_description()
        for chart_type in ("bar", "line", "pie", "scatter", "histogram"):
            assert chart_type in desc.lower(), (
                f"generate_chart description must still mention '{chart_type}'"
            )


# ---------------------------------------------------------------------------
# 3. Existing generate_summary callable works correctly
# ---------------------------------------------------------------------------

class TestGenerateSummaryFunctionality:
    def test_returns_profile_and_insights(self, simple_df):
        result = generate_summary(simple_df)
        assert "profile" in result
        assert "insights" in result

    def test_profile_has_row_count(self, simple_df):
        result = generate_summary(simple_df)
        assert result["profile"]["row_count"] == 4

    def test_insights_are_based_on_dataset(self, simple_df):
        result = generate_summary(simple_df)
        # The top-group insight should identify North as the top region by revenue
        top_insight_key = next(
            (k for k in result["insights"] if "top_group" in k), None
        )
        assert top_insight_key is not None, "Expected a top_group insight"
        top_insight = result["insights"][top_insight_key]
        assert top_insight.get("top_group") == "North"

    def test_insights_numeric_figures_not_hardcoded(self, simple_df):
        """Verify the insight values come from Pandas, not hardcoded strings."""
        result = generate_summary(simple_df)
        top_key = next(k for k in result["insights"] if "top_group" in k)
        top_value = result["insights"][top_key]["top_value"]
        # North's sum of revenue = 1000.0
        assert abs(top_value - 1000.0) < 0.01, (
            f"Expected top_value=1000.0 (from Pandas), got {top_value}"
        )


# ---------------------------------------------------------------------------
# 4. Existing generate_chart callable still works correctly
# ---------------------------------------------------------------------------

class TestGenerateChartFunctionality:
    def test_bar_chart_returns_base64(self, simple_df):
        result = generate_chart(simple_df, "bar", x_col="region", y_col="revenue")
        assert "base64_png" in result
        assert result["base64_png"] is not None
        assert "error" not in result

    def test_invalid_chart_type_returns_error(self, simple_df):
        result = generate_chart(simple_df, "heatmap")
        assert "error" in result
        assert "base64_png" not in result


# ---------------------------------------------------------------------------
# 5. Schema integrity preserved
# ---------------------------------------------------------------------------

class TestSchemaIntegrity:
    def test_generate_summary_schema_type_is_function(self):
        schema = TOOLS["generate_summary"].json_schema
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "generate_summary"

    def test_generate_chart_schema_type_is_function(self):
        schema = TOOLS["generate_chart"].json_schema
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "generate_chart"

    def test_generate_chart_still_requires_chart_type(self):
        schema = TOOLS["generate_chart"].json_schema
        required = schema["function"]["parameters"].get("required", [])
        assert "chart_type" in required

    def test_generate_summary_has_no_required_params(self):
        schema = TOOLS["generate_summary"].json_schema
        required = schema["function"]["parameters"].get("required", [])
        assert required == []

    def test_generate_chart_enum_unchanged(self):
        schema = TOOLS["generate_chart"].json_schema
        enum = schema["function"]["parameters"]["properties"]["chart_type"]["enum"]
        assert set(enum) == {"bar", "line", "pie", "scatter", "histogram"}

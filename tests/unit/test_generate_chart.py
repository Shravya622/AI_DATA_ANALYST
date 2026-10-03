"""
Unit tests for app/tools/generate_chart.py — Task 9.2 Generate Chart Tool.

Covers all acceptance criteria:
- A valid chart_type returns a dict with base64_png set and error absent.
- An invalid chart_type (e.g. "heatmap") returns the expected error dict.
- sampled=True is returned when data exceeded 10 000 points and sampling succeeded.
- error is set and base64_png is absent when sampling fails.
"""

from __future__ import annotations

import base64
import unittest
from unittest.mock import patch

import pandas as pd
import pytest

from app.tools.generate_chart import generate_chart


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_df() -> pd.DataFrame:
    """5-row mixed DataFrame for standard chart tests."""
    return pd.DataFrame(
        {
            "category": ["A", "B", "C", "D", "E"],
            "value": [10.0, 20.0, 15.0, 30.0, 25.0],
        }
    )


def _large_df(n: int = 15_000) -> pd.DataFrame:
    """DataFrame with n rows to trigger sampling in the renderer."""
    import numpy as np

    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "x": rng.integers(0, 100, size=n).astype(float),
            "y": rng.random(size=n) * 1000,
        }
    )


def _is_valid_png(b64_str: str) -> bool:
    """Decode a base64 string and check for PNG magic bytes."""
    try:
        raw = base64.b64decode(b64_str)
        return raw[:8] == b"\x89PNG\r\n\x1a\n"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 1. Valid chart_type → base64_png present, error absent
# ---------------------------------------------------------------------------


class TestValidChartTypes:
    """Acceptance criterion: a valid chart_type returns base64_png and no error."""

    @pytest.mark.parametrize("chart_type", ["bar", "line", "scatter"])
    def test_xy_chart_returns_base64_png(self, chart_type: str):
        df = _small_df()
        result = generate_chart(df, chart_type, x_col="category", y_col="value")

        assert "base64_png" in result, f"{chart_type}: base64_png missing from result"
        assert result["base64_png"] is not None, f"{chart_type}: base64_png is None"
        assert result["base64_png"] != "", f"{chart_type}: base64_png is empty string"

    @pytest.mark.parametrize("chart_type", ["bar", "line", "scatter"])
    def test_xy_chart_has_no_error(self, chart_type: str):
        df = _small_df()
        result = generate_chart(df, chart_type, x_col="category", y_col="value")

        assert "error" not in result, (
            f"{chart_type}: unexpected error key in result: {result.get('error')}"
        )

    def test_bar_chart_produces_valid_png(self):
        df = _small_df()
        result = generate_chart(df, "bar", x_col="category", y_col="value")

        assert _is_valid_png(result["base64_png"]), "bar chart base64 does not decode to valid PNG"

    def test_pie_chart_returns_base64_png(self):
        df = _small_df()
        result = generate_chart(df, "pie", x_col="category", y_col="value")

        assert "base64_png" in result
        assert "error" not in result

    def test_histogram_returns_base64_png(self):
        df = _small_df()
        result = generate_chart(df, "histogram", y_col="value")

        assert "base64_png" in result
        assert "error" not in result

    def test_result_contains_sampled_key(self):
        """sampled key must always be present on success."""
        df = _small_df()
        result = generate_chart(df, "bar", x_col="category", y_col="value")

        assert "sampled" in result

    def test_small_df_sampled_is_false(self):
        """A small DataFrame must not trigger sampling."""
        df = _small_df()
        result = generate_chart(df, "bar", x_col="category", y_col="value")

        assert result["sampled"] is False

    def test_optional_title_does_not_cause_error(self):
        df = _small_df()
        result = generate_chart(df, "line", x_col="category", y_col="value", title="My Chart")

        assert "base64_png" in result
        assert "error" not in result


# ---------------------------------------------------------------------------
# 2. Invalid chart_type → error dict
# ---------------------------------------------------------------------------


class TestInvalidChartType:
    """Acceptance criterion: an invalid chart_type returns the exact error dict."""

    EXPECTED_ERROR = "Unsupported chart type. Supported: bar, line, pie, scatter, histogram"

    def test_heatmap_returns_error(self):
        df = _small_df()
        result = generate_chart(df, "heatmap")

        assert result == {"error": self.EXPECTED_ERROR}

    def test_unknown_type_returns_error(self):
        df = _small_df()
        result = generate_chart(df, "boxplot")

        assert result == {"error": self.EXPECTED_ERROR}

    def test_empty_string_chart_type_returns_error(self):
        df = _small_df()
        result = generate_chart(df, "")

        assert result == {"error": self.EXPECTED_ERROR}

    def test_uppercase_valid_name_returns_error(self):
        """Chart type validation is case-sensitive; 'BAR' is not a valid type."""
        df = _small_df()
        result = generate_chart(df, "BAR")

        assert result == {"error": self.EXPECTED_ERROR}

    def test_invalid_chart_type_has_no_base64_png(self):
        df = _small_df()
        result = generate_chart(df, "radar")

        assert "base64_png" not in result

    def test_error_message_lists_all_supported_types(self):
        """The error message must list bar, line, pie, scatter, histogram."""
        df = _small_df()
        result = generate_chart(df, "heatmap")

        error_msg = result["error"]
        for supported in ["bar", "line", "pie", "scatter", "histogram"]:
            assert supported in error_msg, (
                f"Supported type '{supported}' not found in error message: {error_msg}"
            )


# ---------------------------------------------------------------------------
# 3. sampled=True for large DataFrames
# ---------------------------------------------------------------------------


class TestSamplingBehaviour:
    """Acceptance criterion: sampled=True when data exceeded 10 000 points."""

    def test_large_df_triggers_sampling(self):
        df = _large_df(15_000)
        result = generate_chart(df, "scatter", x_col="x", y_col="y")

        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert result["sampled"] is True

    def test_large_df_still_returns_base64_png(self):
        df = _large_df(15_000)
        result = generate_chart(df, "scatter", x_col="x", y_col="y")

        assert "base64_png" in result
        assert result["base64_png"] not in (None, "")

    def test_df_at_exactly_10000_rows_not_sampled(self):
        """A DataFrame of exactly 10 000 rows must not be sampled."""
        import numpy as np

        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {"x": rng.random(10_000), "y": rng.random(10_000)}
        )
        result = generate_chart(df, "scatter", x_col="x", y_col="y")

        assert "error" not in result
        assert result["sampled"] is False

    def test_df_at_10001_rows_is_sampled(self):
        """A DataFrame of 10 001 rows must trigger sampling."""
        import numpy as np

        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {"x": rng.random(10_001), "y": rng.random(10_001)}
        )
        result = generate_chart(df, "scatter", x_col="x", y_col="y")

        assert "error" not in result
        assert result["sampled"] is True


# ---------------------------------------------------------------------------
# 4. Sampling failure → error set, base64_png absent
# ---------------------------------------------------------------------------


class TestSamplingFailure:
    """Acceptance criterion: error set and base64_png absent when sampling fails."""

    def test_sampling_failure_returns_error(self):
        """Simulate a sampling failure by patching render_chart to return an error."""
        from app.renderer import RenderResult

        df = _large_df(15_000)

        with patch("app.tools.generate_chart.render_chart") as mock_render:
            mock_render.return_value = RenderResult(
                error="Chart could not be produced: forced test error"
            )
            result = generate_chart(df, "scatter", x_col="x", y_col="y")

        assert "error" in result
        assert "base64_png" not in result

    def test_sampling_failure_error_message_is_non_empty(self):
        from app.renderer import RenderResult

        df = _large_df(15_000)

        with patch("app.tools.generate_chart.render_chart") as mock_render:
            mock_render.return_value = RenderResult(
                error="Chart could not be produced: forced test error"
            )
            result = generate_chart(df, "scatter", x_col="x", y_col="y")

        assert result["error"] != ""


# ---------------------------------------------------------------------------
# 5. Registry integration — generate_chart is the 5th registered tool
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """Verify generate_chart is properly registered in the Tool Registry."""

    def test_generate_chart_in_tools_registry(self):
        from app.tools.registry import TOOLS

        assert "generate_chart" in TOOLS

    def test_registry_has_five_tools(self):
        from app.tools.registry import TOOLS

        assert len(TOOLS) == 9

    def test_generate_chart_callable_is_generate_chart_function(self):
        from app.tools.registry import TOOLS

        assert TOOLS["generate_chart"].callable is generate_chart

    def test_generate_chart_json_schema_is_openai_format(self):
        from app.tools.registry import TOOLS

        schema = TOOLS["generate_chart"].json_schema
        assert schema["type"] == "function"
        fn_block = schema["function"]
        assert fn_block["name"] == "generate_chart"
        assert "description" in fn_block
        params = fn_block["parameters"]
        assert params["type"] == "object"
        assert "chart_type" in params["properties"]

    def test_generate_chart_schema_requires_chart_type(self):
        from app.tools.registry import TOOLS

        schema = TOOLS["generate_chart"].json_schema
        required = schema["function"]["parameters"].get("required", [])
        assert "chart_type" in required

    def test_generate_chart_schema_chart_type_enum(self):
        from app.tools.registry import TOOLS

        schema = TOOLS["generate_chart"].json_schema
        chart_type_prop = schema["function"]["parameters"]["properties"]["chart_type"]
        assert set(chart_type_prop["enum"]) == {"bar", "line", "pie", "scatter", "histogram"}

    def test_dispatch_generate_chart_via_registry(self):
        from app.models import Session
        from app.tools.registry import dispatch

        df = _small_df()
        session = Session(
            session_id="test-session",
            datasets={},
            dataframes={"data.csv": df},
            history=[],
        )

        result = dispatch(
            "generate_chart",
            {
                "chart_type": "bar",
                "x_col": "category",
                "y_col": "value",
            },
            session,
        )

        assert "base64_png" in result
        assert "error" not in result

    def test_dispatch_generate_chart_invalid_type_via_registry(self):
        from app.models import Session
        from app.tools.registry import dispatch

        df = _small_df()
        session = Session(
            session_id="test-session",
            datasets={},
            dataframes={"data.csv": df},
            history=[],
        )

        result = dispatch(
            "generate_chart",
            {"chart_type": "heatmap"},
            session,
        )

        assert "error" in result
        assert "base64_png" not in result

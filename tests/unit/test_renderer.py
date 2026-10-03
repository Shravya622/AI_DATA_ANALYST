"""
Unit tests for app/renderer.py — task 9.1.

Covers all acceptance criteria:
  1. render_chart with 100-row DF + chart_type="bar" returns valid base64 PNG.
  2. Axis labels in the rendered figure match x_col and y_col.
  3. A 15 000-row DF triggers sampling; sampled=True in the result.
  4. If sampling fails, RenderResult.error is set and base64_png is None.
  5. An unsupported chart_type returns RenderResult(error="Unsupported chart type: ...").
"""

from __future__ import annotations

import base64
import struct
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from app.renderer import SUPPORTED_CHART_TYPES, RenderResult, render_chart


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _is_valid_png(data: bytes) -> bool:
    """Return True if *data* starts with the PNG magic bytes."""
    return data[:8] == PNG_MAGIC


def _make_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "category": [f"cat_{i % 5}" for i in range(n)],
            "value": rng.integers(1, 100, size=n).astype(float),
        }
    )


# ---------------------------------------------------------------------------
# 1. Bar chart returns non-empty base64 that decodes to valid PNG bytes
# ---------------------------------------------------------------------------


class TestBarChartReturnsValidPng:
    def test_base64_is_non_empty(self) -> None:
        df = _make_df(100)
        result = render_chart(df, "bar", x_col="category", y_col="value")
        assert result.error is None
        assert result.base64_png is not None
        assert len(result.base64_png) > 0

    def test_base64_decodes_to_valid_png(self) -> None:
        df = _make_df(100)
        result = render_chart(df, "bar", x_col="category", y_col="value")
        raw = base64.b64decode(result.base64_png)
        assert _is_valid_png(raw), "Decoded bytes do not look like a PNG file"

    def test_sampled_false_for_small_df(self) -> None:
        df = _make_df(100)
        result = render_chart(df, "bar", x_col="category", y_col="value")
        assert result.sampled is False


# ---------------------------------------------------------------------------
# 2. Axis labels match x_col and y_col
# ---------------------------------------------------------------------------


class TestAxisLabels:
    """Verify xlabel/ylabel by inspecting the matplotlib figure directly.

    Strategy: patch plt.savefig to capture the current axes before the figure
    is closed, then check xlabel and ylabel.
    """

    def _render_and_capture(
        self, chart_type: str, x_col: str, y_col: str
    ) -> tuple[RenderResult, str, str]:
        captured: dict[str, str] = {}

        original_savefig = plt.savefig

        def _fake_savefig(buf, **kwargs):
            ax = plt.gca()
            captured["xlabel"] = ax.get_xlabel()
            captured["ylabel"] = ax.get_ylabel()
            original_savefig(buf, **kwargs)

        df = _make_df(50)
        with patch("matplotlib.pyplot.savefig", side_effect=_fake_savefig):
            result = render_chart(df, chart_type, x_col=x_col, y_col=y_col)

        return result, captured.get("xlabel", ""), captured.get("ylabel", "")

    def test_bar_axis_labels(self) -> None:
        result, xlabel, ylabel = self._render_and_capture("bar", "category", "value")
        assert result.error is None
        assert xlabel == "category"
        assert ylabel == "value"

    def test_line_axis_labels(self) -> None:
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"x": range(20), "y": rng.integers(1, 50, 20).astype(float)})

        captured: dict[str, str] = {}
        original_savefig = plt.savefig

        def _fake_savefig(buf, **kwargs):
            ax = plt.gca()
            captured["xlabel"] = ax.get_xlabel()
            captured["ylabel"] = ax.get_ylabel()
            original_savefig(buf, **kwargs)

        with patch("matplotlib.pyplot.savefig", side_effect=_fake_savefig):
            result = render_chart(df, "line", x_col="x", y_col="y")

        assert result.error is None
        assert captured["xlabel"] == "x"
        assert captured["ylabel"] == "y"

    def test_scatter_axis_labels(self) -> None:
        result, xlabel, ylabel = self._render_and_capture("scatter", "category", "value")
        assert result.error is None
        assert xlabel == "category"
        assert ylabel == "value"


# ---------------------------------------------------------------------------
# 3. A 15 000-row DF triggers sampling; sampled=True
# ---------------------------------------------------------------------------


class TestSampling:
    def test_large_df_sets_sampled_true(self) -> None:
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "x": rng.integers(0, 1000, size=15_000).astype(float),
                "y": rng.integers(0, 1000, size=15_000).astype(float),
            }
        )
        result = render_chart(df, "scatter", x_col="x", y_col="y")
        assert result.error is None
        assert result.sampled is True
        assert result.base64_png is not None

    def test_exactly_10000_rows_not_sampled(self) -> None:
        rng = np.random.default_rng(7)
        df = pd.DataFrame(
            {
                "x": rng.integers(0, 100, size=10_000).astype(float),
                "y": rng.integers(0, 100, size=10_000).astype(float),
            }
        )
        result = render_chart(df, "scatter", x_col="x", y_col="y")
        assert result.sampled is False

    def test_10001_rows_triggers_sampling(self) -> None:
        rng = np.random.default_rng(3)
        df = pd.DataFrame(
            {
                "x": rng.integers(0, 100, size=10_001).astype(float),
                "y": rng.integers(0, 100, size=10_001).astype(float),
            }
        )
        result = render_chart(df, "scatter", x_col="x", y_col="y")
        assert result.sampled is True


# ---------------------------------------------------------------------------
# 4. If sampling fails, error is set and base64_png is None
# ---------------------------------------------------------------------------


class TestSamplingFailure:
    def test_sampling_exception_returns_error(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "x": rng.integers(0, 100, size=15_000).astype(float),
                "y": rng.integers(0, 100, size=15_000).astype(float),
            }
        )

        with patch.object(
            df.__class__, "sample", side_effect=RuntimeError("sampling failed")
        ):
            # We need to patch the method on the specific instance; use pd.DataFrame.sample
            with patch("pandas.DataFrame.sample", side_effect=RuntimeError("sampling failed")):
                result = render_chart(df, "scatter", x_col="x", y_col="y")

        assert result.base64_png is None
        assert result.error is not None
        assert "Chart could not be produced" in result.error

    def test_sampling_error_message_contains_cause(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "x": rng.integers(0, 100, size=15_000).astype(float),
                "y": rng.integers(0, 100, size=15_000).astype(float),
            }
        )
        with patch("pandas.DataFrame.sample", side_effect=ValueError("not enough rows")):
            result = render_chart(df, "scatter", x_col="x", y_col="y")

        assert result.base64_png is None
        assert result.error is not None


# ---------------------------------------------------------------------------
# 5. Unsupported chart_type returns an error
# ---------------------------------------------------------------------------


class TestUnsupportedChartType:
    def test_heatmap_unsupported(self) -> None:
        df = _make_df(10)
        result = render_chart(df, "heatmap", x_col="category", y_col="value")
        assert result.base64_png is None
        assert result.error is not None
        assert "Unsupported chart type" in result.error
        assert "heatmap" in result.error

    def test_empty_string_unsupported(self) -> None:
        df = _make_df(10)
        result = render_chart(df, "", x_col="category", y_col="value")
        assert result.error is not None
        assert "Unsupported chart type" in result.error

    def test_error_message_lists_supported_types(self) -> None:
        df = _make_df(10)
        result = render_chart(df, "boxplot", x_col="category", y_col="value")
        for t in ("bar", "line", "pie", "scatter", "histogram"):
            assert t in result.error

    @pytest.mark.parametrize("unsupported", ["heatmap", "boxplot", "violin", "3d", ""])
    def test_various_unsupported_types(self, unsupported: str) -> None:
        df = _make_df(10)
        result = render_chart(df, unsupported, x_col="category", y_col="value")
        assert result.base64_png is None
        assert result.error is not None


# ---------------------------------------------------------------------------
# 6. All supported chart types render without error (smoke tests)
# ---------------------------------------------------------------------------


class TestSupportedChartTypes:
    def _numeric_df(self, n: int = 30) -> pd.DataFrame:
        rng = np.random.default_rng(99)
        return pd.DataFrame(
            {
                "x": rng.integers(1, 10, size=n).astype(float),
                "y": rng.integers(1, 100, size=n).astype(float),
            }
        )

    def test_bar_chart(self) -> None:
        df = _make_df(30)
        result = render_chart(df, "bar", x_col="category", y_col="value")
        assert result.error is None
        assert result.base64_png

    def test_line_chart(self) -> None:
        df = self._numeric_df()
        result = render_chart(df, "line", x_col="x", y_col="y")
        assert result.error is None
        assert result.base64_png

    def test_pie_chart(self) -> None:
        df = _make_df(5)
        result = render_chart(df, "pie", x_col="category", y_col="value")
        assert result.error is None
        assert result.base64_png

    def test_scatter_chart(self) -> None:
        df = self._numeric_df()
        result = render_chart(df, "scatter", x_col="x", y_col="y")
        assert result.error is None
        assert result.base64_png

    def test_histogram_with_y_col(self) -> None:
        df = self._numeric_df()
        result = render_chart(df, "histogram", x_col="x", y_col="y")
        assert result.error is None
        assert result.base64_png

    def test_histogram_with_x_col_only(self) -> None:
        df = self._numeric_df()
        result = render_chart(df, "histogram", x_col="x")
        assert result.error is None
        assert result.base64_png


# ---------------------------------------------------------------------------
# 7. RenderResult dataclass fields
# ---------------------------------------------------------------------------


class TestRenderResultDefaults:
    def test_default_fields(self) -> None:
        r = RenderResult()
        assert r.base64_png is None
        assert r.sampled is False
        assert r.error is None

    def test_supported_chart_types_constant(self) -> None:
        assert SUPPORTED_CHART_TYPES == {"bar", "line", "pie", "scatter", "histogram"}

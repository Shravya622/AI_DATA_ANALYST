"""
Regression tests for generate_chart time_period aggregation (monthly, weekly,
quarterly, yearly).

Key guarantee: tests verify the NUMBER OF DATA POINTS passed to the renderer,
not just whether a chart was produced. A monthly aggregation of 200 daily rows
spanning 7 months must produce exactly 7 data points, not 200.
"""

from __future__ import annotations

import base64
from unittest.mock import patch

import pandas as pd
import pytest

from app.tools.generate_chart import generate_chart
from app.renderer import RenderResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PNG_STUB = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode()


def _fake_render_capturing(captured: list):
    """Return a fake render_chart that captures the DataFrame it receives."""
    def _render(df, chart_type, x_col=None, y_col=None, title=None):
        captured.append(df.copy())
        return RenderResult(base64_png=PNG_STUB, sampled=False)
    return _render


def _date_revenue_df(n_months: int = 7, rows_per_month: int = 4) -> pd.DataFrame:
    """Build a DataFrame with *rows_per_month* rows per calendar month."""
    dates = pd.date_range("2024-01-01", periods=n_months * rows_per_month, freq="7D")
    revenue = [float(100 + i * 10) for i in range(len(dates))]
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "revenue": revenue})


# ---------------------------------------------------------------------------
# Monthly aggregation
# ---------------------------------------------------------------------------

class TestMonthlyAggregation:
    def test_monthly_produces_one_point_per_month(self):
        """200 rows spanning 7 months → 7 aggregated data points."""
        df = _date_revenue_df(n_months=7, rows_per_month=4)
        captured = []
        with patch("app.tools.generate_chart.render_chart", side_effect=_fake_render_capturing(captured)):
            result = generate_chart(df, "line", x_col="date", y_col="revenue", time_period="M")

        assert "error" not in result, result.get("error")
        assert len(captured) == 1
        rendered_df = captured[0]
        assert len(rendered_df) == 7, (
            f"Expected 7 monthly points, got {len(rendered_df)}. "
            "Monthly aggregation is not working — rows are not being aggregated."
        )

    def test_monthly_labels_are_yyyy_mm(self):
        """Monthly x-axis labels must be formatted as YYYY-MM."""
        df = _date_revenue_df(n_months=3, rows_per_month=4)
        captured = []
        with patch("app.tools.generate_chart.render_chart", side_effect=_fake_render_capturing(captured)):
            generate_chart(df, "line", x_col="date", y_col="revenue", time_period="M")

        labels = list(captured[0]["date"])
        assert labels[0] == "2024-01", f"Expected '2024-01', got '{labels[0]}'"
        assert labels[1] == "2024-02"

    def test_monthly_sums_revenue_correctly(self):
        """Each monthly point must be the SUM of revenue within that month."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-14"],
            "revenue": [100.0, 200.0, 300.0, 400.0],
        })
        captured = []
        with patch("app.tools.generate_chart.render_chart", side_effect=_fake_render_capturing(captured)):
            generate_chart(df, "line", x_col="date", y_col="revenue", time_period="M")

        rendered = captured[0].set_index("date")["revenue"]
        assert abs(rendered["2024-01"] - 300.0) < 0.01
        assert abs(rendered["2024-02"] - 700.0) < 0.01

    def test_monthly_returns_base64_png(self):
        df = _date_revenue_df(n_months=3)
        with patch("app.tools.generate_chart.render_chart",
                   side_effect=_fake_render_capturing([])):
            result = generate_chart(df, "line", x_col="date", y_col="revenue", time_period="M")
        assert "base64_png" in result
        assert "error" not in result


# ---------------------------------------------------------------------------
# Weekly aggregation
# ---------------------------------------------------------------------------

class TestWeeklyAggregation:
    def test_weekly_produces_fewer_points_than_source_rows(self):
        """52 daily rows spanning 7+ weeks → fewer than 52 points."""
        dates = pd.date_range("2024-01-01", periods=52, freq="D")
        df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "revenue": [10.0] * 52})
        captured = []
        with patch("app.tools.generate_chart.render_chart", side_effect=_fake_render_capturing(captured)):
            result = generate_chart(df, "line", x_col="date", y_col="revenue", time_period="W")

        assert "error" not in result
        assert len(captured[0]) < 52, "Weekly aggregation should reduce the number of data points"
        assert len(captured[0]) >= 7, "Should have at least 7 weeks from 52 days"

    def test_weekly_labels_are_year_week_format(self):
        """Weekly labels should be strings, not raw datetime objects."""
        df = _date_revenue_df(n_months=2, rows_per_month=2)
        captured = []
        with patch("app.tools.generate_chart.render_chart", side_effect=_fake_render_capturing(captured)):
            generate_chart(df, "line", x_col="date", y_col="revenue", time_period="W")

        labels = list(captured[0]["date"])
        assert all(isinstance(lbl, str) for lbl in labels), (
            "Weekly labels must be strings"
        )


# ---------------------------------------------------------------------------
# Quarterly aggregation
# ---------------------------------------------------------------------------

class TestQuarterlyAggregation:
    def test_quarterly_produces_correct_number_of_quarters(self):
        """12 months of data → 4 quarterly points."""
        dates = pd.date_range("2024-01-01", periods=12, freq="MS")
        df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "revenue": [1000.0] * 12})
        captured = []
        with patch("app.tools.generate_chart.render_chart", side_effect=_fake_render_capturing(captured)):
            result = generate_chart(df, "bar", x_col="date", y_col="revenue", time_period="Q")

        assert "error" not in result, result.get("error")
        assert len(captured[0]) == 4, (
            f"Expected 4 quarterly points, got {len(captured[0])}"
        )

    def test_quarterly_labels_use_yyyy_qn_format(self):
        """Quarterly labels must be 'YYYY-Q1' … 'YYYY-Q4' — no %q strftime."""
        dates = pd.date_range("2024-01-01", periods=4, freq="QS")
        df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "revenue": [100.0] * 4})
        captured = []
        with patch("app.tools.generate_chart.render_chart", side_effect=_fake_render_capturing(captured)):
            generate_chart(df, "bar", x_col="date", y_col="revenue", time_period="Q")

        labels = list(captured[0]["date"])
        assert labels[0] == "2024-Q1", f"Expected '2024-Q1', got '{labels[0]}'"
        assert labels[1] == "2024-Q2"
        assert labels[2] == "2024-Q3"
        assert labels[3] == "2024-Q4"

    def test_quarterly_labels_are_valid_strings(self):
        """All quarterly labels must be non-empty strings."""
        dates = pd.date_range("2023-01-01", periods=8, freq="QS")
        df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "revenue": [100.0] * 8})
        captured = []
        with patch("app.tools.generate_chart.render_chart", side_effect=_fake_render_capturing(captured)):
            generate_chart(df, "bar", x_col="date", y_col="revenue", time_period="Q")

        labels = list(captured[0]["date"])
        assert all(isinstance(lbl, str) and len(lbl) > 0 for lbl in labels)
        # All must match YYYY-QN pattern
        import re
        for lbl in labels:
            assert re.match(r"^\d{4}-Q[1-4]$", lbl), f"Label '{lbl}' does not match YYYY-QN"


# ---------------------------------------------------------------------------
# Yearly aggregation
# ---------------------------------------------------------------------------

class TestYearlyAggregation:
    def test_yearly_produces_one_point_per_year(self):
        """3 years of monthly data → 3 yearly points."""
        dates = pd.date_range("2022-01-01", periods=36, freq="MS")
        df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "revenue": [500.0] * 36})
        captured = []
        with patch("app.tools.generate_chart.render_chart", side_effect=_fake_render_capturing(captured)):
            result = generate_chart(df, "line", x_col="date", y_col="revenue", time_period="Y")

        assert "error" not in result
        assert len(captured[0]) == 3, f"Expected 3 yearly points, got {len(captured[0])}"

    def test_yearly_labels_are_year_strings(self):
        """Yearly labels must be 4-digit year strings."""
        dates = pd.date_range("2022-01-01", periods=3, freq="YS")
        df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "revenue": [1000.0] * 3})
        captured = []
        with patch("app.tools.generate_chart.render_chart", side_effect=_fake_render_capturing(captured)):
            generate_chart(df, "bar", x_col="date", y_col="revenue", time_period="Y")

        labels = list(captured[0]["date"])
        assert labels[0] == "2022"
        assert labels[1] == "2023"
        assert labels[2] == "2024"


# ---------------------------------------------------------------------------
# Error cases for time_period
# ---------------------------------------------------------------------------

class TestTimePeriodErrors:
    def test_unsupported_time_period_returns_error(self):
        df = _date_revenue_df(n_months=3)
        result = generate_chart(df, "line", x_col="date", y_col="revenue", time_period="D")
        assert "error" in result
        assert "time_period" in result["error"].lower() or "D" in result["error"]

    def test_time_period_without_x_col_returns_error(self):
        df = _date_revenue_df(n_months=3)
        result = generate_chart(df, "line", x_col=None, y_col="revenue", time_period="M")
        assert "error" in result

    def test_time_period_without_y_col_returns_error(self):
        df = _date_revenue_df(n_months=3)
        result = generate_chart(df, "line", x_col="date", y_col=None, time_period="M")
        assert "error" in result

    def test_non_datetime_x_col_returns_error(self):
        df = pd.DataFrame({"label": ["A", "B", "C"], "revenue": [100.0, 200.0, 300.0]})
        result = generate_chart(df, "line", x_col="label", y_col="revenue", time_period="M")
        assert "error" in result

    def test_lowercase_time_period_accepted(self):
        """time_period should be case-insensitive."""
        df = _date_revenue_df(n_months=3)
        captured = []
        with patch("app.tools.generate_chart.render_chart", side_effect=_fake_render_capturing(captured)):
            result = generate_chart(df, "line", x_col="date", y_col="revenue", time_period="m")
        assert "error" not in result


# ---------------------------------------------------------------------------
# Existing non-time-series charts are unchanged
# ---------------------------------------------------------------------------

class TestExistingChartsUnchanged:
    def test_non_grouped_chart_uses_raw_rows(self):
        df = pd.DataFrame({
            "category": ["A", "B", "C"],
            "value": [100.0, 200.0, 150.0],
        })
        captured = []
        with patch("app.tools.generate_chart.render_chart", side_effect=_fake_render_capturing(captured)):
            generate_chart(df, "bar", x_col="category", y_col="value")
        assert len(captured[0]) == 3

    def test_grouped_bar_still_works(self):
        df = pd.DataFrame({
            "region": ["East", "North", "East", "North"],
            "revenue": [100.0, 200.0, 150.0, 250.0],
        })
        captured = []
        with patch("app.tools.generate_chart.render_chart", side_effect=_fake_render_capturing(captured)):
            result = generate_chart(df, "bar", x_col="region", y_col="revenue", group_by="region")

        assert "error" not in result
        assert len(captured[0]) == 2  # East and North

    def test_time_period_none_uses_raw_rows(self):
        """time_period=None must not trigger any aggregation."""
        df = _date_revenue_df(n_months=3, rows_per_month=4)
        raw_len = len(df)
        captured = []
        with patch("app.tools.generate_chart.render_chart", side_effect=_fake_render_capturing(captured)):
            generate_chart(df, "line", x_col="date", y_col="revenue", time_period=None)
        assert len(captured[0]) == raw_len

    def test_pie_chart_with_group_by_still_works(self):
        df = pd.DataFrame({
            "region": ["East", "North", "South", "West"] * 5,
            "revenue": [1000.0] * 20,
        })
        captured = []
        with patch("app.tools.generate_chart.render_chart", side_effect=_fake_render_capturing(captured)):
            result = generate_chart(df, "pie", x_col="region", y_col="revenue", group_by="region")

        assert "error" not in result
        assert len(captured[0]) == 4  # 4 region groups


# ---------------------------------------------------------------------------
# Registry dispatch includes time_period
# ---------------------------------------------------------------------------

class TestRegistryDispatch:
    def test_dispatch_passes_time_period(self):
        from app.models import Session
        from app.tools.registry import dispatch

        df = _date_revenue_df(n_months=3, rows_per_month=4)
        session = Session(
            session_id="test",
            datasets={},
            dataframes={"data.csv": df},
            history=[],
        )

        captured = []
        with patch("app.tools.generate_chart.render_chart", side_effect=_fake_render_capturing(captured)):
            result = dispatch(
                "generate_chart",
                {"chart_type": "line", "x_col": "date", "y_col": "revenue", "time_period": "M"},
                session,
            )

        assert "error" not in result
        assert len(captured[0]) == 3, (
            f"Expected 3 monthly points via registry dispatch, got {len(captured[0])}"
        )

    def test_dispatch_without_time_period_uses_raw_rows(self):
        from app.models import Session
        from app.tools.registry import dispatch

        df = _date_revenue_df(n_months=2, rows_per_month=4)
        session = Session(
            session_id="test",
            datasets={},
            dataframes={"data.csv": df},
            history=[],
        )

        captured = []
        with patch("app.tools.generate_chart.render_chart", side_effect=_fake_render_capturing(captured)):
            dispatch(
                "generate_chart",
                {"chart_type": "line", "x_col": "date", "y_col": "revenue"},
                session,
            )

        assert len(captured[0]) == len(df)

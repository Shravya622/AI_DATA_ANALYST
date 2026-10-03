"""
Generate Chart tool for the AI-powered Data Analyst.

Wraps ``app.renderer.render_chart`` and normalises its ``RenderResult`` into a
plain dict suitable for LLM tool-call responses.
"""

from __future__ import annotations

import pandas as pd

from app.renderer import SUPPORTED_CHART_TYPES, render_chart

# Supported time-period resample codes and their x-axis label formats.
# Quarterly labels are computed via Pandas Period arithmetic (not strftime %q)
# to guarantee valid "YYYY-QN" output across all platforms.
_TIME_PERIODS: dict[str, str] = {
    "M": "%Y-%m",
    "W": "%Y-W%W",
    "Y": "%Y",
    "Q": "quarter",   # special-cased below — uses "QE" resample alias
}

# Pandas 2.x renamed "Q" to "QE" (quarter-end).  Use the newer alias
# and fall back gracefully for older Pandas versions.
_QUARTERLY_RESAMPLE_ALIAS = "QE"


def _format_quarter_labels(date_series: pd.Series) -> list[str]:
    """Convert a datetime Series to 'YYYY-Q1' … 'YYYY-Q4' strings safely."""
    periods = date_series.dt.to_period("Q")
    return [f"{p.year}-Q{p.quarter}" for p in periods]


def generate_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str | None = None,
    y_col: str | None = None,
    title: str | None = None,
    group_by: str | None = None,
    time_period: str | None = None,
) -> dict:
    """Render a chart and return the result as a structured dict.

    Parameters
    ----------
    df:
        Source DataFrame.
    chart_type:
        One of the supported chart types: bar, line, pie, scatter, histogram.
    x_col:
        Column for the x-axis (or labels for pie).
    y_col:
        Column for the y-axis (or values for pie/histogram).
    title:
        Optional chart title.
    group_by:
        Optional: aggregate *y_col* by SUM grouped on this categorical column
        before rendering (e.g. total revenue per region).
    time_period:
        Optional: resample a datetime *x_col* to this period before plotting.
        Supported values: ``"M"`` (monthly), ``"W"`` (weekly), ``"Q"``
        (quarterly, labelled as ``YYYY-Q1``), ``"Y"`` (yearly).
        Produces one data point per period rather than one per source row.

    Returns
    -------
    dict
        On success: ``{"base64_png": "...", "sampled": bool}``
        On validation error / render error: ``{"error": "..."}``
    """
    # Validate chart type
    if chart_type not in SUPPORTED_CHART_TYPES:
        supported = "bar, line, pie, scatter, histogram"
        return {"error": f"Unsupported chart type. Supported: {supported}"}

    plot_df = df

    # ------------------------------------------------------------------ #
    # group_by path: categorical aggregation (region, product, etc.)
    # ------------------------------------------------------------------ #
    if group_by and y_col:
        if group_by not in df.columns:
            return {"error": f"group_by column '{group_by}' not found"}
        if y_col not in df.columns:
            return {"error": f"y_col column '{y_col}' not found"}
        if not pd.api.types.is_numeric_dtype(df[y_col]):
            return {"error": f"y_col column '{y_col}' is not numeric"}
        agg = df.groupby(group_by)[y_col].sum().reset_index()
        agg.columns = [group_by, y_col]
        plot_df = agg
        if x_col is None or x_col == group_by:
            x_col = group_by

    # ------------------------------------------------------------------ #
    # time_period path: temporal resampling (monthly, weekly, etc.)
    # ------------------------------------------------------------------ #
    if time_period:
        period_upper = time_period.strip().upper()
        if period_upper not in _TIME_PERIODS:
            return {
                "error": (
                    f"Unsupported time_period '{time_period}'. "
                    f"Supported: {', '.join(_TIME_PERIODS)}"
                )
            }
        if not x_col or x_col not in plot_df.columns:
            return {"error": "time_period requires a valid x_col (datetime column)"}
        if not y_col or y_col not in plot_df.columns:
            return {"error": "time_period requires a valid y_col (numeric column)"}
        if not pd.api.types.is_numeric_dtype(plot_df[y_col]):
            return {"error": f"y_col column '{y_col}' is not numeric"}

        # Parse x_col as datetime if needed
        work = plot_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(work[x_col]):
            try:
                work[x_col] = pd.to_datetime(work[x_col])
            except Exception:
                return {
                    "error": (
                        f"Cannot parse column '{x_col}' as datetime "
                        "for time-period aggregation."
                    )
                }

        # Resample by summing y_col within each period
        resample_alias = _QUARTERLY_RESAMPLE_ALIAS if period_upper == "Q" else period_upper
        resampled = (
            work.set_index(x_col)
                .resample(resample_alias)[y_col]
                .sum()
                .reset_index()
        )

        # Format x-axis labels
        if period_upper == "Q":
            resampled[x_col] = _format_quarter_labels(resampled[x_col])
        else:
            fmt = _TIME_PERIODS[period_upper]
            resampled[x_col] = resampled[x_col].dt.strftime(fmt)

        plot_df = resampled

    result = render_chart(plot_df, chart_type, x_col=x_col, y_col=y_col, title=title)

    if result.error:
        return {"error": result.error}

    return {"base64_png": result.base64_png, "sampled": result.sampled}

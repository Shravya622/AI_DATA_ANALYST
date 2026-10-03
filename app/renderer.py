"""
Chart renderer for the AI-powered Data Analyst application.

Uses matplotlib with the non-interactive Agg backend for server-side rendering.
All charts are returned as base64-encoded PNG strings.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")  # Must be called before importing pyplot

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_CHART_TYPES = {"bar", "line", "pie", "scatter", "histogram"}
_SAMPLING_LIMIT = 10_000


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RenderResult:
    """Result returned by render_chart."""

    base64_png: str | None = None
    sampled: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str | None = None,
    y_col: str | None = None,
    title: str | None = None,
) -> RenderResult:
    """Render a chart from a DataFrame and return it as a base64-encoded PNG.

    Parameters
    ----------
    df:
        Source data.
    chart_type:
        One of ``"bar"``, ``"line"``, ``"pie"``, ``"scatter"``, ``"histogram"``.
    x_col:
        Column name used for the x-axis (or labels for pie charts).
    y_col:
        Column name used for the y-axis (or values for pie/histogram).
    title:
        Optional chart title.

    Returns
    -------
    RenderResult
        On success: ``base64_png`` is a non-empty string, ``sampled`` indicates
        whether down-sampling occurred.
        On failure: ``error`` describes what went wrong, ``base64_png`` is ``None``.
    """
    # --- Validate chart type ------------------------------------------------
    if chart_type not in SUPPORTED_CHART_TYPES:
        supported = ", ".join(sorted(SUPPORTED_CHART_TYPES))
        return RenderResult(
            error=f"Unsupported chart type: {chart_type}. Supported: {supported}"
        )

    # --- Down-sample if necessary -------------------------------------------
    sampled = False
    if len(df) > _SAMPLING_LIMIT:
        try:
            df = df.sample(n=_SAMPLING_LIMIT, random_state=42)
            sampled = True
        except Exception as exc:  # noqa: BLE001
            return RenderResult(error=f"Chart could not be produced: {exc}")

    # --- Draw the chart -----------------------------------------------------
    fig = plt.figure()
    try:
        if chart_type == "bar":
            plt.bar(df[x_col], df[y_col])
        elif chart_type == "line":
            plt.plot(df[x_col], df[y_col])
        elif chart_type == "pie":
            plt.pie(df[y_col], labels=df[x_col])
        elif chart_type == "scatter":
            plt.scatter(df[x_col], df[y_col])
        elif chart_type == "histogram":
            col = y_col or x_col
            plt.hist(df[col])

        # Apply axis labels
        plt.xlabel(x_col or "")
        plt.ylabel(y_col or "")

        if title:
            plt.title(title)

        # Encode to base64 PNG
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        b64str = base64.b64encode(buf.read()).decode("utf-8")

    finally:
        plt.close(fig)

    return RenderResult(base64_png=b64str, sampled=sampled)

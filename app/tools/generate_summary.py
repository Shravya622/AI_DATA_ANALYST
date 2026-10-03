"""
Generate summary tool.

Provides `generate_summary(df)` which builds a structured dict containing:
  - A ``profile`` section from `dataset_profile`.
  - An ``insights`` dict with up to three labelled sections:
      * ``top_group_by_<numeric_col>``  – top group by aggregated sum
      * ``trend_<numeric_col>``         – first 10 % vs last 10 % mean comparison
      * ``distribution_<categorical_col>`` – value counts for first categorical col

All numeric figures are computed by Pandas — no LLM output is involved.
If fewer than three insights can be computed the available ones are returned
without error.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.tools.dataset_profile import dataset_profile


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _first_numeric_col(df: pd.DataFrame) -> str | None:
    """Return the name of the first numeric column, or None."""
    cols = df.select_dtypes(include="number").columns
    return cols[0] if len(cols) > 0 else None


def _first_categorical_col(df: pd.DataFrame) -> str | None:
    """Return the name of the first non-numeric, non-datetime column, or None."""
    cols = df.select_dtypes(include=["object", "category"]).columns
    return cols[0] if len(cols) > 0 else None


def _top_group_insight(df: pd.DataFrame, num_col: str, cat_col: str) -> dict[str, Any]:
    """Return the top group when grouping *cat_col* and summing *num_col*."""
    grouped: pd.Series = df.groupby(cat_col)[num_col].sum()
    top_group = str(grouped.idxmax())
    top_value = float(grouped.max())
    label = f"top_group_by_{num_col}"
    return {
        "label": label,
        "group_by_column": cat_col,
        "aggregated_column": num_col,
        "aggregation": "sum",
        "top_group": top_group,
        "top_value": round(top_value, 4),
        "text": (
            f"The top {cat_col} by total {num_col} is '{top_group}' "
            f"with a sum of {round(top_value, 4)}."
        ),
    }


def _trend_insight(df: pd.DataFrame, num_col: str) -> dict[str, Any] | None:
    """Compare mean of first 10 % rows vs last 10 % rows of *num_col*.

    Returns None if there are fewer than 10 rows (insufficient data for a
    meaningful 10 % split).
    """
    n = len(df)
    slice_size = max(1, int(n * 0.1))
    if n < 10:
        return None

    first_mean = float(df[num_col].iloc[:slice_size].mean())
    last_mean = float(df[num_col].iloc[-slice_size:].mean())

    if pd.isna(first_mean) or pd.isna(last_mean):
        return None

    if first_mean == 0:
        pct_change_str = "N/A (first-period mean is zero)"
        direction = "unchanged"
    else:
        pct_change = ((last_mean - first_mean) / abs(first_mean)) * 100
        pct_change_str = f"{round(pct_change, 2)}%"
        direction = "increasing" if last_mean > first_mean else "decreasing" if last_mean < first_mean else "unchanged"

    label = f"trend_{num_col}"
    return {
        "label": label,
        "column": num_col,
        "first_period_mean": round(first_mean, 4),
        "last_period_mean": round(last_mean, 4),
        "direction": direction,
        "percent_change": pct_change_str,
        "text": (
            f"{num_col} appears {direction}: first 10% mean = "
            f"{round(first_mean, 4)}, last 10% mean = {round(last_mean, 4)} "
            f"({pct_change_str} change)."
        ),
    }


def _distribution_insight(df: pd.DataFrame, cat_col: str) -> dict[str, Any]:
    """Return value counts for the first categorical column."""
    counts: pd.Series = df[cat_col].value_counts()
    label = f"distribution_{cat_col}"
    distribution = {str(k): int(v) for k, v in counts.items()}
    top_value = str(counts.idxmax())
    return {
        "label": label,
        "column": cat_col,
        "distribution": distribution,
        "top_value": top_value,
        "unique_count": int(counts.shape[0]),
        "text": (
            f"'{cat_col}' has {int(counts.shape[0])} unique values. "
            f"The most frequent is '{top_value}' "
            f"({int(counts.max())} occurrences)."
        ),
    }


def _focused_group_insights(
    df: pd.DataFrame, group_by: str, agg_col: str
) -> dict[str, Any]:
    """Build a concise grouped-mode insight dict for *agg_col* by *group_by*.

    Returned structure:
    ``{"insights_text": str, "group_totals": dict, "top_group": str,
       "bottom_group": str, "total": float, "percentages": dict}``
    """
    grouped: pd.Series = df.groupby(group_by)[agg_col].sum().sort_values(ascending=False)
    total: float = float(grouped.sum())

    if total == 0:
        return {
            "insights_text": f"Total {agg_col} across all {group_by} groups is zero.",
            "group_totals": {},
            "top_group": None,
            "bottom_group": None,
            "total": 0.0,
            "percentages": {},
        }

    top_group = str(grouped.index[0])
    top_value = round(float(grouped.iloc[0]), 2)
    bottom_group = str(grouped.index[-1])
    bottom_value = round(float(grouped.iloc[-1]), 2)

    percentages: dict[str, float] = {
        str(k): round(float(v) / total * 100, 2) for k, v in grouped.items()
    }
    group_totals: dict[str, float] = {
        str(k): round(float(v), 2) for k, v in grouped.items()
    }

    # Build ranked list text
    ranked_lines = [
        f"  {i + 1}. {gv}: {group_totals[gv]} ({percentages[gv]}%)"
        for i, gv in enumerate(group_totals)
    ]

    # Difference between top and bottom
    diff = round(top_value - bottom_value, 2)
    pct_diff = round((diff / bottom_value * 100), 1) if bottom_value else 0.0

    # Build the main insights text
    lines = [
        f"Key insights for {agg_col} by {group_by}:",
        f"",
        f"• Highest: {top_group} with {agg_col} = {top_value} "
        f"({percentages[top_group]}% of total {agg_col}).",
        f"• Lowest:  {bottom_group} with {agg_col} = {bottom_value} "
        f"({percentages[bottom_group]}% of total {agg_col}).",
        f"• {top_group} leads {bottom_group} by {diff} "
        f"({pct_diff}% higher).",
        f"",
        f"All {group_by} groups ranked by total {agg_col}:",
    ] + ranked_lines

    return {
        "insights_text": "\n".join(lines),
        "group_totals": group_totals,
        "top_group": top_group,
        "bottom_group": bottom_group,
        "total": round(total, 2),
        "percentages": percentages,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_summary(
    df: pd.DataFrame,
    group_by: str | None = None,
    agg_col: str | None = None,
) -> dict[str, Any]:
    """Generate a structured summary with a profile section and insight sections.

    Parameters
    ----------
    df:
        The source DataFrame to analyse.
    group_by:
        Optional: override the auto-detected categorical column (e.g. ``"region"``).
        When provided, insight 1 (top-group) and insight 3 (distribution) use this
        column instead of the first categorical column in the DataFrame.
    agg_col:
        Optional: override the auto-detected numeric column (e.g. ``"revenue"``).
        When provided, insights 1 (top-group) and 2 (trend) use this column
        instead of the first numeric column in the DataFrame.

    Returns
    -------
    dict with keys:
        ``profile``  – output of :func:`dataset_profile`
        ``insights`` – dict mapping insight label → insight dict.
                       Always present; may be empty if no insights can be
                       computed (e.g. a DataFrame with no numeric or
                       categorical columns).
    """
    # ---------------------------------------------------------------------------
    # FOCUSED MODE: group_by + agg_col explicitly requested
    # Skip the generic dataset overview and produce targeted group insights.
    # ---------------------------------------------------------------------------
    focused_group_by = group_by if (group_by and group_by in df.columns) else None
    focused_agg_col = agg_col if (
        agg_col and agg_col in df.columns and pd.api.types.is_numeric_dtype(df[agg_col])
    ) else None

    if focused_group_by is not None and focused_agg_col is not None:
        try:
            focused = _focused_group_insights(df, focused_group_by, focused_agg_col)
            return {
                "profile": None,   # not included in focused mode
                "insights": {"focused_group_analysis": focused},
                "focused": True,
            }
        except Exception:
            pass  # fall through to generic mode on error

    # ---------------------------------------------------------------------------
    # GENERIC MODE: auto-detect columns and produce full dataset overview.
    # ---------------------------------------------------------------------------
    profile = dataset_profile(df)

    num_col = focused_agg_col if focused_agg_col else _first_numeric_col(df)
    cat_col = focused_group_by if focused_group_by else _first_categorical_col(df)

    insights: dict[str, Any] = {}

    # Insight 1 – top group by aggregated numeric column
    if num_col is not None and cat_col is not None:
        try:
            insight = _top_group_insight(df, num_col, cat_col)
            insights[insight["label"]] = insight
        except Exception:
            pass

    # Insight 2 – trend (first 10 % vs last 10 % mean)
    if num_col is not None:
        try:
            insight = _trend_insight(df, num_col)
            if insight is not None:
                insights[insight["label"]] = insight
        except Exception:
            pass

    # Insight 3 – distribution of first categorical column
    if cat_col is not None:
        try:
            insight = _distribution_insight(df, cat_col)
            insights[insight["label"]] = insight
        except Exception:
            pass

    return {
        "profile": profile,
        "insights": insights,
    }

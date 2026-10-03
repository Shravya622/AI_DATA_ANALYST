"""
Aggregate data tool.

Provides `aggregate_data(df, group_by, agg_col, agg_fn)` which performs
group-by aggregation on a pandas DataFrame.

Supported aggregation functions: "sum", "mean", "count", "min", "max".
All computation is deterministic Pandas operations — no LLM involvement.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Supported aggregation function names.
SUPPORTED_AGG_FNS: frozenset[str] = frozenset({"sum", "mean", "count", "min", "max"})


def _to_python_native(value: Any) -> Any:
    """Convert numpy scalar types to Python-native types for JSON serialisability."""
    # numpy integers
    if hasattr(value, "item"):
        return value.item()
    return value


def aggregate_data(
    df: pd.DataFrame,
    group_by: str,
    agg_col: str,
    agg_fn: str,
) -> dict[str, Any]:
    """Group a DataFrame by a column and aggregate another column.

    Parameters
    ----------
    df:
        The source DataFrame.
    group_by:
        The column name to group by.
    agg_col:
        The column name to aggregate.
    agg_fn:
        The aggregation function to apply. One of: ``"sum"``, ``"mean"``,
        ``"count"``, ``"min"``, ``"max"``.

    Returns
    -------
    dict with keys:
        ``results``  – list of ``{"group_value": str, "result": <native value>}`` dicts
        ``group_by`` – the group_by column name
        ``agg_col``  – the aggregated column name
        ``agg_fn``   – the aggregation function used
    OR an error dict:
        ``{"error": "column X not found"}`` when an unknown column is referenced.
        ``{"error": "unsupported aggregation function X"}`` for an unknown agg_fn.
    """
    # ------------------------------------------------------------------ #
    # Validate group_by column
    # ------------------------------------------------------------------ #
    if group_by not in df.columns:
        return {"error": f"column {group_by} not found"}

    # ------------------------------------------------------------------ #
    # Validate agg_fn
    # ------------------------------------------------------------------ #
    if agg_fn not in SUPPORTED_AGG_FNS:
        return {"error": f"unsupported aggregation function {agg_fn}"}

    # ------------------------------------------------------------------ #
    # Validate agg_col (not required for count, but still should exist)
    # ------------------------------------------------------------------ #
    if agg_col not in df.columns:
        return {"error": f"column {agg_col} not found"}

    # ------------------------------------------------------------------ #
    # Perform aggregation
    # ------------------------------------------------------------------ #
    grouped: pd.Series = df.groupby(group_by)[agg_col].agg(agg_fn)

    # ------------------------------------------------------------------ #
    # Build results list — convert numpy types to Python-native scalars
    # ------------------------------------------------------------------ #
    results: list[dict[str, Any]] = [
        {"group_value": str(k), "result": _to_python_native(v)}
        for k, v in grouped.items()
    ]

    return {
        "results": results,
        "group_by": group_by,
        "agg_col": agg_col,
        "agg_fn": agg_fn,
    }

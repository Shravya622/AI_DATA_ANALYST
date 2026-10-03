"""
Percentage-of-total tool.

Computes each group's percentage share of the sum of a numeric column,
optionally highlighting a specific group (e.g. "that region").

All computation is deterministic Pandas arithmetic — no LLM involvement.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def percentage_of_total(
    df: pd.DataFrame,
    group_by: str,
    agg_col: str,
    group_value: str | None = None,
) -> dict[str, Any]:
    """Calculate each group's percentage share of the total for a numeric column.

    Parameters
    ----------
    df:
        Source DataFrame.
    group_by:
        Categorical column to group by (e.g. ``"region"``).
    agg_col:
        Numeric column whose sum is used as both numerator and denominator
        (e.g. ``"revenue"``).
    group_value:
        Optional: a specific group to highlight (e.g. ``"North"``).  When
        provided, the response also includes ``highlighted`` with that group's
        percentage.  Pass ``None`` to return all groups.

    Returns
    -------
    dict
        On success:
        ``{"results": [{"group_value": str, "percentage": float}, ...],
           "group_by": str, "agg_col": str,
           "highlighted": {"group_value": str, "percentage": float} | None}``
        On error:
        ``{"error": "<message>"}``
    """
    # ------------------------------------------------------------------ #
    # Validate inputs
    # ------------------------------------------------------------------ #
    if group_by not in df.columns:
        return {"error": f"column {group_by} not found"}

    if agg_col not in df.columns:
        return {"error": f"column {agg_col} not found"}

    if not pd.api.types.is_numeric_dtype(df[agg_col]):
        return {"error": f"column {agg_col} is not numeric"}

    # ------------------------------------------------------------------ #
    # Compute group sums and total
    # ------------------------------------------------------------------ #
    grouped: pd.Series = df.groupby(group_by)[agg_col].sum()
    total: float = float(grouped.sum())

    if total == 0:
        return {"error": f"total of column {agg_col} is zero — cannot compute percentage"}

    # ------------------------------------------------------------------ #
    # Build results list
    # ------------------------------------------------------------------ #
    results: list[dict[str, Any]] = []
    for gv, gsum in grouped.items():
        pct = round(float(gsum) / total * 100, 2)
        results.append({"group_value": str(gv), "percentage": pct})

    # Sort descending by percentage for readability
    results.sort(key=lambda x: x["percentage"], reverse=True)

    # ------------------------------------------------------------------ #
    # Highlight a specific group if requested
    # ------------------------------------------------------------------ #
    # Treat empty string the same as None — an empty group_value means
    # "return all groups" rather than "find the group named ''".
    if not group_value:
        group_value = None

    highlighted: dict[str, Any] | None = None
    if group_value is not None:
        match = next(
            (r for r in results if r["group_value"].lower() == str(group_value).lower()),
            None,
        )
        if match is None:
            return {
                "error": (
                    f"group value '{group_value}' not found in column '{group_by}'. "
                    f"Available values: {', '.join(r['group_value'] for r in results)}"
                )
            }
        highlighted = match

    return {
        "results": results,
        "group_by": group_by,
        "agg_col": agg_col,
        "highlighted": highlighted,
    }

"""
Dataset profiling tool.

Provides `dataset_profile(df)` which returns a deterministic structural and
statistical summary of a pandas DataFrame.  All statistics are computed by
pandas — no LLM output is involved.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _safe_stat(value: float) -> tuple[float | None, bool]:
    """Return (rounded_value, unavailable).

    Returns (None, True) when *value* is NaN or infinite (i.e. the stat cannot
    be meaningfully represented), otherwise returns (value rounded to 4 dp, False).
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None, True
    try:
        return round(float(value), 4), False
    except (TypeError, ValueError):
        return None, True


def dataset_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Profile a DataFrame: row count, column count, dtypes, and numeric stats.

    Parameters
    ----------
    df:
        A non-empty pandas DataFrame to profile.

    Returns
    -------
    dict with keys:
        ``row_count``     – number of rows (int)
        ``column_count``  – number of columns (int)
        ``columns``       – list of per-column metadata dicts
        ``stats``         – dict mapping numeric column name → stat dict
    """
    row_count: int = len(df)
    column_count: int = len(df.columns)

    # Per-column metadata (all columns)
    columns: list[dict[str, Any]] = [
        {
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
        }
        for col in df.columns
    ]

    # Descriptive stats for numeric columns only
    numeric_cols = df.select_dtypes(include="number").columns
    stats: dict[str, dict[str, Any]] = {}

    for col in numeric_cols:
        series = df[col]

        mean_val, mean_unavail = _safe_stat(series.mean())
        median_val, median_unavail = _safe_stat(series.median())
        min_val, min_unavail = _safe_stat(series.min())
        max_val, max_unavail = _safe_stat(series.max())
        std_val, std_unavail = _safe_stat(series.std())

        any_unavailable = any([mean_unavail, median_unavail, min_unavail, max_unavail, std_unavail])

        stats[col] = {
            "mean": mean_val,
            "median": median_val,
            "min": min_val,
            "max": max_val,
            "std": std_val,
            "unavailable": any_unavailable,
        }

    return {
        "row_count": row_count,
        "column_count": column_count,
        "columns": columns,
        "stats": stats,
    }

"""
Column type inference for uploaded CSV DataFrames.

Each column is classified as one of:
  - "numeric"     — pandas numeric dtype
  - "datetime"    — pandas datetime dtype, or ≥80% of non-null values parse as ISO dates
  - "categorical"  — everything else
"""

from __future__ import annotations

import pandas as pd

from app.models import ColumnSchema

# Minimum fraction of non-null values that must parse as dates for a string
# column to be promoted to "datetime".
_DATETIME_THRESHOLD = 0.80


def _classify_column(col_series: pd.Series) -> str:
    """Return the dtype label for a single Series."""
    # 1. Already a numeric dtype (int, float, complex, …)
    if pd.api.types.is_numeric_dtype(col_series):
        return "numeric"

    # 2. Already a proper datetime dtype
    if pd.api.types.is_datetime64_any_dtype(col_series):
        return "datetime"

    # 3. Try to parse string-like columns as datetimes
    try:
        parsed = pd.to_datetime(col_series, errors="coerce")
        non_null_total = col_series.notna().sum()
        if non_null_total > 0:
            parsed_ok = parsed.notna().sum()
            if parsed_ok / non_null_total >= _DATETIME_THRESHOLD:
                return "datetime"
    except Exception:
        # If anything unexpected happens during coercion, fall through to categorical
        pass

    # 4. Default
    return "categorical"


def infer_schema(df: pd.DataFrame) -> list[ColumnSchema]:
    """
    Infer a :class:`~app.models.ColumnSchema` for every column in *df*.

    Parameters
    ----------
    df:
        Any non-empty (or empty) pandas DataFrame produced by the CSV validator.

    Returns
    -------
    list[ColumnSchema]
        One entry per column, in the same order as ``df.columns``.
    """
    result: list[ColumnSchema] = []

    for col in df.columns:
        col_series = df[col]
        dtype = _classify_column(col_series)
        null_count = int(col_series.isna().sum())
        result.append(ColumnSchema(name=col, dtype=dtype, null_count=null_count))

    return result

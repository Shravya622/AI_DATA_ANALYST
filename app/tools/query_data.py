"""
Query data tool.

Provides `query_data(df, filters, columns, limit)` which filters and selects
rows from a pandas DataFrame based on column conditions and column selection.
No LLM output is involved — all computation is deterministic Pandas operations.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Supported filter operators mapped to their pandas equivalents.
_OPERATORS: dict[str, str] = {
    "==": "eq",
    "!=": "ne",
    ">": "gt",
    "<": "lt",
    ">=": "ge",
    "<=": "le",
}


def query_data(
    df: pd.DataFrame,
    filters: list[dict[str, Any]] | None = None,
    columns: list[str] | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Filter and select rows from a DataFrame.

    Parameters
    ----------
    df:
        The source DataFrame to query.
    filters:
        Optional list of filter dicts, each with keys:
        ``{"column": str, "operator": str, "value": Any}``.
        Supported operators: ``==``, ``!=``, ``>``, ``<``, ``>=``, ``<=``.
    columns:
        Optional list of column names to include in the output.
        If ``None``, all columns are returned.
    limit:
        Maximum number of rows to return. Defaults to 100.

    Returns
    -------
    dict with keys:
        ``rows``       – list of row dicts (each row is a ``{column: value}`` mapping)
        ``total_rows`` – total number of rows after filtering, before the ``limit`` cap
    OR an error dict:
        ``{"error": "column '<name>' not found"}`` when an unknown column is referenced.
    """
    result = df

    # ------------------------------------------------------------------ #
    # Apply filters
    # ------------------------------------------------------------------ #
    if filters:
        for f in filters:
            col: str = f["column"]
            operator: str = f["operator"]
            value: Any = f["value"]

            # Validate column existence.
            if col not in result.columns:
                return {"error": f"column '{col}' not found"}

            # Validate operator.
            if operator not in _OPERATORS:
                return {"error": f"unsupported operator '{operator}'"}

            # Apply the filter using the pandas Series method.
            pandas_method = _OPERATORS[operator]
            mask: pd.Series = getattr(result[col], pandas_method)(value)
            result = result[mask]

    # ------------------------------------------------------------------ #
    # Select columns
    # ------------------------------------------------------------------ #
    if columns is not None:
        for col in columns:
            if col not in result.columns:
                return {"error": f"column '{col}' not found"}
        result = result[columns]

    # ------------------------------------------------------------------ #
    # Capture total before applying limit
    # ------------------------------------------------------------------ #
    total_rows: int = len(result)

    # ------------------------------------------------------------------ #
    # Apply limit
    # ------------------------------------------------------------------ #
    result = result.head(limit)

    # ------------------------------------------------------------------ #
    # Serialise to list of dicts
    # ------------------------------------------------------------------ #
    rows: list[dict[str, Any]] = result.to_dict(orient="records")

    return {"rows": rows, "total_rows": total_rows}

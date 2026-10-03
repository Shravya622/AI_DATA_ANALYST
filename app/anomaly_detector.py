"""
Anomaly detection module for the AI-powered Data Analyst application.

Provides :func:`detect_anomalies` which flags statistical outliers in a
DataFrame using either the Z-score or IQR method.
"""

from __future__ import annotations

import pandas as pd

from app.models import AnomalyRecord, AnomalyReport


_NO_NUMERIC_MESSAGE = (
    "Anomaly detection requires at least one numeric column."
)


def detect_anomalies(
    df: pd.DataFrame,
    method: str = "zscore",
    threshold: float = 3.0,
) -> AnomalyReport:
    """Detect outlier rows in *df* and return an :class:`AnomalyReport`.

    Parameters
    ----------
    df:
        Input DataFrame to analyse.
    method:
        Detection strategy – ``"zscore"`` (default) or ``"iqr"``.
    threshold:
        For Z-score: rows with ``|z| > threshold`` are flagged.
        For IQR: the multiplier applied to IQR (default 1.5 is baked in;
        this parameter is currently unused for IQR but kept for API
        consistency).

    Returns
    -------
    AnomalyReport
        Contains ``count``, ``flagged_rows``, and an optional ``message``.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        return AnomalyReport(
            count=0,
            flagged_rows=[],
            message=_NO_NUMERIC_MESSAGE,
        )

    flagged: list[AnomalyRecord] = []

    if method == "zscore":
        flagged = _zscore_method(df, numeric_cols, threshold)
    else:
        flagged = _iqr_method(df, numeric_cols)

    return AnomalyReport(count=len(flagged), flagged_rows=flagged)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _zscore_method(
    df: pd.DataFrame,
    numeric_cols: list[str],
    threshold: float,
) -> list[AnomalyRecord]:
    records: list[AnomalyRecord] = []

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        mean = float(series.mean())
        std = float(series.std())

        if std == 0:
            # All values identical — nothing to flag.
            continue

        for idx in series.index:
            val = float(df.at[idx, col])
            z = (val - mean) / std
            if abs(z) > threshold:
                explanation = (
                    f"Value {val} in column '{col}' is "
                    f"{abs(z):.2f} standard deviations from the mean of {mean:.2f}"
                )
                records.append(
                    AnomalyRecord(
                        row_index=int(idx),
                        column=col,
                        value=val,
                        z_score=float(z),
                        explanation=explanation,
                    )
                )

    return records


def _iqr_method(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> list[AnomalyRecord]:
    records: list[AnomalyRecord] = []

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        for idx in series.index:
            val = float(df.at[idx, col])
            if val < lower or val > upper:
                explanation = (
                    f"Value {val} in column '{col}' is outside the "
                    f"expected range [{lower:.2f}, {upper:.2f}]"
                )
                records.append(
                    AnomalyRecord(
                        row_index=int(idx),
                        column=col,
                        value=val,
                        z_score=None,
                        explanation=explanation,
                    )
                )

    return records

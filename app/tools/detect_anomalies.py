"""
Detect Anomalies tool for the AI-powered Data Analyst.

Wraps ``app.anomaly_detector.detect_anomalies`` and serialises the
resulting :class:`~app.models.AnomalyReport` into a plain dict suitable
for LLM tool-call responses.
"""

from __future__ import annotations

import pandas as pd

from app.anomaly_detector import detect_anomalies


def detect_anomalies_tool(
    df: pd.DataFrame,
    method: str = "zscore",
    threshold: float = 3.0,
) -> dict:
    """Detect statistical anomalies in *df* and return a serialised report.

    Parameters
    ----------
    df:
        Source DataFrame to analyse.
    method:
        Detection strategy — ``"zscore"`` (default) or ``"iqr"``.
    threshold:
        For Z-score: rows where ``|z| > threshold`` are flagged (default 3.0).
        Unused for IQR but accepted for API consistency.

    Returns
    -------
    dict
        Serialised :class:`~app.models.AnomalyReport` with keys:
        ``count``, ``flagged_rows``, and optionally ``message``.
    """
    report = detect_anomalies(df, method=method, threshold=threshold)
    return report.model_dump()

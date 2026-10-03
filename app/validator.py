"""
CSV Validator for the AI-powered Data Analyst application.

Provides ``validate_csv`` which parses raw bytes into a pandas DataFrame
and enforces structural rules before the data is stored in the session.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import pandas as pd

# Maximum allowed upload size (bytes)
_MAX_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


@dataclass
class ValidationResult:
    """Result of a CSV validation attempt.

    Attributes:
        ok:        ``True`` when validation passed, ``False`` otherwise.
        error:     Human-readable error message when ``ok`` is ``False``;
                   ``None`` when ``ok`` is ``True``.
        dataframe: Parsed ``pd.DataFrame`` when ``ok`` is ``True``;
                   ``None`` when ``ok`` is ``False``.
    """

    ok: bool
    error: Optional[str]
    dataframe: Optional[pd.DataFrame]


def validate_csv(file_content: bytes, filename: str) -> ValidationResult:
    """Validate *file_content* as a CSV and return a :class:`ValidationResult`.

    Checks (in order):
    1. File size ≤ 50 MB.
    2. Bytes are parseable by ``pandas.read_csv``.
    3. At least 1 column.
    4. At least 1 data row.
    5. No empty column names.
    6. No duplicate column names.

    Parameters
    ----------
    file_content:
        Raw bytes of the uploaded file.
    filename:
        Original filename (used in error messages).

    Returns
    -------
    :class:`ValidationResult`
    """

    # ------------------------------------------------------------------ #
    # 1. File size check                                                   #
    # ------------------------------------------------------------------ #
    if len(file_content) > _MAX_SIZE_BYTES:
        size_mb = len(file_content) / (1024 * 1024)
        return ValidationResult(
            ok=False,
            error=(
                f"'{filename}' exceeds the 50 MB size limit "
                f"(file is {size_mb:.1f} MB)."
            ),
            dataframe=None,
        )

    # ------------------------------------------------------------------ #
    # 2. Parse CSV bytes                                                   #
    # ------------------------------------------------------------------ #
    try:
        df = pd.read_csv(io.BytesIO(file_content))
    except pd.errors.ParserError as exc:
        return ValidationResult(
            ok=False,
            error=(
                f"'{filename}' could not be parsed as a CSV file: {exc}"
            ),
            dataframe=None,
        )
    except UnicodeDecodeError as exc:
        return ValidationResult(
            ok=False,
            error=(
                f"'{filename}' contains invalid characters and could not be "
                f"decoded: {exc}"
            ),
            dataframe=None,
        )
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(
            ok=False,
            error=(
                f"'{filename}' could not be read: {exc}"
            ),
            dataframe=None,
        )

    # ------------------------------------------------------------------ #
    # 3. At least 1 column                                                 #
    # ------------------------------------------------------------------ #
    if df.shape[1] == 0:
        return ValidationResult(
            ok=False,
            error=f"'{filename}' contains no columns.",
            dataframe=None,
        )

    # ------------------------------------------------------------------ #
    # 4. At least 1 data row                                               #
    # ------------------------------------------------------------------ #
    if len(df) == 0:
        return ValidationResult(
            ok=False,
            error=f"'{filename}' contains no data rows.",
            dataframe=None,
        )

    # ------------------------------------------------------------------ #
    # Re-parse with mangle_dupe_cols=False to catch duplicates and empty  #
    # headers directly from the raw CSV text before pandas renames them.  #
    # ------------------------------------------------------------------ #
    # pandas.read_csv silently renames duplicates ("col" → "col.1") and
    # names empty headers "Unnamed: N".  We must inspect the *original*
    # header row ourselves.
    import csv as _csv

    try:
        text = file_content.decode("utf-8", errors="replace")
        reader = _csv.reader(iter(text.splitlines()))
        raw_headers: list[str] = next(reader, [])
    except Exception:
        raw_headers = list(df.columns)

    # ------------------------------------------------------------------ #
    # 5. No empty column names                                             #
    # ------------------------------------------------------------------ #
    empty_cols = [h for h in raw_headers if h.strip() == ""]
    if empty_cols:
        return ValidationResult(
            ok=False,
            error=(
                f"'{filename}' has one or more empty column names. "
                "All column headers must be non-empty strings."
            ),
            dataframe=None,
        )

    # ------------------------------------------------------------------ #
    # 6. No duplicate column names                                         #
    # ------------------------------------------------------------------ #
    seen: set[str] = set()
    duplicates: list[str] = []
    for h in raw_headers:
        if h in seen:
            if h not in duplicates:
                duplicates.append(h)
        else:
            seen.add(h)

    if duplicates:
        dup_list = ", ".join(f"'{d}'" for d in duplicates)
        return ValidationResult(
            ok=False,
            error=(
                f"'{filename}' has duplicate column names: {dup_list}. "
                "All column headers must be unique."
            ),
            dataframe=None,
        )

    # ------------------------------------------------------------------ #
    # All checks passed                                                    #
    # ------------------------------------------------------------------ #
    return ValidationResult(ok=True, error=None, dataframe=df)

"""
Context builder — produces a deterministic schema summary for LLM prompts.

The summary is prepended to every LLM system message so the model knows what
data is available without ever receiving raw DataFrame rows.

Public interface
----------------
build_schema_context(session: Session) -> str
    Return a deterministic text description of every dataset in *session*.
    Returns a fixed "no datasets" string when the session has no datasets.
"""

from app.models import Session


def build_schema_context(session: Session) -> str:
    """Build a deterministic text summary of all datasets in *session*.

    Used as the system message prefix for LLM calls so the model knows
    what data is available without receiving raw DataFrame rows.

    Parameters
    ----------
    session:
        The current user session containing zero or more ``DatasetRecord``
        entries.

    Returns
    -------
    str
        A human-readable, deterministic schema summary. If no datasets are
        loaded the string ``"No datasets are currently loaded in this session."``
        is returned.

    Notes
    -----
    * The function is pure: no side effects, no randomness.
    * Datasets are iterated in sorted filename order to guarantee determinism.
    * Column schemas are listed in the order stored in ``DatasetRecord``.
    """
    if not session.datasets:
        return "No datasets are currently loaded in this session."

    blocks: list[str] = []

    for filename in sorted(session.datasets.keys()):
        record = session.datasets[filename]
        num_cols = len(record.column_schemas)
        header = (
            f"Dataset: {filename} ({record.row_count} rows, {num_cols} columns)"
        )
        col_lines: list[str] = []
        for col in record.column_schemas:
            col_lines.append(
                f"  - {col.name}: {col.dtype} (null count: {col.null_count})"
            )
        blocks.append("\n".join([header] + col_lines))

    return "\n\n".join(blocks)

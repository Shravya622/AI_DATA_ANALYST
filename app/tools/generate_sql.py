"""
Generate SQL Tool — wraps the CodeGenerator to produce SQL snippets.

Returns a dict with ``snippet`` and ``language="sql"`` on success,
or ``snippet=None`` and an ``error`` message on failure.
Never raises an exception to the caller.
"""

from app.code_generator import generate_sql_snippet


def generate_sql(
    df,
    operation: str,
    table_name: str = "data",
    params: dict | None = None,
) -> dict:
    """Generate a SQL snippet for the given operation.

    Parameters
    ----------
    df:
        The active DataFrame (used for context only; never queried directly).
    operation:
        One of ``"aggregate"``, ``"filter"``, ``"select"``.
    table_name:
        SQL table name to reference in the generated snippet. Defaults to
        ``"data"``.
    params:
        Operation-specific parameters forwarded to the code generator.

    Returns
    -------
    dict
        On success: ``{"snippet": "<sql>", "language": "sql"}``
        On failure: ``{"snippet": None, "error": "<message>", "language": "sql"}``
    """
    params = params or {}
    snippet = generate_sql_snippet(operation, table_name, params)
    if snippet is None:
        return {
            "snippet": None,
            "error": f"Could not generate SQL for operation '{operation}'",
            "language": "sql",
        }
    return {"snippet": snippet, "language": "sql"}

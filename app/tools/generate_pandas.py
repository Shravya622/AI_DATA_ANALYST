"""
Generate Pandas Tool — wraps the CodeGenerator to produce Pandas code snippets.

Returns a dict with ``snippet`` and ``language="python"`` on success,
or ``snippet=None`` and an ``error`` message on failure.
Never raises an exception to the caller.
"""

from app.code_generator import generate_pandas_snippet


def generate_pandas(
    df,
    operation: str,
    params: dict | None = None,
) -> dict:
    """Generate a Pandas code snippet for the given operation.

    Parameters
    ----------
    df:
        The active DataFrame (used for context only; never executed against).
    operation:
        One of ``"aggregate"``, ``"filter"``, ``"select"``.
    params:
        Operation-specific parameters forwarded to the code generator.

    Returns
    -------
    dict
        On success: ``{"snippet": "<python>", "language": "python"}``
        On failure: ``{"snippet": None, "error": "<message>", "language": "python"}``
    """
    params = params or {}
    snippet = generate_pandas_snippet(operation, params)
    if snippet is None:
        return {
            "snippet": None,
            "error": f"Could not generate Pandas code for operation '{operation}'",
            "language": "python",
        }
    return {"snippet": snippet, "language": "python"}

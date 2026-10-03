"""
Code Generator — produces SQL and Pandas code snippets via string templates.

Never executes generated code. Validates Python with ast.parse and SQL with
sqlglot.parse. Retries once with an alternate template on validation failure.
Returns None if both attempts fail.
"""

import ast

import sqlglot

SUPPORTED_OPERATIONS = {"aggregate", "filter", "select"}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_python(snippet: str) -> bool:
    """Return True if *snippet* is syntactically valid Python."""
    try:
        ast.parse(snippet)
        return True
    except SyntaxError:
        return False


def _validate_sql(snippet: str) -> bool:
    """Return True if *snippet* can be parsed as SQL by sqlglot."""
    try:
        result = sqlglot.parse(snippet)
        return len(result) > 0 and result[0] is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pandas template builders
# ---------------------------------------------------------------------------


def _pandas_primary_template(operation: str, params: dict) -> str | None:
    """Build the primary Pandas snippet for *operation*."""
    if operation == "aggregate":
        group_by = params.get("group_by", "column")
        agg_col = params.get("agg_col", "value")
        agg_fn = params.get("agg_fn", "sum")
        return f"result = df.groupby('{group_by}')['{agg_col}'].{agg_fn}()"

    if operation == "filter":
        column = params.get("column", "column")
        operator = params.get("operator", "==")
        value = params.get("value", 0)
        return f"result = df[df['{column}'] {operator} {repr(value)}]"

    if operation == "select":
        columns = params.get("columns") or []
        cols_repr = ", ".join(repr(c) for c in columns)
        return f"result = df[[{cols_repr}]]"

    return None


def _pandas_alternate_template(operation: str, params: dict) -> str | None:
    """Build the alternate (chained) Pandas snippet for *operation*."""
    if operation == "aggregate":
        group_by = params.get("group_by", "column")
        agg_col = params.get("agg_col", "value")
        agg_fn = params.get("agg_fn", "sum")
        return (
            f"result = (\n"
            f"    df\n"
            f"    .groupby('{group_by}')['{agg_col}']\n"
            f"    .{agg_fn}()\n"
            f")"
        )

    if operation == "filter":
        column = params.get("column", "column")
        operator = params.get("operator", "==")
        value = params.get("value", 0)
        return (
            f"mask = df['{column}'] {operator} {repr(value)}\n"
            f"result = df[mask]"
        )

    if operation == "select":
        columns = params.get("columns") or []
        cols_repr = ", ".join(repr(c) for c in columns)
        return (
            f"cols = [{cols_repr}]\n"
            f"result = df[cols]"
        )

    return None


# ---------------------------------------------------------------------------
# SQL template builders
# ---------------------------------------------------------------------------


def _sql_primary_template(operation: str, table_name: str, params: dict) -> str | None:
    """Build the primary SQL snippet for *operation*."""
    if operation == "aggregate":
        group_by = params.get("group_by", "column")
        agg_col = params.get("agg_col", "value")
        agg_fn = params.get("agg_fn", "SUM").upper()
        return (
            f"SELECT {group_by}, {agg_fn}({agg_col}) AS agg_result "
            f"FROM {table_name} "
            f"GROUP BY {group_by}"
        )

    if operation == "filter":
        column = params.get("column", "column")
        operator = params.get("operator", "=")
        value = params.get("value", 0)
        # Format value for SQL: strings get single-quoted, numbers are bare
        if isinstance(value, str):
            sql_value = f"'{value}'"
        else:
            sql_value = repr(value)
        return f"SELECT * FROM {table_name} WHERE {column} {operator} {sql_value}"

    if operation == "select":
        columns = params.get("columns") or []
        cols_str = ", ".join(columns) if columns else "*"
        return f"SELECT {cols_str} FROM {table_name}"

    return None


def _sql_alternate_template(operation: str, table_name: str, params: dict) -> str | None:
    """Build the alternate SQL snippet for *operation*."""
    if operation == "aggregate":
        group_by = params.get("group_by", "column")
        agg_col = params.get("agg_col", "value")
        agg_fn = params.get("agg_fn", "SUM").upper()
        return (
            f"SELECT\n"
            f"    {group_by},\n"
            f"    {agg_fn}({agg_col}) AS agg_result\n"
            f"FROM {table_name}\n"
            f"GROUP BY {group_by}"
        )

    if operation == "filter":
        column = params.get("column", "column")
        operator = params.get("operator", "=")
        value = params.get("value", 0)
        if isinstance(value, str):
            sql_value = f"'{value}'"
        else:
            sql_value = repr(value)
        return (
            f"SELECT *\n"
            f"FROM {table_name}\n"
            f"WHERE {column} {operator} {sql_value}"
        )

    if operation == "select":
        columns = params.get("columns") or []
        cols_str = ", ".join(columns) if columns else "*"
        return f"SELECT\n    {cols_str}\nFROM {table_name}"

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_pandas_snippet(operation: str, params: dict) -> str | None:
    """Generate a syntactically valid Pandas code snippet using string templates.

    Validates with ast.parse. Retries once with an alternate template on failure.
    Never executes the generated code. Returns None if both attempts fail.
    """
    primary = _pandas_primary_template(operation, params)
    if primary and _validate_python(primary):
        return primary

    alternate = _pandas_alternate_template(operation, params)
    if alternate and _validate_python(alternate):
        return alternate

    return None


def generate_sql_snippet(operation: str, table_name: str, params: dict) -> str | None:
    """Generate valid SQLite-compatible SQL using string templates.

    Validates with sqlglot.parse. Retries once with an alternate template on
    failure. Never executes the generated code. Returns None if both attempts fail.
    """
    primary = _sql_primary_template(operation, table_name, params)
    if primary and _validate_sql(primary):
        return primary

    alternate = _sql_alternate_template(operation, table_name, params)
    if alternate and _validate_sql(alternate):
        return alternate

    return None

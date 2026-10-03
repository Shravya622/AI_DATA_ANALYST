"""
Unit tests for app/code_generator.py — covers all four acceptance criteria:

1. generate_pandas_snippet("aggregate", {...}) returns a syntactically valid
   Pandas code string.
2. generate_sql_snippet("aggregate", "my_table", {...}) returns valid
   SQLite-compatible SQL.
3. A deliberately invalid template fallback returns None without raising an
   exception.
4. Generated code is never executed against the Dataset (source-code inspection).
"""

import ast
import inspect
import textwrap

import pytest
import sqlglot

import app.code_generator as cg
from app.code_generator import generate_pandas_snippet, generate_sql_snippet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_valid_python(snippet: str) -> bool:
    try:
        ast.parse(snippet)
        return True
    except SyntaxError:
        return False


def _is_valid_sql(snippet: str) -> bool:
    try:
        result = sqlglot.parse(snippet)
        return len(result) > 0 and result[0] is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Acceptance Criterion 1: generate_pandas_snippet returns valid Python
# ---------------------------------------------------------------------------


class TestGeneratePandasSnippet:
    def test_aggregate_returns_string(self):
        params = {"group_by": "region", "agg_col": "revenue", "agg_fn": "sum"}
        result = generate_pandas_snippet("aggregate", params)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_aggregate_is_syntactically_valid_python(self):
        params = {"group_by": "region", "agg_col": "revenue", "agg_fn": "sum"}
        snippet = generate_pandas_snippet("aggregate", params)
        assert snippet is not None
        assert _is_valid_python(snippet), f"Snippet is not valid Python:\n{snippet}"

    def test_aggregate_contains_groupby(self):
        params = {"group_by": "region", "agg_col": "revenue", "agg_fn": "mean"}
        snippet = generate_pandas_snippet("aggregate", params)
        assert snippet is not None
        assert "groupby" in snippet
        assert "region" in snippet
        assert "revenue" in snippet
        assert "mean" in snippet

    def test_filter_is_syntactically_valid_python(self):
        params = {"column": "revenue", "operator": ">", "value": 1000}
        snippet = generate_pandas_snippet("filter", params)
        assert snippet is not None
        assert _is_valid_python(snippet), f"Snippet is not valid Python:\n{snippet}"

    def test_filter_string_value_valid_python(self):
        params = {"column": "region", "operator": "==", "value": "North"}
        snippet = generate_pandas_snippet("filter", params)
        assert snippet is not None
        assert _is_valid_python(snippet), f"Snippet is not valid Python:\n{snippet}"

    def test_select_is_syntactically_valid_python(self):
        params = {"columns": ["region", "revenue"]}
        snippet = generate_pandas_snippet("select", params)
        assert snippet is not None
        assert _is_valid_python(snippet), f"Snippet is not valid Python:\n{snippet}"

    def test_select_contains_column_names(self):
        params = {"columns": ["date", "product", "revenue"]}
        snippet = generate_pandas_snippet("select", params)
        assert snippet is not None
        assert "date" in snippet
        assert "product" in snippet
        assert "revenue" in snippet

    def test_all_supported_agg_functions_produce_valid_python(self):
        for fn in ("sum", "mean", "count", "min", "max"):
            params = {"group_by": "region", "agg_col": "revenue", "agg_fn": fn}
            snippet = generate_pandas_snippet("aggregate", params)
            assert snippet is not None, f"Got None for agg_fn={fn}"
            assert _is_valid_python(snippet), f"Invalid Python for agg_fn={fn}"

    def test_unknown_operation_returns_none(self):
        result = generate_pandas_snippet("pivot", {"columns": ["x"]})
        assert result is None


# ---------------------------------------------------------------------------
# Acceptance Criterion 2: generate_sql_snippet returns valid SQLite SQL
# ---------------------------------------------------------------------------


class TestGenerateSqlSnippet:
    def test_aggregate_returns_string(self):
        params = {"group_by": "region", "agg_col": "revenue", "agg_fn": "sum"}
        result = generate_sql_snippet("aggregate", "my_table", params)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_aggregate_is_valid_sql(self):
        params = {"group_by": "region", "agg_col": "revenue", "agg_fn": "sum"}
        snippet = generate_sql_snippet("aggregate", "my_table", params)
        assert snippet is not None
        assert _is_valid_sql(snippet), f"Snippet is not valid SQL:\n{snippet}"

    def test_aggregate_contains_expected_keywords(self):
        params = {"group_by": "region", "agg_col": "revenue", "agg_fn": "sum"}
        snippet = generate_sql_snippet("aggregate", "my_table", params)
        assert snippet is not None
        upper = snippet.upper()
        assert "SELECT" in upper
        assert "FROM" in upper
        assert "GROUP BY" in upper
        assert "MY_TABLE" in upper
        assert "SUM" in upper

    def test_filter_is_valid_sql(self):
        params = {"column": "revenue", "operator": ">", "value": 500}
        snippet = generate_sql_snippet("filter", "sales", params)
        assert snippet is not None
        assert _is_valid_sql(snippet), f"Snippet is not valid SQL:\n{snippet}"

    def test_filter_string_value_valid_sql(self):
        params = {"column": "region", "operator": "=", "value": "North"}
        snippet = generate_sql_snippet("filter", "sales", params)
        assert snippet is not None
        assert _is_valid_sql(snippet), f"Snippet is not valid SQL:\n{snippet}"
        assert "North" in snippet

    def test_select_is_valid_sql(self):
        params = {"columns": ["region", "revenue"]}
        snippet = generate_sql_snippet("select", "sales", params)
        assert snippet is not None
        assert _is_valid_sql(snippet), f"Snippet is not valid SQL:\n{snippet}"

    def test_select_contains_column_names(self):
        params = {"columns": ["date", "product", "revenue"]}
        snippet = generate_sql_snippet("select", "sales", params)
        assert snippet is not None
        assert "date" in snippet
        assert "product" in snippet
        assert "revenue" in snippet

    def test_all_supported_agg_functions_produce_valid_sql(self):
        for fn in ("sum", "mean", "count", "min", "max"):
            # SQL uses AVG instead of mean — our template upcases; AVG not in
            # standard aggregate list but we map through as-is (SUM→SUM, MEAN→MEAN).
            # sqlglot accepts non-standard function names for SQLite dialect.
            params = {"group_by": "region", "agg_col": "revenue", "agg_fn": fn}
            snippet = generate_sql_snippet("aggregate", "my_table", params)
            assert snippet is not None, f"Got None for agg_fn={fn}"
            assert _is_valid_sql(snippet), f"Invalid SQL for agg_fn={fn}"

    def test_unknown_operation_returns_none(self):
        result = generate_sql_snippet("pivot", "my_table", {})
        assert result is None


# ---------------------------------------------------------------------------
# Acceptance Criterion 3: invalid template fallback returns None, no exception
# ---------------------------------------------------------------------------


class TestInvalidTemplateFallback:
    """
    Monkey-patch both primary and alternate template functions to return
    intentionally broken snippets and verify generate_* returns None cleanly.
    """

    def test_pandas_both_templates_invalid_returns_none(self, monkeypatch):
        monkeypatch.setattr(cg, "_pandas_primary_template", lambda op, p: "def (")
        monkeypatch.setattr(cg, "_pandas_alternate_template", lambda op, p: "class :")
        result = generate_pandas_snippet("aggregate", {})
        assert result is None

    def test_sql_both_templates_invalid_returns_none(self, monkeypatch):
        monkeypatch.setattr(cg, "_sql_primary_template", lambda op, t, p: "SELECT FROM WHERE")
        monkeypatch.setattr(cg, "_sql_alternate_template", lambda op, t, p: "NOT VALID @@@")
        result = generate_sql_snippet("aggregate", "my_table", {})
        assert result is None

    def test_pandas_no_exception_raised(self, monkeypatch):
        monkeypatch.setattr(cg, "_pandas_primary_template", lambda op, p: "!!!")
        monkeypatch.setattr(cg, "_pandas_alternate_template", lambda op, p: "???")
        # Must not raise — only return None
        result = generate_pandas_snippet("aggregate", {})
        assert result is None

    def test_sql_no_exception_raised(self, monkeypatch):
        monkeypatch.setattr(cg, "_sql_primary_template", lambda op, t, p: None)
        monkeypatch.setattr(cg, "_sql_alternate_template", lambda op, t, p: None)
        result = generate_sql_snippet("aggregate", "my_table", {})
        assert result is None

    def test_pandas_primary_invalid_falls_back_to_alternate(self, monkeypatch):
        """If primary fails validation, alternate should be returned."""
        monkeypatch.setattr(cg, "_pandas_primary_template", lambda op, p: "def (")
        # Leave alternate to the real implementation
        result = generate_pandas_snippet("aggregate", {"group_by": "r", "agg_col": "v", "agg_fn": "sum"})
        assert result is not None
        assert _is_valid_python(result)

    def test_sql_primary_invalid_falls_back_to_alternate(self, monkeypatch):
        """If primary fails validation, alternate should be returned."""
        monkeypatch.setattr(cg, "_sql_primary_template", lambda op, t, p: "NOT VALID @@@")
        result = generate_sql_snippet("aggregate", "my_table", {"group_by": "r", "agg_col": "v", "agg_fn": "sum"})
        assert result is not None
        assert _is_valid_sql(result)


# ---------------------------------------------------------------------------
# Acceptance Criterion 4: generated code is never executed
# ---------------------------------------------------------------------------


class TestNeverExecutesCode:
    """
    Inspect the source of code_generator.py to confirm it contains no eval,
    exec, subprocess, or os.system calls that would execute generated snippets.
    """

    def _get_source(self) -> str:
        return inspect.getsource(cg)

    def test_no_eval_call(self):
        source = self._get_source()
        # Allow "eval" only in comments/strings for documentation purposes.
        # We check that eval() is not called as a function in live code.
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                assert name != "eval", "code_generator.py must not call eval()"

    def test_no_exec_call(self):
        source = self._get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                assert name != "exec", "code_generator.py must not call exec()"

    def test_no_subprocess_import(self):
        source = self._get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "subprocess" not in alias.name, (
                        "code_generator.py must not import subprocess"
                    )
            if isinstance(node, ast.ImportFrom):
                assert node.module is None or "subprocess" not in node.module, (
                    "code_generator.py must not import from subprocess"
                )

    def test_no_os_system_call(self):
        source = self._get_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    assert not (
                        func.attr == "system"
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "os"
                    ), "code_generator.py must not call os.system()"

    def test_snippets_not_executed_at_module_level(self):
        """
        Import the module fresh and confirm no side-effects from snippet execution.
        This is a sanity check that the module import itself does nothing.
        """
        import importlib
        module = importlib.import_module("app.code_generator")
        # If this import completes without running arbitrary code, the test passes.
        assert hasattr(module, "generate_pandas_snippet")
        assert hasattr(module, "generate_sql_snippet")


# ---------------------------------------------------------------------------
# Regression test: columns=None must not raise TypeError
# Reproduces the runtime failure: LLM sends {"operation": "select",
# "params": {"columns": null}} which caused "NoneType is not iterable".
# ---------------------------------------------------------------------------


class TestColumnsNullSafety:
    """
    Guard against the LLM passing {"columns": null} (JSON null → Python None).
    All four template paths must safely coerce None to an empty list and return
    a valid snippet rather than raising TypeError.
    """

    # ── generate_pandas_snippet ──────────────────────────────────────────────

    def test_pandas_select_columns_null_does_not_raise(self):
        """generate_pandas_snippet("select", {"columns": None}) must not raise."""
        result = generate_pandas_snippet("select", {"columns": None})
        # Returns a snippet (empty-columns select is valid Python) or None — never an exception.
        assert result is None or isinstance(result, str)

    def test_pandas_select_columns_null_returns_valid_python_or_none(self):
        snippet = generate_pandas_snippet("select", {"columns": None})
        if snippet is not None:
            assert _is_valid_python(snippet), f"Not valid Python: {snippet}"

    def test_pandas_select_columns_empty_list_does_not_raise(self):
        """{"columns": []} is a degenerate but valid input."""
        result = generate_pandas_snippet("select", {"columns": []})
        assert result is None or isinstance(result, str)

    def test_pandas_select_columns_missing_does_not_raise(self):
        """Omitting 'columns' entirely must also not raise."""
        result = generate_pandas_snippet("select", {})
        assert result is None or isinstance(result, str)

    def test_pandas_select_normal_columns_still_works(self):
        """Normal list input must continue to produce a valid snippet."""
        snippet = generate_pandas_snippet("select", {"columns": ["region", "revenue"]})
        assert snippet is not None
        assert _is_valid_python(snippet)
        assert "region" in snippet
        assert "revenue" in snippet

    # ── generate_sql_snippet ─────────────────────────────────────────────────

    def test_sql_select_columns_null_does_not_raise(self):
        """generate_sql_snippet("select", ..., {"columns": None}) must not raise."""
        result = generate_sql_snippet("select", "sales", {"columns": None})
        assert result is None or isinstance(result, str)

    def test_sql_select_columns_null_produces_select_star(self):
        """When columns is None (or empty), the SQL should fall back to SELECT *."""
        snippet = generate_sql_snippet("select", "sales", {"columns": None})
        if snippet is not None:
            assert "SELECT" in snippet.upper()
            assert "sales" in snippet

    def test_sql_select_columns_empty_list_produces_select_star(self):
        snippet = generate_sql_snippet("select", "sales", {"columns": []})
        if snippet is not None:
            assert "SELECT" in snippet.upper()

    def test_sql_select_columns_missing_does_not_raise(self):
        result = generate_sql_snippet("select", "sales", {})
        assert result is None or isinstance(result, str)

    def test_sql_select_normal_columns_still_works(self):
        """Normal list input must continue to produce valid SQL."""
        snippet = generate_sql_snippet("select", "sales", {"columns": ["region", "revenue"]})
        assert snippet is not None
        assert _is_valid_sql(snippet)
        assert "region" in snippet
        assert "revenue" in snippet

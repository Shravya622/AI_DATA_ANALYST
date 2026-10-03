"""
Unit tests for app/tools/registry.py — Task 6.2 Tool Registry.

Covers all acceptance criteria:
- dispatch("dataset_profile", {...}, session) calls the correct function and returns its result.
- dispatch("nonexistent_tool", {...}, session) raises UnknownToolError.
- TOOLS contains JSON schemas for all registered tools.
- The registry module contains no eval, exec, or subprocess calls.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.exceptions import UnknownToolError
from app.models import Session
from app.tools.registry import (
    TOOLS,
    ToolDefinition,
    dispatch,
    get_tool_schemas,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_session(df: pd.DataFrame | None = None, filename: str = "test.csv") -> Session:
    """Build a minimal Session with one optional DataFrame."""
    dataframes: dict[str, pd.DataFrame] = {}
    if df is not None:
        dataframes[filename] = df

    return Session(
        session_id="test-session-id",
        datasets={},
        dataframes=dataframes,
        history=[],
    )


def _sample_df() -> pd.DataFrame:
    """Return a small mixed-type DataFrame for testing."""
    return pd.DataFrame(
        {
            "region": ["North", "South", "East", "West", "North"],
            "revenue": [100.0, 200.0, 150.0, 300.0, 250.0],
            "units_sold": [10, 20, 15, 30, 25],
        }
    )


# ---------------------------------------------------------------------------
# 1. TOOLS registry structure
# ---------------------------------------------------------------------------


class TestToolsRegistry:
    """Verify the TOOLS dict is correctly populated."""

    EXPECTED_TOOLS = {
        "dataset_profile",
        "query_data",
        "aggregate_data",
        "generate_summary",
        "generate_chart",
        "detect_anomalies",
        "generate_sql",
        "generate_pandas",
        "percentage_of_total",
    }

    def test_tools_contains_all_expected_keys(self):
        assert set(TOOLS.keys()) == self.EXPECTED_TOOLS

    def test_each_entry_is_tool_definition(self):
        for name, tool_def in TOOLS.items():
            assert isinstance(tool_def, ToolDefinition), (
                f"TOOLS['{name}'] is not a ToolDefinition"
            )

    def test_each_tool_definition_has_callable(self):
        for name, tool_def in TOOLS.items():
            assert callable(tool_def.callable), (
                f"TOOLS['{name}'].callable is not callable"
            )

    def test_each_tool_definition_has_description(self):
        for name, tool_def in TOOLS.items():
            assert isinstance(tool_def.description, str), (
                f"TOOLS['{name}'].description is not a str"
            )
            assert len(tool_def.description) > 0, (
                f"TOOLS['{name}'].description is empty"
            )

    def test_each_tool_has_json_schema(self):
        """TOOLS contains JSON schemas for all registered tools."""
        for name, tool_def in TOOLS.items():
            schema = tool_def.json_schema
            assert isinstance(schema, dict), (
                f"TOOLS['{name}'].json_schema is not a dict"
            )

    def test_json_schemas_are_openai_function_format(self):
        """Each JSON schema uses the OpenAI function-calling format."""
        for name, tool_def in TOOLS.items():
            schema = tool_def.json_schema
            assert schema.get("type") == "function", (
                f"TOOLS['{name}'].json_schema missing type='function'"
            )
            assert "function" in schema, (
                f"TOOLS['{name}'].json_schema missing 'function' key"
            )
            fn_block = schema["function"]
            assert "name" in fn_block, (
                f"TOOLS['{name}'].json_schema['function'] missing 'name'"
            )
            assert fn_block["name"] == name, (
                f"TOOLS['{name}'].json_schema['function']['name'] != '{name}'"
            )
            assert "description" in fn_block, (
                f"TOOLS['{name}'].json_schema['function'] missing 'description'"
            )
            assert "parameters" in fn_block, (
                f"TOOLS['{name}'].json_schema['function'] missing 'parameters'"
            )

    def test_json_schema_parameters_are_objects(self):
        for name, tool_def in TOOLS.items():
            params = tool_def.json_schema["function"]["parameters"]
            assert params.get("type") == "object", (
                f"TOOLS['{name}'] parameters type is not 'object'"
            )
            assert "properties" in params, (
                f"TOOLS['{name}'] parameters missing 'properties'"
            )


# ---------------------------------------------------------------------------
# 2. get_tool_schemas()
# ---------------------------------------------------------------------------


class TestGetToolSchemas:
    def test_returns_list(self):
        schemas = get_tool_schemas()
        assert isinstance(schemas, list)

    def test_length_matches_tools_count(self):
        schemas = get_tool_schemas()
        assert len(schemas) == len(TOOLS)

    def test_all_schemas_are_dicts(self):
        for schema in get_tool_schemas():
            assert isinstance(schema, dict)

    def test_each_schema_has_function_type(self):
        for schema in get_tool_schemas():
            assert schema.get("type") == "function"

    def test_schemas_cover_all_tool_names(self):
        schema_names = {s["function"]["name"] for s in get_tool_schemas()}
        assert schema_names == set(TOOLS.keys())


# ---------------------------------------------------------------------------
# 3. dispatch — happy paths
# ---------------------------------------------------------------------------


class TestDispatchHappyPath:
    """Acceptance criterion: dispatch("dataset_profile", {...}, session) calls
    the correct function and returns its result."""

    def test_dispatch_dataset_profile_calls_correct_function(self):
        df = _sample_df()
        session = _make_session(df)

        result = dispatch("dataset_profile", {}, session)

        assert isinstance(result, dict)
        assert result["row_count"] == 5
        assert result["column_count"] == 3

    def test_dispatch_dataset_profile_with_explicit_filename(self):
        df = _sample_df()
        session = _make_session(df, filename="sales.csv")

        result = dispatch("dataset_profile", {"dataset_filename": "sales.csv"}, session)

        assert result["row_count"] == 5

    def test_dispatch_generate_summary_returns_profile_and_insights(self):
        df = _sample_df()
        session = _make_session(df)

        result = dispatch("generate_summary", {}, session)

        assert "profile" in result
        assert "insights" in result

    def test_dispatch_query_data_with_filters(self):
        df = _sample_df()
        session = _make_session(df)

        result = dispatch(
            "query_data",
            {"filters": [{"column": "revenue", "operator": ">", "value": 150}]},
            session,
        )

        assert "rows" in result
        # Only rows where revenue > 150: 200, 300, 250 → 3 rows
        assert result["total_rows"] == 3

    def test_dispatch_query_data_with_columns_and_limit(self):
        df = _sample_df()
        session = _make_session(df)

        result = dispatch(
            "query_data",
            {"columns": ["region", "revenue"], "limit": 2},
            session,
        )

        assert len(result["rows"]) == 2
        for row in result["rows"]:
            assert set(row.keys()) == {"region", "revenue"}

    def test_dispatch_aggregate_data_sum(self):
        df = _sample_df()
        session = _make_session(df)

        result = dispatch(
            "aggregate_data",
            {"group_by": "region", "agg_col": "revenue", "agg_fn": "sum"},
            session,
        )

        assert "results" in result
        assert result["agg_fn"] == "sum"

    def test_dispatch_aggregate_data_mean(self):
        df = _sample_df()
        session = _make_session(df)

        result = dispatch(
            "aggregate_data",
            {"group_by": "region", "agg_col": "units_sold", "agg_fn": "mean"},
            session,
        )

        assert result["group_by"] == "region"
        assert result["agg_col"] == "units_sold"

    def test_dispatch_falls_back_to_first_df_when_no_filename(self):
        """When dataset_filename is not in arguments, the first available df is used."""
        df = _sample_df()
        session = _make_session(df, filename="data.csv")

        result = dispatch("dataset_profile", {}, session)

        assert result["row_count"] == 5

    def test_dispatch_falls_back_to_first_df_when_filename_unknown(self):
        df = _sample_df()
        session = _make_session(df, filename="data.csv")

        result = dispatch("dataset_profile", {"dataset_filename": "nonexistent.csv"}, session)

        assert result["row_count"] == 5

    def test_dispatch_uses_correct_df_when_multiple_dataframes(self):
        """When multiple DataFrames are in the session, the correct one is chosen."""
        df_small = pd.DataFrame({"x": [1, 2]})
        df_large = pd.DataFrame({"y": [10, 20, 30, 40, 50]})

        session = Session(
            session_id="multi-session",
            datasets={},
            dataframes={"small.csv": df_small, "large.csv": df_large},
            history=[],
        )

        result = dispatch("dataset_profile", {"dataset_filename": "large.csv"}, session)
        assert result["row_count"] == 5

        result2 = dispatch("dataset_profile", {"dataset_filename": "small.csv"}, session)
        assert result2["row_count"] == 2


# ---------------------------------------------------------------------------
# 4. dispatch — UnknownToolError
# ---------------------------------------------------------------------------


class TestDispatchUnknownTool:
    """Acceptance criterion: dispatch("nonexistent_tool", {...}, session) raises UnknownToolError."""

    def test_dispatch_raises_unknown_tool_error_for_unregistered_name(self):
        session = _make_session(_sample_df())

        with pytest.raises(UnknownToolError):
            dispatch("nonexistent_tool", {}, session)

    def test_dispatch_raises_unknown_tool_error_for_empty_string(self):
        session = _make_session(_sample_df())

        with pytest.raises(UnknownToolError):
            dispatch("", {}, session)

    def test_dispatch_raises_unknown_tool_error_for_similar_name(self):
        """Typos / close-but-wrong names must also raise UnknownToolError."""
        session = _make_session(_sample_df())

        with pytest.raises(UnknownToolError):
            dispatch("dataset_profiles", {}, session)  # trailing 's'

    def test_unknown_tool_error_message_is_descriptive(self):
        session = _make_session(_sample_df())

        with pytest.raises(UnknownToolError) as exc_info:
            dispatch("bad_tool", {}, session)

        assert "bad_tool" in exc_info.value.message

    def test_dispatch_does_not_silently_succeed_for_unknown_tool(self):
        """dispatch must not return a result for an unknown tool."""
        session = _make_session(_sample_df())
        raised = False

        try:
            dispatch("totally_unknown", {}, session)
        except UnknownToolError:
            raised = True

        assert raised, "UnknownToolError was not raised for an unknown tool name"


# ---------------------------------------------------------------------------
# 5. Security: no eval / exec / subprocess in registry module
# ---------------------------------------------------------------------------


class TestNoForbiddenCalls:
    """Acceptance criterion: the registry module contains no eval, exec, or subprocess calls."""

    def _get_registry_source(self) -> str:
        registry_path = Path(__file__).parent.parent.parent / "app" / "tools" / "registry.py"
        return registry_path.read_text(encoding="utf-8")

    def _get_registry_ast(self) -> ast.Module:
        source = self._get_registry_source()
        return ast.parse(source)

    def test_no_eval_call(self):
        source = self._get_registry_source()
        assert "eval(" not in source, "registry.py contains a call to eval()"

    def test_no_exec_call(self):
        source = self._get_registry_source()
        assert "exec(" not in source, "registry.py contains a call to exec()"

    def test_no_subprocess_import(self):
        """Check AST-level imports — the word may appear in docstrings, but no actual import."""
        tree = self._get_registry_ast()
        for node in ast.walk(tree):
            # import subprocess
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "subprocess", (
                        "registry.py contains 'import subprocess'"
                    )
            # from subprocess import ...
            if isinstance(node, ast.ImportFrom):
                assert node.module != "subprocess", (
                    "registry.py contains 'from subprocess import ...'"
                )

    def test_no_eval_in_ast(self):
        """AST-level check: no Name node with id='eval'."""
        tree = self._get_registry_ast()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "eval":
                    pytest.fail("registry.py AST contains an eval() call")

    def test_no_exec_in_ast(self):
        """AST-level check: no Name node with id='exec'."""
        tree = self._get_registry_ast()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "exec":
                    pytest.fail("registry.py AST contains an exec() call")


# ---------------------------------------------------------------------------
# 6. ToolDefinition dataclass fields
# ---------------------------------------------------------------------------


class TestToolDefinitionDataclass:
    def test_tool_definition_has_callable_field(self):
        td = ToolDefinition(
            callable=lambda df: {},
            json_schema={"type": "function", "function": {"name": "x", "description": "", "parameters": {"type": "object", "properties": {}}}},
            description="test",
        )
        assert callable(td.callable)

    def test_tool_definition_has_json_schema_field(self):
        schema = {"type": "function", "function": {"name": "x", "description": "", "parameters": {}}}
        td = ToolDefinition(callable=lambda: None, json_schema=schema, description="d")
        assert td.json_schema is schema

    def test_tool_definition_has_description_field(self):
        td = ToolDefinition(callable=lambda: None, json_schema={}, description="my description")
        assert td.description == "my description"


# ---------------------------------------------------------------------------
# 7. Aggregate data tool-specific schema validation
# ---------------------------------------------------------------------------


class TestAggregateDataSchema:
    def test_aggregate_data_schema_requires_group_by_agg_col_agg_fn(self):
        schema = TOOLS["aggregate_data"].json_schema
        required = schema["function"]["parameters"].get("required", [])
        assert "group_by" in required
        assert "agg_col" in required
        assert "agg_fn" in required

    def test_aggregate_data_agg_fn_enum_contains_valid_values(self):
        schema = TOOLS["aggregate_data"].json_schema
        props = schema["function"]["parameters"]["properties"]
        agg_fn_enum = props["agg_fn"].get("enum", [])
        assert set(agg_fn_enum) == {"sum", "mean", "count", "min", "max"}

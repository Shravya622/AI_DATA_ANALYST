"""
Tool Registry for the AI-powered Data Analyst.

Defines `ToolDefinition`, the `TOOLS` registry dict, the `dispatch` function,
and `get_tool_schemas()` helper.

Security: No `eval`, `exec`, or `subprocess` calls anywhere in this module.
All tool invocations go through explicit, pre-registered callables only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.exceptions import UnknownToolError
from app.models import Session
from app.tools.aggregate_data import aggregate_data
from app.tools.dataset_profile import dataset_profile
from app.tools.detect_anomalies import detect_anomalies_tool
from app.tools.generate_chart import generate_chart
from app.tools.generate_pandas import generate_pandas
from app.tools.generate_sql import generate_sql
from app.tools.generate_summary import generate_summary
from app.tools.percentage_of_total import percentage_of_total
from app.tools.query_data import query_data


# ---------------------------------------------------------------------------
# ToolDefinition
# ---------------------------------------------------------------------------


@dataclass
class ToolDefinition:
    """Metadata and callable for a single registered tool."""

    callable: Callable[..., dict[str, Any]]
    json_schema: dict[str, Any]   # OpenAI function-calling schema format
    description: str


# ---------------------------------------------------------------------------
# TOOLS registry
# ---------------------------------------------------------------------------

TOOLS: dict[str, ToolDefinition] = {
    "dataset_profile": ToolDefinition(
        callable=dataset_profile,
        description=(
            "Profile a dataset: returns row count, column count, column dtypes, "
            "null counts, and descriptive statistics for numeric columns."
        ),
        json_schema={
            "type": "function",
            "function": {
                "name": "dataset_profile",
                "description": (
                    "Profile a dataset: returns row count, column count, column "
                    "dtypes, null counts, and descriptive statistics for numeric "
                    "columns."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_filename": {
                            "type": "string",
                            "description": (
                                "The filename of the uploaded dataset to profile. "
                                "If omitted, the first available dataset is used."
                            ),
                        },
                        "combine_all_datasets": {
                            "type": "boolean",
                            "description": (
                                "Set to true when the user asks to analyse ALL uploaded datasets "
                                "together (e.g. 'across all datasets', 'across all uploaded files', "
                                "'combined', 'all files'). False by default."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        },
    ),

    "query_data": ToolDefinition(
        callable=query_data,
        description=(
            "Filter and select rows from a dataset based on column conditions "
            "and column selection, returning matching rows up to a limit."
        ),
        json_schema={
            "type": "function",
            "function": {
                "name": "query_data",
                "description": (
                    "Filter and select rows from a dataset based on column "
                    "conditions and column selection."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_filename": {
                            "type": "string",
                            "description": "The filename of the dataset to query.",
                        },
                        "filters": {
                            "type": "array",
                            "description": (
                                "List of filter conditions. Each filter is an object "
                                "with 'column', 'operator' (one of ==, !=, >, <, >=, <=), "
                                "and 'value'."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column": {"type": "string"},
                                    "operator": {
                                        "type": "string",
                                        "enum": ["==", "!=", ">", "<", ">=", "<="],
                                    },
                                    "value": {},
                                },
                                "required": ["column", "operator", "value"],
                            },
                        },
                        "columns": {
                            "type": "array",
                            "description": "List of column names to include. Omit to return all columns.",
                            "items": {"type": "string"},
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of rows to return. Defaults to 100.",
                            "default": 100,
                        },
                        "combine_all_datasets": {
                            "type": "boolean",
                            "description": (
                                "Set to true when the user asks to query ALL uploaded datasets "
                                "together (e.g. 'across all datasets', 'all files', 'combined'). "
                                "False by default."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        },
    ),

    "aggregate_data": ToolDefinition(
        callable=aggregate_data,
        description=(
            "Group a dataset by a column and aggregate another column using "
            "sum, mean, count, min, or max."
        ),
        json_schema={
            "type": "function",
            "function": {
                "name": "aggregate_data",
                "description": (
                    "Group a dataset by a column and aggregate another column."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_filename": {
                            "type": "string",
                            "description": "The filename of the dataset to aggregate.",
                        },
                        "group_by": {
                            "type": "string",
                            "description": "The column name to group by.",
                        },
                        "agg_col": {
                            "type": "string",
                            "description": "The column name to aggregate.",
                        },
                        "agg_fn": {
                            "type": "string",
                            "enum": ["sum", "mean", "count", "min", "max"],
                            "description": "The aggregation function to apply.",
                        },
                        "combine_all_datasets": {
                            "type": "boolean",
                            "description": (
                                "Set to true when the user asks to aggregate across ALL uploaded "
                                "datasets (e.g. 'across all datasets', 'all files combined', "
                                "'across all uploaded files'). False by default."
                            ),
                        },
                    },
                    "required": ["group_by", "agg_col", "agg_fn"],
                },
            },
        },
    ),

    "generate_summary": ToolDefinition(
        callable=generate_summary,
        description=(
            "Generate a structured textual summary of a dataset including a full profile "
            "and up to three insight sections: top group, trend, and distribution."
        ),
        json_schema={
            "type": "function",
            "function": {
                "name": "generate_summary",
                "description": (
                    "Generate a TEXTUAL summary and business insights from a dataset. "
                    "Use this tool when the user asks for: business insights, key insights, "
                    "a summary, trends, patterns, or analysis IN TEXT FORM. "
                    "Also use this when the user says 'do not create a chart', "
                    "'no chart', 'in text', 'text only', or 'give me insights'. "
                    "Do NOT use generate_chart when the user asks for text insights. "
                    "Returns plain-text findings: top group, trend direction, and distribution — "
                    "all computed deterministically from the dataset."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_filename": {
                            "type": "string",
                            "description": (
                                "The filename of the dataset to summarise. "
                                "If omitted, the first available dataset is used."
                            ),
                        },
                        "group_by": {
                            "type": "string",
                            "description": (
                                "Optional: categorical column to focus insights on "
                                "(e.g. 'region', 'product'). Infer from the query when possible."
                            ),
                        },
                        "agg_col": {
                            "type": "string",
                            "description": (
                                "Optional: numeric column to focus insights on "
                                "(e.g. 'revenue', 'units_sold'). Infer from the query when possible."
                            ),
                        },
                        "combine_all_datasets": {
                            "type": "boolean",
                            "description": (
                                "Set to true when the user asks for a summary across ALL uploaded "
                                "datasets (e.g. 'across all datasets', 'all files', 'combined'). "
                                "False by default."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        },
    ),

    "detect_anomalies": ToolDefinition(
        callable=detect_anomalies_tool,
        description=(
            "Detect statistical anomalies (outliers) in a dataset using the Z-score "
            "or IQR method. Returns a count of flagged rows and a plain-language "
            "explanation for each anomalous value."
        ),
        json_schema={
            "type": "function",
            "function": {
                "name": "detect_anomalies",
                "description": (
                    "Detect statistical anomalies in a dataset using the Z-score "
                    "or IQR method. Returns flagged rows with explanations."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_filename": {
                            "type": "string",
                            "description": (
                                "The filename of the dataset to analyse. "
                                "If omitted, the first available dataset is used."
                            ),
                        },
                        "method": {
                            "type": "string",
                            "enum": ["zscore", "iqr"],
                            "description": (
                                "Detection method. 'zscore' flags rows where the "
                                "absolute Z-score exceeds the threshold (default). "
                                "'iqr' flags rows outside the 1.5×IQR fences."
                            ),
                        },
                        "threshold": {
                            "type": "number",
                            "description": (
                                "Z-score threshold for flagging outliers. "
                                "Only used when method='zscore'. Defaults to 3.0."
                            ),
                        },
                        "combine_all_datasets": {
                            "type": "boolean",
                            "description": (
                                "Set to true when the user asks to detect anomalies across ALL "
                                "uploaded datasets (e.g. 'across all datasets', 'all files'). "
                                "False by default."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        },
    ),

    "generate_chart": ToolDefinition(
        callable=generate_chart,
        description=(
            "Render a visual chart (image) from a dataset. "
            "Only use this when the user explicitly asks for a chart, graph, plot, or visualisation."
        ),
        json_schema={
            "type": "function",
            "function": {
                "name": "generate_chart",
                "description": (
                    "Render a VISUAL CHART (image) from a dataset and return it as a base64-encoded PNG. "
                    "ONLY use this when the user explicitly asks for a chart, graph, bar chart, "
                    "line chart, pie chart, scatter plot, or histogram. "
                    "Do NOT use this for text-based summaries, business insights, or analysis. "
                    "If the user says 'do not create a chart' or 'in text', use generate_summary instead. "
                    "Supported chart types: bar, line, pie, scatter, histogram."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_filename": {
                            "type": "string",
                            "description": (
                                "The filename of the dataset to chart. "
                                "If omitted, the first available dataset is used."
                            ),
                        },
                        "chart_type": {
                            "type": "string",
                            "enum": ["bar", "line", "pie", "scatter", "histogram"],
                            "description": "The type of chart to render.",
                        },
                        "x_col": {
                            "type": "string",
                            "description": "Column name for the x-axis (or labels for pie).",
                        },
                        "y_col": {
                            "type": "string",
                            "description": "Column name for the y-axis (or values for pie/histogram).",
                        },
                        "title": {
                            "type": "string",
                            "description": "Optional chart title.",
                        },
                        "group_by": {
                            "type": "string",
                            "description": (
                                "Optional: categorical column to group and aggregate y_col by SUM "
                                "before rendering. Use this for grouped comparisons such as "
                                "'total revenue per region' or 'units sold by product'. "
                                "Without this, every row becomes a separate data point."
                            ),
                        },
                        "time_period": {
                            "type": "string",
                            "enum": ["M", "W", "Q", "Y"],
                            "description": (
                                "Optional: resample a datetime x_col to this time period before "
                                "plotting. M=monthly, W=weekly, Q=quarterly, Y=yearly. "
                                "Use for trend charts such as 'monthly revenue trend' or "
                                "'quarterly sales'. Produces one data point per period, not one "
                                "per source row. Requires x_col to be a datetime column."
                            ),
                        },
                    },
                    "required": ["chart_type"],
                },
            },
        },
    ),

    "generate_sql": ToolDefinition(
        callable=generate_sql,
        description=(
            "Generate a SQL code snippet for a given operation (aggregate, filter, select) "
            "against a named table. Returns the snippet and language='sql'."
        ),
        json_schema={
            "type": "function",
            "function": {
                "name": "generate_sql",
                "description": (
                    "Generate a SQL code snippet for aggregate, filter, or select operations."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["aggregate", "filter", "select"],
                            "description": "The SQL operation to generate code for.",
                        },
                        "table_name": {
                            "type": "string",
                            "description": (
                                "The SQL table name to use in the generated snippet. "
                                "Defaults to 'data'."
                            ),
                        },
                        "params": {
                            "type": "object",
                            "description": (
                                "Operation-specific parameters such as group_by, agg_col, "
                                "agg_fn, column, operator, value, or columns."
                            ),
                        },
                    },
                    "required": ["operation"],
                },
            },
        },
    ),

    "generate_pandas": ToolDefinition(
        callable=generate_pandas,
        description=(
            "Generate a Pandas code snippet for a given operation (aggregate, filter, select). "
            "Returns the snippet and language='python'."
        ),
        json_schema={
            "type": "function",
            "function": {
                "name": "generate_pandas",
                "description": (
                    "Generate a Pandas code snippet for aggregate, filter, or select operations."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["aggregate", "filter", "select"],
                            "description": "The Pandas operation to generate code for.",
                        },
                        "params": {
                            "type": "object",
                            "description": (
                                "Operation-specific parameters such as group_by, agg_col, "
                                "agg_fn, column, operator, value, or columns."
                            ),
                        },
                    },
                    "required": ["operation"],
                },
            },
        },
    ),

    "percentage_of_total": ToolDefinition(
        callable=percentage_of_total,
        description=(
            "Calculate each group's percentage share of the total for a numeric column. "
            "Use this when the user asks 'what percentage', 'what fraction', or 'what share' "
            "of a total came from a specific group (e.g. region, product). "
            "Also use this for follow-up questions referencing a previous aggregation result, "
            "such as 'what percentage came from that region?'."
        ),
        json_schema={
            "type": "function",
            "function": {
                "name": "percentage_of_total",
                "description": (
                    "Calculate each group's percentage share of the total for a numeric column. "
                    "Use when the user asks about percentage, fraction, or share of total. "
                    "Provide group_value to highlight a specific group (e.g. 'North')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_filename": {
                            "type": "string",
                            "description": (
                                "The filename of the dataset. "
                                "If omitted, the first available dataset is used."
                            ),
                        },
                        "group_by": {
                            "type": "string",
                            "description": (
                                "The categorical column to group by "
                                "(e.g. 'region', 'product')."
                            ),
                        },
                        "agg_col": {
                            "type": "string",
                            "description": (
                                "The numeric column whose sum is used to compute percentages "
                                "(e.g. 'revenue', 'units_sold')."
                            ),
                        },
                        "group_value": {
                            "type": "string",
                            "description": (
                                "Optional: a specific group value to highlight "
                                "(e.g. 'North'). Infer from conversation history when "
                                "the user says 'that region' or 'that product'."
                            ),
                        },
                    },
                    "required": ["group_by", "agg_col"],
                },
            },
        },
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return all tool JSON schemas in OpenAI format for passing to `call_llm`.

    Returns
    -------
    list of OpenAI-compatible tool schema dicts, one per registered tool.
    """
    return [tool_def.json_schema for tool_def in TOOLS.values()]


def _resolve_dataframe(arguments: dict[str, Any], session: Session):  # type: ignore[return]
    """Return the DataFrame for the requested dataset from the session.

    When ``combine_all_datasets=True`` is in *arguments*: concatenates all
    DataFrames in the session that share compatible column sets (at minimum
    they must have overlapping columns).  Returns the combined DataFrame.
    When columns are incompatible (no shared columns), raises ``ValueError``
    with a descriptive message so the caller can surface a clear error.

    When ``combine_all_datasets`` is False/absent: looks up
    ``arguments["dataset_filename"]`` first; falls back to the first
    available DataFrame when the key is absent or maps to an unknown filename.
    Original session DataFrames are never mutated.
    """
    import pandas as pd

    combine = arguments.get("combine_all_datasets", False)

    if combine and len(session.dataframes) > 1:
        dfs = list(session.dataframes.values())

        # Find the intersection of columns across all DataFrames.
        shared_cols = set(dfs[0].columns)
        for df in dfs[1:]:
            shared_cols &= set(df.columns)

        if not shared_cols:
            names = list(session.dataframes.keys())
            raise ValueError(
                f"Cannot combine datasets: no shared columns found. "
                f"Uploaded files: {', '.join(names)}. "
                "Please ensure all datasets have at least one column in common."
            )

        # Use only the shared columns to guarantee compatible dtypes/shapes.
        shared_sorted = sorted(shared_cols)
        slices = [df[shared_sorted].copy() for df in dfs]
        return pd.concat(slices, ignore_index=True)

    elif combine and len(session.dataframes) == 1:
        # Only one dataset — combine_all_datasets is a no-op.
        return next(iter(session.dataframes.values()))

    # --- Default single-dataset path (unchanged) ---
    filename: str | None = arguments.get("dataset_filename")
    if filename and filename in session.dataframes:
        return session.dataframes[filename]

    # Fall back to first available DataFrame
    if session.dataframes:
        return next(iter(session.dataframes.values()))

    return None


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def dispatch(
    tool_name: str,
    arguments: dict[str, Any],
    session: Session,
) -> dict[str, Any]:
    """Dispatch a tool call by name, validate it, resolve the DataFrame, and invoke.

    Parameters
    ----------
    tool_name:
        Name of the tool to call.  Must be a key in ``TOOLS``.
    arguments:
        Raw argument dict from the LLM tool-call payload.
    session:
        The active ``Session``; provides access to loaded DataFrames.

    Returns
    -------
    The dict result produced by the tool function.

    Raises
    ------
    UnknownToolError
        When ``tool_name`` is not registered in ``TOOLS``.
    """
    if tool_name not in TOOLS:
        raise UnknownToolError(
            f"Unknown tool '{tool_name}'. "
            f"Registered tools: {sorted(TOOLS.keys())}"
        )

    tool_def = TOOLS[tool_name]

    # Resolve the DataFrame from the session.
    # _resolve_dataframe may raise ValueError when combine_all_datasets=True
    # and the session's datasets have no shared columns.  Convert to an error
    # dict so the analyst can surface a clean message to the user.
    try:
        df = _resolve_dataframe(arguments, session)
    except ValueError as exc:
        return {"error": str(exc)}

    # Dispatch to each tool with the correct positional/keyword arguments.
    if tool_name == "dataset_profile":
        return tool_def.callable(df)

    if tool_name == "generate_summary":
        return tool_def.callable(
            df,
            group_by=arguments.get("group_by"),
            agg_col=arguments.get("agg_col"),
        )

    if tool_name == "query_data":
        return tool_def.callable(
            df,
            filters=arguments.get("filters"),
            columns=arguments.get("columns"),
            limit=arguments.get("limit", 100),
        )

    if tool_name == "aggregate_data":
        return tool_def.callable(
            df,
            group_by=arguments["group_by"],
            agg_col=arguments["agg_col"],
            agg_fn=arguments["agg_fn"],
        )

    if tool_name == "generate_chart":
        return tool_def.callable(
            df,
            chart_type=arguments["chart_type"],
            x_col=arguments.get("x_col"),
            y_col=arguments.get("y_col"),
            title=arguments.get("title"),
            group_by=arguments.get("group_by"),
            time_period=arguments.get("time_period"),
        )

    if tool_name == "detect_anomalies":
        return tool_def.callable(
            df,
            method=arguments.get("method", "zscore"),
            threshold=float(arguments.get("threshold", 3.0)),
        )

    if tool_name == "generate_sql":
        return tool_def.callable(
            df,
            operation=arguments["operation"],
            table_name=arguments.get("table_name", "data"),
            params=arguments.get("params", {}),
        )

    if tool_name == "generate_pandas":
        return tool_def.callable(
            df,
            operation=arguments["operation"],
            params=arguments.get("params", {}),
        )

    if tool_name == "percentage_of_total":
        return tool_def.callable(
            df,
            group_by=arguments["group_by"],
            agg_col=arguments["agg_col"],
            group_value=arguments.get("group_value"),
        )

    # Fallback — should never be reached for tools listed above
    return tool_def.callable(df, **{
        k: v for k, v in arguments.items() if k != "dataset_filename"
    })

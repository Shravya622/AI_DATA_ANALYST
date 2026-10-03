"""
Analyst orchestrator — the core of the AI-powered Data Analyst.

Public interface
----------------
async def analyse(session_id, query, dataset_filename) -> QueryResponse
    Orchestrates the full query pipeline:
    1. Fetch session from the store
    2. Build schema context (schema summary only — no raw DataFrame rows)
    3. Prepend schema context as system message
    4. Append conversation history and current user query
    5. Call the LLM with all registered tool schemas
    6. Dispatch each returned tool_call via the Tool Registry
    7. Assemble a QueryResponse with answer, optional extras, and ReasoningTrace
    8. Append the exchange to session history

Security note
-------------
Raw DataFrame rows are NEVER passed to the LLM — only the deterministic
schema summary produced by `build_schema_context` is included in the prompt.
"""

from __future__ import annotations

import logging
import re

from app.context_builder import build_schema_context
from app.exceptions import SessionNotFoundError, UnknownToolError
from app.llm_client import LLMResponse, ToolCall, call_llm
from app.models import Exchange, QueryResponse, ReasoningTrace, Session
from app.session_store import session_store
from app.tools.registry import dispatch, get_tool_schemas

logger = logging.getLogger(__name__)

# Phrases that signal the LLM is asking for clarification rather than being
# genuinely unable to answer.  Kept as multi-word phrases only to avoid
# false positives — e.g. "which" alone matches factual answers like
# "North, which had the highest revenue."
_AMBIGUITY_SIGNALS: tuple[str, ...] = (
    "could you clarify",
    "can you clarify",
    "what do you mean",
    "unclear",
    "please specify",
    "could you provide more detail",
)


async def analyse(
    session_id: str,
    query: str,
    dataset_filename: str | None = None,
) -> QueryResponse:
    """Orchestrate a natural-language query against the session's datasets.

    Parameters
    ----------
    session_id:
        UUID string identifying the active session.
    query:
        The user's natural-language question.
    dataset_filename:
        Optional hint for which dataset to operate on. Passed through to each
        tool call so the Tool Registry can resolve the correct DataFrame.

    Returns
    -------
    QueryResponse
        Always contains a non-empty ``answer`` and a populated
        ``reasoning_trace``.

    Raises
    ------
    SessionNotFoundError
        If *session_id* does not exist in the store.
    """
    # ------------------------------------------------------------------
    # 1. Fetch session
    # ------------------------------------------------------------------
    session = session_store.get_session(session_id)

    # ------------------------------------------------------------------
    # Step A — Missing column detection (before LLM call)
    # ------------------------------------------------------------------
    missing_col_msg = _check_missing_columns(query, session)
    if missing_col_msg:
        error_trace = ReasoningTrace(
            tools_selected=[],
            columns_used=[],
            computation_description="Query referenced a non-existent column.",
        )
        _append_exchange(session_id, query, missing_col_msg)
        return QueryResponse(
            answer=missing_col_msg,
            reasoning_trace=error_trace,
        )

    # ------------------------------------------------------------------
    # 2. Build schema context (schema text only — never raw rows)
    # ------------------------------------------------------------------
    schema_context = build_schema_context(session)

    # ------------------------------------------------------------------
    # 3 & 4. Assemble message list
    #   [system: schema] + [history] + [user: query]
    # ------------------------------------------------------------------
    messages: list[dict] = (
        [{"role": "system", "content": schema_context}]
        + session_store.build_message_history(session_id)
        + [{"role": "user", "content": query}]
    )

    # ------------------------------------------------------------------
    # 5. Call the LLM with all registered tool schemas
    # ------------------------------------------------------------------
    tool_schemas = get_tool_schemas()
    llm_response: LLMResponse = call_llm(messages, tools=tool_schemas)

    # ------------------------------------------------------------------
    # 6. Handle LLM error — return immediately with an error QueryResponse
    # ------------------------------------------------------------------
    if llm_response.error:
        logger.error("LLM returned an error: %s", llm_response.error)
        error_trace = ReasoningTrace(
            tools_selected=[],
            columns_used=[],
            computation_description=(
                f"LLM call failed: {llm_response.error}"
            ),
        )
        _append_exchange(session_id, query, llm_response.error)
        return QueryResponse(
            answer=f"I was unable to process your query: {llm_response.error}",
            reasoning_trace=error_trace,
        )

    # ------------------------------------------------------------------
    # 7. Dispatch each tool_call, collect results
    # ------------------------------------------------------------------
    tool_calls: list[ToolCall] = llm_response.tool_calls or []
    tool_results: list[dict] = []
    tools_selected: list[str] = []

    for tc in tool_calls:
        tools_selected.append(tc.name)
        try:
            result = dispatch(
                tc.name,
                {**tc.arguments, "dataset_filename": dataset_filename},
                session,
            )
            tool_results.append({"tool": tc.name, "result": result})
        except UnknownToolError as exc:
            logger.warning("Unknown tool '%s' returned by LLM: %s", tc.name, exc)
            tool_results.append({"tool": tc.name, "result": {"error": str(exc)}})
        except Exception as exc:  # noqa: BLE001
            logger.error("Tool '%s' raised an exception: %s", tc.name, exc)
            tool_results.append({"tool": tc.name, "result": {"error": str(exc)}})

    # ------------------------------------------------------------------
    # 8. Assemble answer
    # ------------------------------------------------------------------
    answer: str
    if tool_results:
        answer = _format_tool_results(tool_results, llm_response)
    else:
        # No tool was invoked by the LLM.
        # Before declaring the query unresolvable, attempt a deterministic
        # fallback for explicit business-insight / summary requests.
        if _is_summary_query(query) and session.dataframes:
            try:
                group_by_hint, agg_col_hint = _extract_column_hints(query, session)
                summary_result = dispatch(
                    "generate_summary",
                    {
                        "dataset_filename": dataset_filename,
                        "group_by": group_by_hint,
                        "agg_col": agg_col_hint,
                    },
                    session,
                )
                tool_results = [{"tool": "generate_summary", "result": summary_result}]
                tools_selected = ["generate_summary"]
                answer = _format_tool_results(tool_results, llm_response)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Summary fallback failed: %s", exc)
                answer = (
                    "I was unable to find an answer to your query. "
                    "Please rephrase or provide more context."
                )
        else:
            # Classify as ambiguous or unresolvable.
            llm_content = llm_response.content or ""
            if llm_content and _is_ambiguous_response(llm_content):
                # LLM is genuinely asking for clarification — pass through.
                answer = llm_content
            elif llm_content:
                # LLM returned a factual/conversational response without using
                # a tool.  Preserve the content rather than discarding it with
                # a failure message.
                answer = llm_content
            else:
                # No content and no tool — truly unresolvable.
                answer = (
                    "I was unable to find an answer to your query. "
                    "Please rephrase or provide more context."
                )

    # ------------------------------------------------------------------
    # 9. Build ReasoningTrace
    # ------------------------------------------------------------------
    if tool_results:
        computation_description = (
            f"Used tool(s): {', '.join(tools_selected)}."
        )
    elif not tool_results and llm_response.content and _is_ambiguous_response(llm_response.content):
        computation_description = "Query was ambiguous; clarification requested."
    else:
        computation_description = "Query could not be resolved."

    reasoning_trace = ReasoningTrace(
        tools_selected=tools_selected,
        columns_used=[],
        computation_description=computation_description,
    )

    # ------------------------------------------------------------------
    # 10. Extract optional extras from tool results
    # ------------------------------------------------------------------
    chart_base64: str | None = None
    code_snippet: str | None = None
    code_language: str | None = None
    sampled: bool = False
    anomaly_report = None

    for tr in tool_results:
        tool_n = tr.get("tool", "")
        result = tr.get("result", {})
        if isinstance(result, dict):
            # Chart
            if result.get("base64_png"):
                chart_base64 = result["base64_png"]
                sampled = bool(result.get("sampled", False))
            # Code snippets
            if result.get("snippet") is not None:
                code_snippet = result["snippet"]
                code_language = result.get("language")
            # Anomaly report
            if tool_n == "detect_anomalies" and "flagged_rows" in result:
                from app.models import AnomalyRecord, AnomalyReport
                flagged = [AnomalyRecord(**row) for row in result.get("flagged_rows", [])]
                anomaly_report = AnomalyReport(
                    count=result.get("count", len(flagged)),
                    flagged_rows=flagged,
                )

    # ------------------------------------------------------------------
    # 11. Append exchange to session history
    # ------------------------------------------------------------------
    # Store the tool context alongside the answer so that the LLM has
    # explicit context on subsequent turns (e.g. "that region" references).
    # The user-facing API response is always the plain `answer`; only the
    # string saved to history is enriched.
    if tools_selected:
        tool_prefix = f"[Used tool(s): {', '.join(tools_selected)}]\n"
        history_answer = tool_prefix + answer
    else:
        history_answer = answer

    _append_exchange(session_id, query, history_answer)

    # ------------------------------------------------------------------
    # 12. Return QueryResponse
    # ------------------------------------------------------------------
    return QueryResponse(
        answer=answer,
        chart_base64=chart_base64,
        code_snippet=code_snippet,
        code_language=code_language,
        anomaly_report=anomaly_report,
        reasoning_trace=reasoning_trace,
        sampled=sampled,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _format_tool_results(
    tool_results: list[dict],
    llm_response: LLMResponse,
) -> str:
    """Format tool results into a human-readable answer string.

    Combines any LLM conversational content (if present) with structured
    summaries of each tool's output.

    Parameters
    ----------
    tool_results:
        List of ``{"tool": name, "result": dict}`` entries.
    llm_response:
        The raw LLM response, used to prepend any conversational text.
    """
    parts: list[str] = []

    # Prepend LLM's own text if it provided any alongside tool calls
    if llm_response.content:
        parts.append(llm_response.content)

    for tr in tool_results:
        tool_name = tr["tool"]
        result = tr.get("result", {})

        if not isinstance(result, dict):
            parts.append(f"[{tool_name}]: {result}")
            continue

        if "error" in result:
            parts.append(f"[{tool_name} error]: {result['error']}")
            continue

        # Tool-specific formatting
        if tool_name == "dataset_profile":
            parts.append(_format_profile_with_columns(result))
        elif tool_name == "generate_summary":
            parts.append(_format_summary(result))
        elif tool_name == "aggregate_data":
            parts.append(_format_aggregation(result))
        elif tool_name == "query_data":
            parts.append(_format_query_data(result))
        elif tool_name == "generate_chart":
            # Chart binary is surfaced via chart_base64; just note it here
            sampled_note = " (data was sampled)" if result.get("sampled") else ""
            parts.append(f"Chart generated successfully{sampled_note}.")
        elif tool_name == "detect_anomalies":
            count = result.get("count", 0)
            parts.append(f"Found {count} anomalous record(s).")
        elif tool_name == "percentage_of_total":
            rows = result.get("results", [])
            group_by = result.get("group_by", "group")
            agg_col = result.get("agg_col", "value")
            highlighted = result.get("highlighted")
            if highlighted:
                gv = highlighted["group_value"]
                pct = highlighted["percentage"]
                parts.append(
                    f"{gv} contributed {pct}% of the total {agg_col}.\n"
                    f"\nAll {group_by} percentages of total {agg_col}:\n"
                    + "\n".join(f"  {r['group_value']}: {r['percentage']}%" for r in rows)
                )
            else:
                lines = [f"Percentage of total {agg_col} by {group_by}:"]
                for r in rows:
                    lines.append(f"  {r['group_value']}: {r['percentage']}%")
                parts.append("\n".join(lines))
        elif tool_name in ("generate_sql", "generate_pandas"):
            snippet = result.get("snippet")
            lang = result.get("language", "code")
            if snippet:
                parts.append(f"Here is the {lang} snippet:\n```{lang}\n{snippet}\n```")
            else:
                parts.append(f"Code generation failed: {result.get('error', 'unknown error')}")
        else:
            # Generic fallback: summarise the keys present
            parts.append(f"[{tool_name}]: {_brief_dict_summary(result)}")

    return "\n\n".join(p for p in parts if p)


def _format_profile(result: dict) -> str:
    """Format a dataset_profile result into readable text."""
    row_count = result.get("row_count", "?")
    col_count = result.get("column_count", "?")
    lines = [f"Dataset has {row_count} rows and {col_count} columns."]

    stats = result.get("stats", {})
    if stats:
        lines.append("Numeric column statistics:")
        for col, s in stats.items():
            if s.get("unavailable"):
                lines.append(f"  {col}: statistics unavailable")
            else:
                lines.append(
                    f"  {col}: mean={s.get('mean')}, "
                    f"min={s.get('min')}, max={s.get('max')}"
                )
    return "\n".join(lines)


def _format_summary(result: dict) -> str:
    """Format a generate_summary result into readable text.

    Handles two modes:
    - Focused mode (group_by + agg_col provided): renders the pre-computed
      insights_text directly, which already contains ranked groups,
      percentages, and comparisons.
    - Generic mode: renders full profile + labelled insight sections.
    All numeric figures are sourced from the Pandas-computed ``result`` dict —
    no values are generated by the LLM.
    """
    # Focused mode: result contains a single "focused_group_analysis" insight
    focused_data = (
        result.get("insights", {}).get("focused_group_analysis")
    )
    if result.get("focused") and focused_data:
        return focused_data.get("insights_text", str(focused_data))

    # Generic mode
    parts: list[str] = []
    profile = result.get("profile")
    if profile:
        parts.append(_format_profile_with_columns(profile))

    insights = result.get("insights", {})
    for key, value in insights.items():
        if isinstance(value, dict) and "text" in value:
            parts.append(f"• {value['text']}")
        else:
            parts.append(f"{key}: {value}")
    return "\n\n".join(parts) if parts else str(result)


def _format_profile_with_columns(profile: dict) -> str:
    """Format a dataset_profile result including dtypes and null counts.

    Extends ``_format_profile`` with a per-column section that lists every
    column's dtype and null count, satisfying the requirement that dtypes and
    null counts appear in summary answers.
    """
    row_count = profile.get("row_count", "?")
    col_count = profile.get("column_count", "?")
    lines = [f"Dataset Overview: {row_count} rows × {col_count} columns"]

    # Per-column dtype and null count
    columns: list[dict] = profile.get("columns", [])
    if columns:
        lines.append("\nColumn details:")
        for col_info in columns:
            name = col_info.get("name", "?")
            dtype = col_info.get("dtype", "?")
            null_count = col_info.get("null_count", 0)
            null_label = (
                f"{null_count} null(s)"
                if null_count > 0
                else "no nulls"
            )
            lines.append(f"  {name}: dtype={dtype}, {null_label}")

    # Numeric descriptive statistics
    stats: dict = profile.get("stats", {})
    if stats:
        lines.append("\nNumeric column statistics:")
        for col, s in stats.items():
            if s.get("unavailable"):
                lines.append(f"  {col}: statistics unavailable")
            else:
                mean = s.get("mean")
                min_ = s.get("min")
                max_ = s.get("max")
                median = s.get("median")
                std = s.get("std")
                lines.append(
                    f"  {col}: mean={mean}, median={median}, "
                    f"min={min_}, max={max_}, std={std}"
                )

    return "\n".join(lines)


def _format_aggregation(result: dict) -> str:
    """Format an aggregate_data result into readable text."""
    rows = result.get("results", [])
    if not rows:
        return "No aggregation results returned."
    lines = []
    for row in rows[:20]:  # cap display at 20 rows
        group = row.get("group_value", "?")
        value = row.get("result", "?")
        lines.append(f"  {group}: {value}")
    suffix = f"\n  ... ({len(rows) - 20} more)" if len(rows) > 20 else ""
    return "Aggregation results:\n" + "\n".join(lines) + suffix


def _format_query_data(result: dict) -> str:
    """Format a query_data result into readable text."""
    rows = result.get("rows", [])
    count = len(rows)
    if count == 0:
        return "No rows matched the query."
    # Show up to 5 rows as a brief preview
    preview = rows[:5]
    lines = [f"Query returned {count} row(s). Preview:"]
    for row in preview:
        lines.append(f"  {row}")
    if count > 5:
        lines.append(f"  ... ({count - 5} more row(s))")
    return "\n".join(lines)


def _brief_dict_summary(d: dict) -> str:
    """Return a one-line summary of a dict's top-level keys."""
    keys = list(d.keys())
    if len(keys) <= 5:
        return str(d)
    return "{" + ", ".join(f"{k}: ..." for k in keys[:5]) + ", ...}"


def _check_missing_columns(query: str, session: Session) -> str | None:
    """Detect quoted column references in *query* that don't exist in any dataset.

    Parameters
    ----------
    query:
        The user's natural-language question.
    session:
        The active session; used to collect all known column names.

    Returns
    -------
    str | None
        An error message listing available columns if any quoted name is not
        found, otherwise ``None``.
    """
    # Collect all column names across every dataset in the session.
    all_cols: list[str] = []
    for record in session.datasets.values():
        for col_schema in record.column_schemas:
            all_cols.append(col_schema.name)

    if not all_cols:
        return None  # No datasets loaded — nothing to check against.

    # Extract words wrapped in double or single quotes from the query.
    quoted = re.findall(r'["\']([^"\']+)["\']', query)

    missing = [name for name in quoted if name not in all_cols]
    if missing:
        return (
            f"Column not found. Available columns are: {', '.join(sorted(all_cols))}"
        )
    return None


def _is_ambiguous_response(content: str) -> bool:
    """Return True if *content* contains signals that the LLM is asking for
    clarification (i.e. the query was ambiguous)."""
    lowered = content.lower()
    return any(signal in lowered for signal in _AMBIGUITY_SIGNALS)


# Phrases that indicate the user wants a textual summary or business insights
# (used by the deterministic summary fallback when the LLM returns no tool call).
_SUMMARY_TRIGGERS: tuple[str, ...] = (
    "business insight",
    "key insight",
    "summarize",
    "summarise",
    "summary",
    "key finding",
    "in text",
    "do not create a chart",
    "no chart",
    "text only",
    "give me insight",
    "tell me about",
    "what are the insights",
    "analyse the data",
    "analyze the data",
)

# Phrases that indicate an explicit chart/visualisation request — the fallback
# must NOT fire for these even if "insights" appears elsewhere in the query.
# Note: We only block when the phrase is NOT negated ("do not create a chart"
# should NOT be treated as a chart request).
_CHART_TRIGGERS: tuple[str, ...] = (
    "bar chart",
    "line chart",
    "pie chart",
    "scatter plot",
    "histogram",
    "plot the",
    "visualize",
    "visualise",
    "draw a",
    "show a chart",
    "show me a chart",
    "generate a chart",
)

# Negation prefixes that cancel a chart trigger — if any of these appear
# the query is treated as a summary request, not a chart request.
_CHART_NEGATIONS: tuple[str, ...] = (
    "do not",
    "don't",
    "no chart",
    "not a chart",
    "without a chart",
    "instead of a chart",
)


def _is_summary_query(query: str) -> bool:
    """Return True when the query matches a textual-insight/summary pattern
    and is NOT an explicit chart request.

    This is used as a deterministic fallback when the LLM returns no tool call
    for queries that clearly ask for textual business insights or summaries.
    """
    lowered = query.lower()

    # Explicit chart requests must never trigger the summary fallback —
    # but only if the chart phrase is not negated ("do not create a chart"
    # is a summary request, not a chart request).
    negated = any(neg in lowered for neg in _CHART_NEGATIONS)
    if not negated and any(trigger in lowered for trigger in _CHART_TRIGGERS):
        return False

    return any(trigger in lowered for trigger in _SUMMARY_TRIGGERS)


def _extract_column_hints(query: str, session: Session) -> tuple[str | None, str | None]:
    """Extract group_by and agg_col hints from the query by matching against
    known column names in the session.

    Returns (group_by, agg_col) where either may be None.
    The first categorical column match becomes group_by, the first numeric
    column match becomes agg_col.  This lets "revenue by region" map to
    agg_col='revenue', group_by='region' without hardcoding any names.
    """
    from app.type_inferrer import infer_schema  # avoid circular at module level

    lowered = query.lower()
    group_by: str | None = None
    agg_col: str | None = None

    for record in session.datasets.values():
        for col_schema in record.column_schemas:
            name = col_schema.name
            if name.lower() in lowered:
                if col_schema.dtype == "numeric" and agg_col is None:
                    agg_col = name
                elif col_schema.dtype in ("categorical", "datetime") and group_by is None:
                    group_by = name

    return group_by, agg_col


def _append_exchange(session_id: str, query: str, answer: str) -> None:
    """Append user + assistant exchanges to session history."""
    session_store.append_history(session_id, Exchange(role="user", content=query))
    session_store.append_history(session_id, Exchange(role="assistant", content=answer))

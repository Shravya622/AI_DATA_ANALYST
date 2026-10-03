"""
Property-based tests for the AI-powered Data Analyst.

Feature: ai-data-analyst

Each test class covers one design-document property, tagged with:
    # Feature: ai-data-analyst, Property N: <text>

All tests use @settings(max_examples=100, deadline=None) and Hypothesis strategies.
DataFrames are kept small (1-5 rows, 1-3 columns) for speed.

Validates: Requirements 11.3, 11.4
"""

from __future__ import annotations

import io
import os

# Set env var before any app import so pydantic-settings is satisfied.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")

import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
import hypothesis.extra.pandas as hpd

from app.validator import validate_csv
from app.type_inferrer import infer_schema
from app.tools.aggregate_data import aggregate_data, SUPPORTED_AGG_FNS
from app.models import ColumnSchema, DatasetRecord
from app.session_store import SessionStore
from app.tools.registry import TOOLS, dispatch
from app.exceptions import UnknownToolError
from app.models import Exchange, Session


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to CSV bytes (in-memory)."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _is_parseable_csv(b: bytes) -> bool:
    """Return True if pandas can parse b as a CSV with ≥1 col and ≥1 row."""
    try:
        df = pd.read_csv(io.BytesIO(b))
        if df.shape[1] == 0 or len(df) == 0:
            return False
        # Exclude DataFrames that pandas silently renamed (unnamed/duplicates)
        cols = list(df.columns)
        if any(str(c).strip() == "" or str(c).startswith("Unnamed:") for c in cols):
            return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# hypothesis.extra.pandas in 6.x: `rows` parameter must be omitted or an
# integer strategy that produces ROW DATA (tuples), not a row count.
# To control the number of rows, use `min_size` / `max_size` instead.

_FLOAT_ELEMENTS = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)

def _small_float_df(col_names: list[str]) -> st.SearchStrategy:
    """Return a strategy producing small DataFrames with float columns."""
    return hpd.data_frames(
        columns=[hpd.column(name, dtype=float, elements=_FLOAT_ELEMENTS) for name in col_names],
        index=hpd.range_indexes(min_size=1, max_size=5),
    )


def _small_str_df(col_names: list[str]) -> st.SearchStrategy:
    """Return a strategy producing small DataFrames with object/str columns."""
    return hpd.data_frames(
        columns=[
            hpd.column(name, dtype=str, elements=st.text(min_size=0, max_size=10))
            for name in col_names
        ],
        index=hpd.range_indexes(min_size=1, max_size=5),
    )


# ===========================================================================
# Property 1 — CSV validation correctness
# ===========================================================================
# Feature: ai-data-analyst, Property 1: Valid DataFrames converted to CSV bytes
# return ok=True; random binary bytes (non-parseable) return ok=False with
# non-empty error.

class TestProperty1CSVValidationCorrectness:
    """
    Property 1: CSV validation correctness.

    Validates: Requirements 1.1, 1.2
    """

    @given(_small_float_df(["a", "b", "c"]))
    @settings(max_examples=100, deadline=None)
    def test_valid_dataframe_csv_returns_ok_true(self, df: pd.DataFrame) -> None:
        # Feature: ai-data-analyst, Property 1: valid DataFrames -> ok=True
        assume(len(df) >= 1)  # Hypothesis may still generate empty DFs; skip them
        csv_bytes = _df_to_csv_bytes(df)
        result = validate_csv(csv_bytes, "test.csv")
        assert result.ok is True, (
            f"Expected ok=True for valid CSV, got error: {result.error}"
        )
        assert result.dataframe is not None
        assert result.error is None

    @given(
        st.binary(min_size=1).filter(lambda b: not _is_parseable_csv(b))
    )
    @settings(max_examples=100, deadline=None)
    def test_random_binary_returns_ok_false_with_error(self, data: bytes) -> None:
        # Feature: ai-data-analyst, Property 1: non-parseable bytes -> ok=False
        result = validate_csv(data, "junk.bin")
        assert result.ok is False, "Expected ok=False for non-parseable binary"
        assert result.error is not None
        assert len(result.error) > 0, "Error message must be non-empty"


# ===========================================================================
# Property 2 — Column header validation
# ===========================================================================
# Feature: ai-data-analyst, Property 2: CSVs with empty or duplicate headers
# return ok=False with non-empty error.

class TestProperty2ColumnHeaderValidation:
    """
    Property 2: Column header validation.

    Validates: Requirement 1.5
    """

    @given(
        st.integers(min_value=1, max_value=5),  # number of data rows
        st.integers(min_value=2, max_value=4),  # total number of columns (≥2 so there's a non-empty col)
    )
    @settings(max_examples=100, deadline=None)
    def test_empty_header_returns_ok_false(self, num_rows: int, num_cols: int) -> None:
        # Feature: ai-data-analyst, Property 2: empty header -> ok=False
        # Build headers: first one is empty, rest are "col_1", "col_2", ...
        col_names = [""] + [f"col_{i}" for i in range(1, num_cols)]
        rows = [",".join(col_names)]
        for row_idx in range(num_rows):
            rows.append(",".join(str(row_idx) for _ in col_names))
        csv_bytes = "\n".join(rows).encode("utf-8")

        result = validate_csv(csv_bytes, "empty_header.csv")
        assert result.ok is False, (
            f"Expected ok=False for CSV with empty header. "
            f"Headers: {col_names}, csv: {csv_bytes[:200]}"
        )
        assert result.error is not None
        assert len(result.error) > 0

    @given(
        st.integers(min_value=1, max_value=5),  # number of data rows
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu")),
            min_size=1,
            max_size=8,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_duplicate_header_returns_ok_false(self, num_rows: int, col_name: str) -> None:
        # Feature: ai-data-analyst, Property 2: duplicate headers -> ok=False
        # Build CSV bytes with a duplicated column name plus a distinct third column.
        dup_cols = [col_name, col_name, "other_distinct_col"]
        rows = [",".join(dup_cols)]
        for row_idx in range(num_rows):
            rows.append(",".join(str(row_idx) for _ in dup_cols))
        csv_bytes = "\n".join(rows).encode("utf-8")

        result = validate_csv(csv_bytes, "dup_header.csv")
        assert result.ok is False, (
            f"Expected ok=False for CSV with duplicate header '{col_name}'"
        )
        assert result.error is not None
        assert len(result.error) > 0


# ===========================================================================
# Property 3 — Column schema coverage
# ===========================================================================
# Feature: ai-data-analyst, Property 3: infer_schema(df) returns exactly N
# ColumnSchema entries for an N-column DataFrame; each dtype is valid.

VALID_DTYPES = frozenset({"numeric", "datetime", "categorical"})


class TestProperty3ColumnSchemaCoverage:
    """
    Property 3: Column type inference covers all columns.

    Validates: Requirement 1.6
    """

    @given(_small_float_df(["x", "y", "z"]))
    @settings(max_examples=100, deadline=None)
    def test_schema_count_matches_column_count(self, df: pd.DataFrame) -> None:
        # Feature: ai-data-analyst, Property 3: schema length == N columns
        assume(len(df) >= 1)
        schemas = infer_schema(df)
        assert len(schemas) == len(df.columns), (
            f"Expected {len(df.columns)} schema entries, got {len(schemas)}"
        )

    @given(_small_float_df(["p", "q", "r"]))
    @settings(max_examples=100, deadline=None)
    def test_each_dtype_is_valid_for_float_columns(self, df: pd.DataFrame) -> None:
        # Feature: ai-data-analyst, Property 3: each dtype in valid set
        assume(len(df) >= 1)
        schemas = infer_schema(df)
        for schema in schemas:
            assert schema.dtype in VALID_DTYPES, (
                f"Column '{schema.name}' has invalid dtype '{schema.dtype}'. "
                f"Must be one of {VALID_DTYPES}"
            )

    @given(_small_str_df(["m", "n"]))
    @settings(max_examples=100, deadline=None)
    def test_schema_dtypes_valid_for_string_columns(self, df: pd.DataFrame) -> None:
        # Feature: ai-data-analyst, Property 3: string columns get valid dtype
        assume(len(df) >= 1)
        schemas = infer_schema(df)
        assert len(schemas) == len(df.columns)
        for schema in schemas:
            assert schema.dtype in VALID_DTYPES


# ===========================================================================
# Property 6 — Aggregation idempotence
# ===========================================================================
# Feature: ai-data-analyst, Property 6: calling aggregate_data twice on the
# same unchanged DataFrame with the same parameters returns identical results.

class TestProperty6AggregationIdempotence:
    """
    Property 6: Aggregation idempotence.

    Validates: Requirements 11.4, 3.2, 3.3
    """

    @given(
        st.integers(min_value=1, max_value=5),          # num rows
        st.sampled_from(["A", "B", "C"]),                # group values
        st.sampled_from(sorted(SUPPORTED_AGG_FNS)),      # agg function
    )
    @settings(max_examples=100, deadline=None)
    def test_aggregate_data_idempotent_simple(
        self, num_rows: int, group_val: str, agg_fn: str
    ) -> None:
        # Feature: ai-data-analyst, Property 6: aggregate_data is idempotent
        df = pd.DataFrame({
            "group": [group_val] * num_rows,
            "value": list(range(1, num_rows + 1)),
        })

        result1 = aggregate_data(df, group_by="group", agg_col="value", agg_fn=agg_fn)
        result2 = aggregate_data(df, group_by="group", agg_col="value", agg_fn=agg_fn)

        assert result1 == result2, (
            f"aggregate_data not idempotent for agg_fn='{agg_fn}': "
            f"first={result1}, second={result2}"
        )

    @given(
        hpd.data_frames(
            columns=[
                hpd.column(
                    "cat",
                    dtype=str,
                    elements=st.sampled_from(["X", "Y", "Z"]),
                ),
                hpd.column(
                    "num",
                    dtype=float,
                    elements=st.floats(
                        min_value=0.0,
                        max_value=1000.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                ),
            ],
            index=hpd.range_indexes(min_size=1, max_size=5),
        ),
        st.sampled_from(sorted(SUPPORTED_AGG_FNS)),
    )
    @settings(max_examples=100, deadline=None)
    def test_aggregate_data_idempotent_with_generated_df(
        self, df: pd.DataFrame, agg_fn: str
    ) -> None:
        # Feature: ai-data-analyst, Property 6: idempotent on generated DataFrames
        assume(len(df) >= 1)
        result1 = aggregate_data(df, group_by="cat", agg_col="num", agg_fn=agg_fn)
        result2 = aggregate_data(df, group_by="cat", agg_col="num", agg_fn=agg_fn)

        assert result1 == result2, (
            f"aggregate_data not idempotent for agg_fn='{agg_fn}'"
        )


# ===========================================================================
# Property 7 — DatasetRecord round-trip
# ===========================================================================
# Feature: ai-data-analyst, Property 7: serialising and deserialising
# DatasetRecord via model_dump_json()/model_validate_json() produces
# field-for-field equivalent records.

class TestProperty7DatasetRecordRoundTrip:
    """
    Property 7: DatasetRecord serialisation round-trip.

    Validates: Requirements 11.3
    """

    @given(
        st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(
                whitelist_categories=("Ll", "Lu", "Nd"),
                whitelist_characters="._-",
            ),
        ),
        st.integers(min_value=0, max_value=10_000),
        st.lists(
            st.builds(
                ColumnSchema,
                name=st.text(
                    min_size=1,
                    max_size=20,
                    alphabet=st.characters(
                        whitelist_categories=("Ll", "Lu", "Nd"),
                        whitelist_characters="_",
                    ),
                ),
                dtype=st.sampled_from(["numeric", "datetime", "categorical"]),
                null_count=st.integers(min_value=0, max_value=100),
            ),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_dataset_record_roundtrip(
        self,
        filename: str,
        row_count: int,
        column_schemas: list[ColumnSchema],
    ) -> None:
        # Feature: ai-data-analyst, Property 7: DatasetRecord round-trip via JSON
        original = DatasetRecord(
            filename=filename,
            row_count=row_count,
            column_schemas=column_schemas,
        )

        json_str = original.model_dump_json()
        recovered = DatasetRecord.model_validate_json(json_str)

        assert recovered.filename == original.filename
        assert recovered.row_count == original.row_count
        assert len(recovered.column_schemas) == len(original.column_schemas)

        for orig_col, rec_col in zip(original.column_schemas, recovered.column_schemas):
            assert rec_col.name == orig_col.name
            assert rec_col.dtype == orig_col.dtype
            assert rec_col.null_count == orig_col.null_count


# ===========================================================================
# Property 13 — Conversation history cap
# ===========================================================================
# Feature: ai-data-analyst, Property 13: after appending >20 exchanges,
# get_history returns exactly 20, the most recent ones.

class TestProperty13ConversationHistoryCap:
    """
    Property 13: Conversation history cap preserves most-recent exchanges.

    Validates: Requirement 8.3
    """

    @given(st.integers(min_value=21, max_value=100))
    @settings(max_examples=100, deadline=None)
    def test_history_capped_at_20(self, num_exchanges: int) -> None:
        # Feature: ai-data-analyst, Property 13: history capped at 20
        store = SessionStore(max_history=20)
        session_id = store.create_session()

        for i in range(num_exchanges):
            exchange = Exchange(role="user", content=f"message {i}")
            store.append_history(session_id, exchange)

        history = store.get_history(session_id)
        assert len(history) == 20, (
            f"Expected exactly 20 exchanges after {num_exchanges} appended, "
            f"got {len(history)}"
        )

    @given(st.integers(min_value=21, max_value=100))
    @settings(max_examples=100, deadline=None)
    def test_history_preserves_most_recent(self, num_exchanges: int) -> None:
        # Feature: ai-data-analyst, Property 13: most recent 20 are retained
        store = SessionStore(max_history=20)
        session_id = store.create_session()

        for i in range(num_exchanges):
            exchange = Exchange(role="user", content=f"message {i}")
            store.append_history(session_id, exchange)

        history = store.get_history(session_id)

        # The last 20 messages should have content "message X" where X
        # ranges from (num_exchanges - 20) to (num_exchanges - 1).
        expected_start = num_exchanges - 20
        for idx, ex in enumerate(history):
            expected_content = f"message {expected_start + idx}"
            assert ex.content == expected_content, (
                f"History[{idx}] should be '{expected_content}', got '{ex.content}'"
            )


# ===========================================================================
# Property 14 — Tool registry rejects unknown tool names
# ===========================================================================
# Feature: ai-data-analyst, Property 14: any string not in TOOLS.keys()
# causes dispatch to raise UnknownToolError.

class TestProperty14ToolRegistryRejectsUnknownNames:
    """
    Property 14: Tool registry rejects unknown tool names.

    Validates: Requirement 9.2
    """

    @given(
        st.text(min_size=1, max_size=50).filter(lambda s: s not in TOOLS)
    )
    @settings(max_examples=100, deadline=None)
    def test_unknown_tool_raises_unknown_tool_error(self, tool_name: str) -> None:
        # Feature: ai-data-analyst, Property 14: unknown tool -> UnknownToolError
        # Minimal session (dispatch raises before resolving DataFrames)
        session = Session(
            session_id="test-session",
            datasets={},
            dataframes={},
            history=[],
        )

        with pytest.raises(UnknownToolError):
            dispatch(tool_name, {}, session)

    @given(st.just(""))
    @settings(max_examples=10)
    def test_empty_string_raises_unknown_tool_error(self, tool_name: str) -> None:
        # Feature: ai-data-analyst, Property 14: empty string -> UnknownToolError
        session = Session(
            session_id="test-session",
            datasets={},
            dataframes={},
            history=[],
        )

        with pytest.raises(UnknownToolError):
            dispatch(tool_name, {}, session)


# ===========================================================================
# Property 15 — API rejects path traversal inputs
# ===========================================================================
# Feature: ai-data-analyst, Property 15: strings containing ../, ..\, or
# \x00 return 422 from the API.
#
# Note: We test via query parameters only. Embedding traversal sequences
# directly inside URL path segments is normalised by httpx before they reach
# the server (e.g. /sessions/../ collapses), and null bytes are rejected by
# httpx as invalid URL characters. Query parameter values are not normalised,
# making them the correct vector for this property test.

class TestProperty15APIRejectsPathTraversalInputs:
    """
    Property 15: API rejects inputs containing path traversal sequences.

    Validates: Requirement 10.4
    """

    @staticmethod
    def _get_client():
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app, raise_server_exceptions=False)

    @given(
        st.sampled_from(["../", "..\\"]),   # URL-safe traversal sequences
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
            min_size=0,
            max_size=20,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_query_param_traversal_dot_slash_returns_422(
        self, traversal: str, suffix: str
    ) -> None:
        # Feature: ai-data-analyst, Property 15: ../ or ..\ in query param -> 422
        client = self._get_client()
        param_value = traversal + suffix
        response = client.get("/health", params={"q": param_value})
        assert response.status_code == 422, (
            f"Expected 422 for traversal input '{param_value}', "
            f"got {response.status_code}"
        )

    @given(
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
            min_size=0,
            max_size=20,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_query_param_null_byte_returns_422(self, suffix: str) -> None:
        # Feature: ai-data-analyst, Property 15: null byte in query param -> 422
        # httpx encodes \x00 as %00 in query param values, which reaches the server.
        client = self._get_client()
        param_value = "\x00" + suffix
        # Pass as raw bytes via the query string to bypass httpx URL encoding issues
        encoded_qs = ("q=%00" + suffix).encode("utf-8")
        from starlette.testclient import TestClient as StarletteClient
        from app.main import app as _app
        sc = StarletteClient(_app, raise_server_exceptions=False)
        raw_response = sc.get("/health?" + encoded_qs.decode("ascii", errors="replace"))
        assert raw_response.status_code == 422, (
            f"Expected 422 for null-byte query param, got {raw_response.status_code}"
        )

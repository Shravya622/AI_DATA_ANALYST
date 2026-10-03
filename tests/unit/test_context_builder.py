"""
Unit tests for app.context_builder.build_schema_context.

Acceptance criteria verified
-----------------------------
AC1  Output lists every dataset by filename with all column dtypes / null counts.
AC2  Output is deterministic (same session → same string).
AC3  Empty session returns the canonical "no datasets" message.
"""

import pytest

from app.context_builder import build_schema_context
from app.models import ColumnSchema, DatasetRecord, Session

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NO_DATASETS_MSG = "No datasets are currently loaded in this session."


def _make_session(datasets: dict[str, DatasetRecord] | None = None) -> Session:
    """Create a minimal Session with the given dataset records."""
    return Session(
        session_id="test-session-id",
        datasets=datasets or {},
        dataframes={},
        history=[],
    )


def _make_record(
    filename: str,
    row_count: int,
    columns: list[tuple[str, str, int]],
) -> DatasetRecord:
    """Helper to build a DatasetRecord from a list of (name, dtype, null_count) tuples."""
    schemas = [
        ColumnSchema(name=name, dtype=dtype, null_count=null_count)
        for name, dtype, null_count in columns
    ]
    return DatasetRecord(filename=filename, row_count=row_count, column_schemas=schemas)


# ---------------------------------------------------------------------------
# AC3 — empty session
# ---------------------------------------------------------------------------


class TestEmptySession:
    def test_returns_no_datasets_message(self):
        session = _make_session()
        result = build_schema_context(session)
        assert result == NO_DATASETS_MSG

    def test_returns_string_type(self):
        session = _make_session()
        result = build_schema_context(session)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# AC1 — output lists every dataset with all column info
# ---------------------------------------------------------------------------


class TestSingleDataset:
    def test_filename_in_output(self):
        record = _make_record("sales.csv", 100, [("revenue", "numeric", 0)])
        session = _make_session({"sales.csv": record})
        result = build_schema_context(session)
        assert "sales.csv" in result

    def test_row_count_in_output(self):
        record = _make_record("sales.csv", 250, [("revenue", "numeric", 3)])
        session = _make_session({"sales.csv": record})
        result = build_schema_context(session)
        assert "250 rows" in result

    def test_column_count_in_output(self):
        record = _make_record(
            "sales.csv",
            10,
            [("a", "numeric", 0), ("b", "categorical", 1)],
        )
        session = _make_session({"sales.csv": record})
        result = build_schema_context(session)
        assert "2 columns" in result

    def test_column_name_in_output(self):
        record = _make_record("data.csv", 10, [("price", "numeric", 2)])
        session = _make_session({"data.csv": record})
        result = build_schema_context(session)
        assert "price" in result

    def test_dtype_in_output(self):
        record = _make_record("data.csv", 10, [("price", "numeric", 0)])
        session = _make_session({"data.csv": record})
        result = build_schema_context(session)
        assert "numeric" in result

    def test_null_count_in_output(self):
        record = _make_record("data.csv", 10, [("price", "numeric", 7)])
        session = _make_session({"data.csv": record})
        result = build_schema_context(session)
        assert "null count: 7" in result

    def test_header_format(self):
        record = _make_record("report.csv", 50, [("col_a", "categorical", 0)])
        session = _make_session({"report.csv": record})
        result = build_schema_context(session)
        assert "Dataset: report.csv (50 rows, 1 columns)" in result

    def test_column_line_format(self):
        record = _make_record("report.csv", 50, [("date_col", "datetime", 3)])
        session = _make_session({"report.csv": record})
        result = build_schema_context(session)
        assert "  - date_col: datetime (null count: 3)" in result

    def test_all_columns_present(self):
        cols = [
            ("id", "numeric", 0),
            ("name", "categorical", 1),
            ("created_at", "datetime", 2),
        ]
        record = _make_record("users.csv", 300, cols)
        session = _make_session({"users.csv": record})
        result = build_schema_context(session)
        assert "id" in result
        assert "name" in result
        assert "created_at" in result

    def test_zero_null_count(self):
        record = _make_record("clean.csv", 5, [("amount", "numeric", 0)])
        session = _make_session({"clean.csv": record})
        result = build_schema_context(session)
        assert "null count: 0" in result


class TestMultipleDatasets:
    def test_all_filenames_appear(self):
        records = {
            "alpha.csv": _make_record("alpha.csv", 10, [("x", "numeric", 0)]),
            "beta.csv": _make_record("beta.csv", 20, [("y", "categorical", 1)]),
        }
        session = _make_session(records)
        result = build_schema_context(session)
        assert "alpha.csv" in result
        assert "beta.csv" in result

    def test_datasets_separated_by_blank_line(self):
        records = {
            "a.csv": _make_record("a.csv", 5, [("col1", "numeric", 0)]),
            "b.csv": _make_record("b.csv", 5, [("col2", "numeric", 0)]),
        }
        session = _make_session(records)
        result = build_schema_context(session)
        # Blank line separation means "\n\n" appears between blocks
        assert "\n\n" in result

    def test_three_datasets_all_present(self):
        records = {
            "c.csv": _make_record("c.csv", 1, [("c1", "categorical", 0)]),
            "a.csv": _make_record("a.csv", 2, [("a1", "numeric", 0)]),
            "b.csv": _make_record("b.csv", 3, [("b1", "datetime", 0)]),
        }
        session = _make_session(records)
        result = build_schema_context(session)
        assert "a.csv" in result
        assert "b.csv" in result
        assert "c.csv" in result


# ---------------------------------------------------------------------------
# AC2 — determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_session_same_output(self):
        record = _make_record("data.csv", 100, [("value", "numeric", 5)])
        session = _make_session({"data.csv": record})
        result1 = build_schema_context(session)
        result2 = build_schema_context(session)
        assert result1 == result2

    def test_insertion_order_does_not_affect_output(self):
        """Datasets inserted in different orders must produce the same string."""
        cols = [("amount", "numeric", 0)]
        record_a = _make_record("aaa.csv", 10, cols)
        record_b = _make_record("bbb.csv", 20, cols)

        # Insert in alphabetical order
        session1 = _make_session({"aaa.csv": record_a, "bbb.csv": record_b})
        # Insert in reverse alphabetical order
        session2 = _make_session({"bbb.csv": record_b, "aaa.csv": record_a})

        assert build_schema_context(session1) == build_schema_context(session2)

    def test_sorted_dataset_order(self):
        """First dataset block must be the lexicographically smallest filename."""
        record_z = _make_record("z_data.csv", 1, [("col", "numeric", 0)])
        record_a = _make_record("a_data.csv", 2, [("col", "numeric", 0)])
        session = _make_session({"z_data.csv": record_z, "a_data.csv": record_a})
        result = build_schema_context(session)
        pos_a = result.index("a_data.csv")
        pos_z = result.index("z_data.csv")
        assert pos_a < pos_z, "a_data.csv should appear before z_data.csv"

    def test_multiple_calls_identical(self):
        cols = [("x", "datetime", 0), ("y", "categorical", 3)]
        record = _make_record("report.csv", 500, cols)
        session = _make_session({"report.csv": record})
        outputs = [build_schema_context(session) for _ in range(5)]
        assert len(set(outputs)) == 1, "All calls must return the same string"

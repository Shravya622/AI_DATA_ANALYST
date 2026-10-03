"""
Integration tests for POST /sessions/{session_id}/files

Covers all acceptance criteria from task 3.4:
  - Valid CSV → row count + column schemas returned.
  - Non-CSV file → FileUploadResponse with non-empty errors.
  - File > 50 MB → FileUploadResponse with size-limit error.
  - Multiple valid CSVs in one request → all stored, one response per file.
  - Non-existent session ID → HTTP 404.
"""

from __future__ import annotations

import io
import os

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Ensure a dummy OPENAI_API_KEY is present so app/config.py doesn't exit(1)
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-integration-tests")

from app.main import app  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Fixtures — small CSV payloads defined inline as bytes
# ---------------------------------------------------------------------------

VALID_CSV_BYTES: bytes = (
    b"name,age,salary\n"
    b"Alice,30,70000\n"
    b"Bob,25,50000\n"
    b"Charlie,35,90000\n"
)

VALID_CSV_2_BYTES: bytes = (
    b"product,revenue,region\n"
    b"Widget,1000,North\n"
    b"Gadget,2000,South\n"
    b"Doohickey,3000,East\n"
)

NON_CSV_BYTES: bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR fake png bytes"


def _post_session() -> str:
    """Helper: create a new session and return its ID."""
    resp = client.post("/sessions")
    assert resp.status_code == 201, f"Session creation failed: {resp.text}"
    return resp.json()["session_id"]


def _upload(
    session_id: str,
    files: list[tuple[str, bytes, str]],
) -> list[dict]:
    """Helper: upload files and return parsed JSON list.

    *files* is a list of (field_name, content_bytes, filename).
    """
    multipart = [
        ("files", (fname, io.BytesIO(data), "application/octet-stream"))
        for _, data, fname in files
    ]
    resp = client.post(f"/sessions/{session_id}/files", files=multipart)
    assert resp.status_code == 200, f"Unexpected status {resp.status_code}: {resp.text}"
    return resp.json()


# ---------------------------------------------------------------------------
# AC1 — Valid CSV returns row count and column schemas
# ---------------------------------------------------------------------------


class TestValidCsvUpload:
    def test_row_count_is_correct(self):
        session_id = _post_session()
        results = _upload(session_id, [("files", VALID_CSV_BYTES, "data.csv")])
        assert len(results) == 1
        result = results[0]
        assert result["errors"] == []
        # VALID_CSV_BYTES has 3 data rows
        assert result["row_count"] == 3

    def test_column_schemas_returned(self):
        session_id = _post_session()
        results = _upload(session_id, [("files", VALID_CSV_BYTES, "data.csv")])
        result = results[0]
        schema = result["column_schemas"]
        # 3 columns: name, age, salary
        assert len(schema) == 3
        col_names = {col["name"] for col in schema}
        assert col_names == {"name", "age", "salary"}

    def test_column_dtypes_inferred(self):
        session_id = _post_session()
        results = _upload(session_id, [("files", VALID_CSV_BYTES, "data.csv")])
        schema_by_name = {
            col["name"]: col for col in results[0]["column_schemas"]
        }
        # 'name' is categorical, 'age' and 'salary' are numeric
        assert schema_by_name["name"]["dtype"] == "categorical"
        assert schema_by_name["age"]["dtype"] == "numeric"
        assert schema_by_name["salary"]["dtype"] == "numeric"

    def test_filename_in_response(self):
        session_id = _post_session()
        results = _upload(session_id, [("files", VALID_CSV_BYTES, "mydata.csv")])
        assert results[0]["filename"] == "mydata.csv"


# ---------------------------------------------------------------------------
# AC2 — Non-CSV file returns FileUploadResponse with non-empty errors
# ---------------------------------------------------------------------------


class TestNonCsvUpload:
    def test_non_csv_extension_rejected(self):
        session_id = _post_session()
        results = _upload(session_id, [("files", NON_CSV_BYTES, "image.png")])
        assert len(results) == 1
        result = results[0]
        assert len(result["errors"]) > 0, "Expected a non-empty errors list"
        assert result["row_count"] == 0
        assert result["column_schemas"] == []

    def test_non_csv_error_message_mentions_filename(self):
        session_id = _post_session()
        results = _upload(session_id, [("files", NON_CSV_BYTES, "report.xlsx")])
        error_text = " ".join(results[0]["errors"])
        assert "report.xlsx" in error_text

    def test_non_csv_bytes_with_csv_extension_rejected(self):
        """PNG bytes disguised with .csv extension should fail CSV parsing."""
        session_id = _post_session()
        results = _upload(session_id, [("files", NON_CSV_BYTES, "fake.csv")])
        result = results[0]
        # validate_csv should detect this is not a valid CSV
        assert len(result["errors"]) > 0


# ---------------------------------------------------------------------------
# AC3 — File > 50 MB returns error
# ---------------------------------------------------------------------------


class TestOversizeUpload:
    def test_oversize_file_rejected(self):
        session_id = _post_session()
        # Build a CSV that exceeds 50 MB
        header = b"col_a,col_b,col_c\n"
        row = b"aaaaaaaaaa,bbbbbbbbbb,cccccccccc\n"  # ~32 bytes per row
        # 50 MB + a bit extra: ~1_600_000 rows × 32 bytes ≈ 51.2 MB
        oversize_csv = header + row * 1_650_000
        assert len(oversize_csv) > 50 * 1024 * 1024

        results = _upload(session_id, [("files", oversize_csv, "big.csv")])
        result = results[0]
        assert len(result["errors"]) > 0
        error_text = " ".join(result["errors"])
        # Error should mention size limit
        assert any(
            kw in error_text.lower() for kw in ("50", "mb", "size", "limit")
        ), f"Expected size-limit mention in: {error_text!r}"

    def test_oversize_file_row_count_zero(self):
        session_id = _post_session()
        # Build a CSV that is clearly > 50 MB
        header = b"col_a,col_b,col_c\n"
        row = b"aaaaaaaaaa,bbbbbbbbbb,cccccccccc\n"  # ~32 bytes per row
        oversize_csv = header + row * 1_650_000  # ~52.8 MB
        assert len(oversize_csv) > 50 * 1024 * 1024
        results = _upload(session_id, [("files", oversize_csv, "huge.csv")])
        assert results[0]["row_count"] == 0


# ---------------------------------------------------------------------------
# AC4 — Multiple valid CSVs → all stored, one response per file
# ---------------------------------------------------------------------------


class TestMultipleFileUpload:
    def test_two_valid_csvs_returns_two_responses(self):
        session_id = _post_session()
        results = _upload(
            session_id,
            [
                ("files", VALID_CSV_BYTES, "people.csv"),
                ("files", VALID_CSV_2_BYTES, "sales.csv"),
            ],
        )
        assert len(results) == 2

    def test_both_files_have_no_errors(self):
        session_id = _post_session()
        results = _upload(
            session_id,
            [
                ("files", VALID_CSV_BYTES, "people.csv"),
                ("files", VALID_CSV_2_BYTES, "sales.csv"),
            ],
        )
        for r in results:
            assert r["errors"] == [], f"Unexpected error in {r['filename']}: {r['errors']}"

    def test_correct_filenames_in_responses(self):
        session_id = _post_session()
        results = _upload(
            session_id,
            [
                ("files", VALID_CSV_BYTES, "people.csv"),
                ("files", VALID_CSV_2_BYTES, "sales.csv"),
            ],
        )
        returned_names = {r["filename"] for r in results}
        assert returned_names == {"people.csv", "sales.csv"}

    def test_correct_row_counts(self):
        session_id = _post_session()
        results = _upload(
            session_id,
            [
                ("files", VALID_CSV_BYTES, "people.csv"),
                ("files", VALID_CSV_2_BYTES, "sales.csv"),
            ],
        )
        by_name = {r["filename"]: r for r in results}
        assert by_name["people.csv"]["row_count"] == 3
        assert by_name["sales.csv"]["row_count"] == 3

    def test_mixed_valid_and_invalid_returns_both(self):
        """One valid + one non-CSV: two responses with appropriate success/error."""
        session_id = _post_session()
        results = _upload(
            session_id,
            [
                ("files", VALID_CSV_BYTES, "good.csv"),
                ("files", NON_CSV_BYTES, "bad.png"),
            ],
        )
        assert len(results) == 2
        by_name = {r["filename"]: r for r in results}
        assert by_name["good.csv"]["errors"] == []
        assert len(by_name["bad.png"]["errors"]) > 0


# ---------------------------------------------------------------------------
# AC5 — Non-existent session ID → HTTP 404
# ---------------------------------------------------------------------------


class TestNonExistentSession:
    def test_unknown_session_returns_404(self):
        fake_session_id = "00000000-0000-0000-0000-000000000000"
        multipart = [
            ("files", ("data.csv", io.BytesIO(VALID_CSV_BYTES), "application/octet-stream"))
        ]
        resp = client.post(
            f"/sessions/{fake_session_id}/files", files=multipart
        )
        assert resp.status_code == 404

    def test_404_response_has_detail(self):
        fake_session_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        multipart = [
            ("files", ("data.csv", io.BytesIO(VALID_CSV_BYTES), "application/octet-stream"))
        ]
        resp = client.post(
            f"/sessions/{fake_session_id}/files", files=multipart
        )
        body = resp.json()
        assert "detail" in body

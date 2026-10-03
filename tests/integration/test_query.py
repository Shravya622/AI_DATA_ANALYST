"""
Integration tests for POST /sessions/{session_id}/query

Covers all acceptance criteria from task 7.2:
  AC1 — Valid query to an existing session returns HTTP 200 with a QueryResponse body.
  AC2 — Query to a non-existent session returns HTTP 404.
  AC3 — Missing `query` field returns HTTP 422.
  AC4 — LLM API failure returns HTTP 503.
"""

from __future__ import annotations

import io
import os
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure OPENAI_API_KEY is set before importing app modules.
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-integration-tests")

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402
from app.llm_client import LLMResponse, ToolCall  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

VALID_CSV_BYTES: bytes = (
    b"name,age,salary\n"
    b"Alice,30,70000\n"
    b"Bob,25,50000\n"
    b"Charlie,35,90000\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_session() -> str:
    """Create a new session and return its session_id."""
    resp = client.post("/sessions")
    assert resp.status_code == 201, f"Session creation failed: {resp.text}"
    return resp.json()["session_id"]


def _upload_csv(session_id: str, csv_bytes: bytes = VALID_CSV_BYTES, filename: str = "data.csv") -> None:
    """Upload a CSV file to the given session."""
    resp = client.post(
        f"/sessions/{session_id}/files",
        files=[("files", (filename, io.BytesIO(csv_bytes), "text/csv"))],
    )
    assert resp.status_code == 200, f"File upload failed: {resp.text}"


def _fake_dataset_profile_llm_response() -> LLMResponse:
    """Return an LLMResponse that simulates the LLM invoking dataset_profile."""
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCall(
                id="call_test_001",
                name="dataset_profile",
                arguments={},
            )
        ],
    )


# ---------------------------------------------------------------------------
# AC1 — Valid query to an existing session returns HTTP 200 with QueryResponse
# ---------------------------------------------------------------------------


class TestValidQuery:
    def test_valid_query_returns_200(self):
        """A well-formed query to an existing session returns HTTP 200."""
        session_id = _create_session()
        _upload_csv(session_id)

        with patch("app.analyst.call_llm", return_value=_fake_dataset_profile_llm_response()):
            resp = client.post(
                f"/sessions/{session_id}/query",
                json={"query": "Profile the dataset", "dataset_filename": "data.csv"},
            )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    def test_valid_query_returns_query_response_body(self):
        """Response body contains all required QueryResponse fields."""
        session_id = _create_session()
        _upload_csv(session_id)

        with patch("app.analyst.call_llm", return_value=_fake_dataset_profile_llm_response()):
            resp = client.post(
                f"/sessions/{session_id}/query",
                json={"query": "Profile the dataset", "dataset_filename": "data.csv"},
            )

        body = resp.json()
        # Required fields defined in QueryResponse
        assert "answer" in body, "Response missing 'answer'"
        assert "reasoning_trace" in body, "Response missing 'reasoning_trace'"
        assert body["answer"], "answer should be non-empty"

    def test_valid_query_reasoning_trace_populated(self):
        """reasoning_trace is always present and structured correctly."""
        session_id = _create_session()
        _upload_csv(session_id)

        with patch("app.analyst.call_llm", return_value=_fake_dataset_profile_llm_response()):
            resp = client.post(
                f"/sessions/{session_id}/query",
                json={"query": "What does the data look like?", "dataset_filename": "data.csv"},
            )

        trace = resp.json()["reasoning_trace"]
        assert "tools_selected" in trace
        assert "columns_used" in trace
        assert "computation_description" in trace

    def test_valid_query_without_dataset_filename(self):
        """dataset_filename is optional; omitting it should still return 200.

        Fix C: when the LLM returns non-empty content and no tool call,
        that content is preserved as the answer (not replaced with failure message).
        """
        session_id = _create_session()
        _upload_csv(session_id)

        llm_resp = LLMResponse(content="Here is what I found.", tool_calls=[])
        with patch("app.analyst.call_llm", return_value=llm_resp):
            resp = client.post(
                f"/sessions/{session_id}/query",
                json={"query": "Tell me something about the data"},
            )

        assert resp.status_code == 200
        # Fix C: LLM content is now preserved, not replaced with failure message
        assert resp.json()["answer"] == "Here is what I found."

    def test_valid_query_tool_call_reflected_in_tools_selected(self):
        """When LLM invokes dataset_profile, tools_selected lists it."""
        session_id = _create_session()
        _upload_csv(session_id)

        with patch("app.analyst.call_llm", return_value=_fake_dataset_profile_llm_response()):
            resp = client.post(
                f"/sessions/{session_id}/query",
                json={"query": "Profile the data", "dataset_filename": "data.csv"},
            )

        trace = resp.json()["reasoning_trace"]
        assert "dataset_profile" in trace["tools_selected"]


# ---------------------------------------------------------------------------
# AC2 — Query to a non-existent session returns HTTP 404
# ---------------------------------------------------------------------------


class TestNonExistentSession:
    def test_unknown_session_returns_404(self):
        """POST /sessions/{id}/query with an unknown session_id → 404."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        resp = client.post(
            f"/sessions/{fake_id}/query",
            json={"query": "What is in the data?"},
        )
        assert resp.status_code == 404, f"Expected 404, got {resp.status_code}: {resp.text}"

    def test_unknown_session_404_has_detail(self):
        """404 response includes a 'detail' key in the body."""
        fake_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        resp = client.post(
            f"/sessions/{fake_id}/query",
            json={"query": "Anything"},
        )
        body = resp.json()
        assert "detail" in body


# ---------------------------------------------------------------------------
# AC3 — Missing `query` field returns HTTP 422
# ---------------------------------------------------------------------------


class TestMissingQueryField:
    def test_missing_query_field_returns_422(self):
        """Omitting the required 'query' field triggers Pydantic validation → 422."""
        session_id = _create_session()
        resp = client.post(
            f"/sessions/{session_id}/query",
            json={"dataset_filename": "data.csv"},  # 'query' is absent
        )
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}: {resp.text}"

    def test_missing_query_field_422_has_validation_errors(self):
        """422 body contains Pydantic validation error details."""
        session_id = _create_session()
        resp = client.post(
            f"/sessions/{session_id}/query",
            json={},
        )
        assert resp.status_code == 422
        body = resp.json()
        # FastAPI wraps Pydantic errors under 'detail'
        assert "detail" in body

    def test_empty_body_returns_422(self):
        """Sending an empty JSON body (no fields at all) also yields 422."""
        session_id = _create_session()
        resp = client.post(
            f"/sessions/{session_id}/query",
            json={},
        )
        assert resp.status_code == 422

    def test_null_query_returns_422(self):
        """Explicitly setting 'query' to null should fail validation → 422."""
        session_id = _create_session()
        resp = client.post(
            f"/sessions/{session_id}/query",
            json={"query": None},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# AC4 — LLM API failure returns HTTP 503
# ---------------------------------------------------------------------------


class TestLLMFailure:
    def test_openai_api_error_returns_503(self):
        """When call_llm raises openai.APIError, the endpoint returns 503."""
        import openai

        session_id = _create_session()
        _upload_csv(session_id)

        mock_request = MagicMock()
        mock_request.url = "https://api.openai.com/v1/chat/completions"

        with patch(
            "app.analyst.call_llm",
            side_effect=openai.APIError("Service unavailable", request=mock_request, body=None),
        ):
            resp = client.post(
                f"/sessions/{session_id}/query",
                json={"query": "Summarise the data", "dataset_filename": "data.csv"},
            )

        assert resp.status_code == 503, f"Expected 503, got {resp.status_code}: {resp.text}"

    def test_503_response_has_detail(self):
        """503 response body contains a 'detail' key."""
        import openai

        session_id = _create_session()
        _upload_csv(session_id)

        mock_request = MagicMock()
        mock_request.url = "https://api.openai.com/v1/chat/completions"

        with patch(
            "app.analyst.call_llm",
            side_effect=openai.APIError("Rate limit exceeded", request=mock_request, body=None),
        ):
            resp = client.post(
                f"/sessions/{session_id}/query",
                json={"query": "Show me the stats"},
            )

        assert "detail" in resp.json()

    def test_openai_rate_limit_error_returns_503(self):
        """openai.RateLimitError (a subclass of APIError) also yields 503."""
        import openai

        session_id = _create_session()
        _upload_csv(session_id)

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}

        with patch(
            "app.analyst.call_llm",
            side_effect=openai.RateLimitError(
                "Rate limit exceeded",
                response=mock_response,
                body={"error": {"message": "Rate limit exceeded"}},
            ),
        ):
            resp = client.post(
                f"/sessions/{session_id}/query",
                json={"query": "Aggregate by region"},
            )

        assert resp.status_code == 503

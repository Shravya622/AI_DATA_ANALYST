"""
Unit tests for the SanitisationMiddleware (task 2.5).

Acceptance criteria verified:
  - A request with ``../`` in the session ID path parameter returns 422.
  - A request with ``\x00`` in the query string / body field returns 422.
  - A clean request with no traversal sequences is not blocked (returns
    the normal handler response, not 422).

Additional cases:
  - ``..\`` (Windows-style) in path → 422
  - Percent-encoded ``%2e%2e%2f`` in path → 422
  - ``../`` in a JSON body string value → 422
  - ``\x00`` in a JSON body string value → 422
  - Multipart (non-JSON) body with an arbitrary binary value → not blocked
  - ``../`` in a query parameter key or value → 422
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app

# raise_server_exceptions=False so HTTP error responses are returned normally
# rather than re-raised as Python exceptions in the test process.
client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ERROR_DETAIL = "Invalid input: path traversal or null byte detected"


def _is_blocked(response) -> bool:
    return (
        response.status_code == 422
        and response.json().get("detail") == ERROR_DETAIL
    )


# ---------------------------------------------------------------------------
# Path / path-parameter checks
# ---------------------------------------------------------------------------


class TestPathTraversal:
    def test_unix_traversal_in_path_returns_422(self):
        """``../`` percent-encoded in a path parameter → 422.

        Raw ``../`` is normalised away by the HTTP client before the request
        reaches the server (RFC 3986 path resolution).  The realistic attack
        vector is a percent-encoded form that bypasses normalisation, e.g.
        ``..%2f``.  The middleware decodes the path before inspection so this
        is still caught.
        """
        response = client.get("/sessions/..%2fetc%2fpasswd")
        assert _is_blocked(response)

    def test_windows_traversal_in_path_returns_422(self):
        """``..\\`` embedded in a path parameter → 422."""
        response = client.delete("/sessions/..\\windows\\system32")
        assert _is_blocked(response)

    def test_null_byte_in_path_returns_422(self):
        """Null byte (percent-encoded as ``%00``) in path → 422.

        Raw null bytes are rejected by the HTTP client library before the
        request is sent; percent-encoded ``%00`` is the realistic attack form
        and must be caught after URL-decoding.
        """
        response = client.get("/sessions/abc%00def")
        assert _is_blocked(response)

    def test_percent_encoded_traversal_in_path_returns_422(self):
        """URL-encoded ``%2e%2e%2f`` must also be blocked."""
        response = client.get("/sessions/%2e%2e%2fetc%2fpasswd")
        assert _is_blocked(response)

    def test_clean_path_is_not_blocked(self):
        """A normal path that hits the health endpoint must not be blocked."""
        response = client.get("/health")
        assert response.status_code == 200
        assert not _is_blocked(response)


# ---------------------------------------------------------------------------
# Query string checks
# ---------------------------------------------------------------------------


class TestQueryStringTraversal:
    def test_traversal_in_query_value_returns_422(self):
        """``../`` in a query parameter value → 422."""
        response = client.get("/health", params={"file": "../secret"})
        assert _is_blocked(response)

    def test_null_byte_in_query_value_returns_422(self):
        """Null byte in a query parameter value → 422."""
        response = client.get("/health", params={"q": "data\x00injected"})
        assert _is_blocked(response)

    def test_traversal_in_query_key_returns_422(self):
        """``../`` in a query parameter key → 422."""
        # httpx / requests encode the key literally; the middleware must catch it.
        response = client.get("/health?..%2F=value")
        assert _is_blocked(response)

    def test_clean_query_string_is_not_blocked(self):
        """Normal query parameters must not trigger the middleware."""
        response = client.get("/health", params={"status": "active"})
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# JSON body checks
# ---------------------------------------------------------------------------


class TestJsonBodyTraversal:
    """Middleware must scan string values inside a JSON request body."""

    def test_traversal_in_json_body_string_returns_422(self):
        """``../`` as a string value in a JSON body → 422."""
        payload = {"query": "show ../etc/passwd"}
        # We POST to /sessions (cheapest JSON-accepting endpoint in the app)
        response = client.post(
            "/sessions",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        assert _is_blocked(response)

    def test_null_byte_in_json_body_string_returns_422(self):
        """Null byte embedded in a JSON string value → 422."""
        payload = {"query": "select \x00 from table"}
        response = client.post(
            "/sessions",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        assert _is_blocked(response)

    def test_traversal_in_nested_json_value_returns_422(self):
        """Traversal sequence nested inside a JSON object → 422."""
        payload = {"outer": {"inner": "../traversal"}}
        response = client.post(
            "/sessions",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        assert _is_blocked(response)

    def test_traversal_in_json_list_value_returns_422(self):
        """Traversal sequence inside a JSON array element → 422."""
        payload = {"files": ["normal.csv", "../secret.csv"]}
        response = client.post(
            "/sessions",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        assert _is_blocked(response)

    def test_clean_json_body_is_not_blocked(self):
        """A valid JSON body with no traversal sequences must pass through."""
        # POST /sessions takes no body; an extra JSON body should be ignored
        # by the router but must not be blocked by the middleware.
        response = client.post(
            "/sessions",
            content=json.dumps({"note": "clean payload"}),
            headers={"Content-Type": "application/json"},
        )
        # The sessions router returns 201 on success.
        assert response.status_code == 201

    def test_malformed_json_body_is_not_blocked_by_middleware(self):
        """Malformed JSON must not cause the middleware to return 422.
        FastAPI's own validation will handle it.
        """
        response = client.post(
            "/sessions",
            content=b"not-valid-json{{{",
            headers={"Content-Type": "application/json"},
        )
        # The middleware should pass this through; the router returns 201 (it
        # ignores the body for this endpoint) or a parse error — but NOT our
        # middleware's 422 with the traversal detail.
        if response.status_code == 422:
            assert response.json().get("detail") != ERROR_DETAIL


# ---------------------------------------------------------------------------
# Non-JSON content types (e.g. multipart) must not be scanned
# ---------------------------------------------------------------------------


class TestNonJsonBodyNotScanned:
    def test_multipart_body_not_blocked(self):
        """Multipart bodies must not be read/scanned by the middleware."""
        # We intentionally don't scan non-JSON bodies; this call should not
        # return the middleware 422 (the router itself may return a different
        # status for an unrecognised endpoint, which is fine).
        import io

        response = client.post(
            "/sessions",
            files={"file": ("test.csv", io.BytesIO(b"col1,col2\n1,2"), "text/csv")},
        )
        # Should NOT be blocked by middleware — status may be anything else.
        assert not _is_blocked(response)

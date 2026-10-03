"""
Unit tests for the sessions router (task 2.4).

Tests cover:
  POST   /sessions  — returns 201 with a UUID session_id
  DELETE /sessions/{id} — returns 204 for a valid ID, 404 for unknown
  GET    /health    — always returns 200 with {"status": "ok"}
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# POST /sessions
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_returns_201(self):
        response = client.post("/sessions")
        assert response.status_code == 201

    def test_response_contains_session_id(self):
        response = client.post("/sessions")
        body = response.json()
        assert "session_id" in body

    def test_session_id_is_valid_uuid(self):
        response = client.post("/sessions")
        session_id = response.json()["session_id"]
        # uuid.UUID constructor raises ValueError for invalid strings
        parsed = uuid.UUID(session_id)
        assert str(parsed) == session_id

    def test_each_call_returns_unique_session_id(self):
        id1 = client.post("/sessions").json()["session_id"]
        id2 = client.post("/sessions").json()["session_id"]
        assert id1 != id2


# ---------------------------------------------------------------------------
# DELETE /sessions/{session_id}
# ---------------------------------------------------------------------------


class TestDeleteSession:
    def _create_session(self) -> str:
        return client.post("/sessions").json()["session_id"]

    def test_valid_id_returns_204(self):
        session_id = self._create_session()
        response = client.delete(f"/sessions/{session_id}")
        assert response.status_code == 204

    def test_valid_id_returns_empty_body(self):
        session_id = self._create_session()
        response = client.delete(f"/sessions/{session_id}")
        assert response.content == b""

    def test_unknown_id_returns_404(self):
        fake_id = str(uuid.uuid4())
        response = client.delete(f"/sessions/{fake_id}")
        assert response.status_code == 404

    def test_double_delete_returns_404_on_second(self):
        session_id = self._create_session()
        client.delete(f"/sessions/{session_id}")
        response = client.delete(f"/sessions/{session_id}")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_status_ok(self):
        response = client.get("/health")
        assert response.json() == {"status": "ok"}

    def test_always_returns_ok_across_multiple_calls(self):
        for _ in range(3):
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"

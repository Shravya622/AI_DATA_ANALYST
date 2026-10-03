"""
Sessions router.

Endpoints
---------
POST   /sessions
    Create a new session.  Returns ``CreateSessionResponse`` with a UUID
    ``session_id``.

DELETE /sessions/{session_id}
    Delete an existing session.  Returns HTTP 204 on success.
    ``SessionNotFoundError`` is caught by the global handler in
    ``app.main`` and converted to HTTP 404.

GET    /health
    Liveness probe — always returns HTTP 200 with ``{"status": "ok"}``.
"""

from fastapi import APIRouter, Response

from app.models import CreateSessionResponse
from app.session_store import session_store

router = APIRouter(tags=["sessions"])


@router.post("/sessions", response_model=CreateSessionResponse, status_code=201)
async def create_session() -> CreateSessionResponse:
    """Create a new analysis session and return its UUID."""
    session_id = session_store.create_session()
    return CreateSessionResponse(session_id=session_id)


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, response: Response) -> None:
    """
    Delete a session by ID.

    Returns HTTP 204 on success.  If the session does not exist,
    ``SessionNotFoundError`` propagates to the global exception handler
    which returns HTTP 404.
    """
    session_store.delete_session(session_id)


@router.get("/health", tags=["health"])
async def health_check() -> dict:
    """Liveness probe — always returns HTTP 200 with status ok."""
    return {"status": "ok"}

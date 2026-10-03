"""
Query router — POST /sessions/{session_id}/query

Accepts a QueryRequest, delegates to the Analyst orchestrator, and returns a
QueryResponse.

Error mapping
-------------
- SessionNotFoundError  → propagates to the global 404 handler in main.py
- openai.APIError / any LLM-level failure → HTTP 503 Service Unavailable
- Pydantic validation (missing/invalid `query` field) → HTTP 422 (automatic)
"""

from __future__ import annotations

import logging

import openai
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app import analyst
from app.exceptions import SessionNotFoundError
from app.models import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


@router.post(
    "/sessions/{session_id}/query",
    response_model=QueryResponse,
    status_code=200,
)
async def query_session(
    session_id: str,
    request: QueryRequest,
) -> QueryResponse | JSONResponse:
    """Run a natural-language query against the datasets in a session.

    Parameters
    ----------
    session_id:
        UUID identifying the active session.
    request:
        Validated request body containing ``query`` and optional
        ``dataset_filename``.

    Returns
    -------
    QueryResponse
        HTTP 200 on success.

    HTTP Error Responses
    --------------------
    404 — session not found (raised by SessionStore, handled globally).
    422 — missing or invalid ``query`` field (handled by Pydantic automatically).
    503 — LLM service is unavailable or returned a fatal API error.
    """
    try:
        response: QueryResponse = await analyst.analyse(
            session_id,
            request.query,
            request.dataset_filename,
        )
        return response
    except SessionNotFoundError:
        # Re-raise so the global exception handler in main.py returns 404.
        raise
    except openai.APIError as exc:
        logger.error(
            "LLM API error while processing query for session %s: %s",
            session_id,
            exc,
        )
        return JSONResponse(
            status_code=503,
            content={"detail": "LLM service is currently unavailable. Please try again later."},
        )
    except Exception as exc:  # noqa: BLE001
        # Check if the exception message indicates an LLM failure surfaced
        # from within analyst.py (e.g. wrapped openai errors propagated up).
        exc_str = str(exc).lower()
        if any(
            keyword in exc_str
            for keyword in ("llm", "openai", "api error", "service unavailable", "rate limit")
        ):
            logger.error(
                "LLM-related error for session %s: %s", session_id, exc
            )
            return JSONResponse(
                status_code=503,
                content={"detail": "LLM service is currently unavailable. Please try again later."},
            )
        # All other unhandled exceptions propagate to the global 500 handler.
        raise

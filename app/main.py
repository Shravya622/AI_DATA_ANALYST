"""
FastAPI application entry point.

Startup behaviour
-----------------
- Validates required environment variables via ``app.config.Settings``.
  If any required variable is missing the application logs the names of
  the missing fields and exits with code 1.  Environment variable
  *values* are never written to any log — only the variable names are
  reported.

Exception handling
------------------
- ``SessionNotFoundError``   → HTTP 404
- ``DatasetNotFoundError``   → HTTP 404
- Any other unhandled ``Exception`` → HTTP 500 with a UUID error
  reference logged at ERROR level (full traceback) but never exposed
  in the response body.
"""

import logging
import sys
import traceback
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration validation
# Importing Settings triggers environment-variable resolution.  Any missing
# required variable raises a pydantic.ValidationError before the FastAPI
# application is created, so the app never starts in a misconfigured state.
# ---------------------------------------------------------------------------
try:
    from app.config import settings  # noqa: F401 — import for side-effect/validation
except ValidationError as exc:
    missing: list[str] = []
    for error in exc.errors():
        # Each error dict has a 'loc' tuple identifying the field path.
        # We join the path elements to produce a readable variable name.
        field_name = ".".join(str(loc) for loc in error["loc"])
        missing.append(field_name)

    logger.error(
        "Application startup failed — the following required environment "
        "variable(s) are not set: %s",
        ", ".join(missing),
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Custom exceptions (imported after config so startup failure is clean)
# ---------------------------------------------------------------------------
from app.exceptions import DatasetNotFoundError, SessionNotFoundError  # noqa: E402

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI-Powered Data Analyst",
    description=(
        "Upload CSV datasets and ask natural-language questions. "
        "Powered by GPT-4o tool-calling."
    ),
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(SessionNotFoundError)
async def session_not_found_handler(
    request: Request, exc: SessionNotFoundError
) -> JSONResponse:
    """Return HTTP 404 when a session ID is not found in the store."""
    return JSONResponse(
        status_code=404,
        content={"detail": exc.message},
    )


@app.exception_handler(DatasetNotFoundError)
async def dataset_not_found_handler(
    request: Request, exc: DatasetNotFoundError
) -> JSONResponse:
    """Return HTTP 404 when a dataset filename is not found in a session."""
    return JSONResponse(
        status_code=404,
        content={"detail": exc.message},
    )


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Catch-all handler for any unhandled exception.

    - Generates a UUID error reference so a support ticket can correlate
      the user-facing error ID to a specific log entry.
    - Logs the full traceback at ERROR level for internal diagnostics.
    - Returns HTTP 500 with *only* the error reference — no internal
      details are exposed to the caller.
    """
    error_id = str(uuid.uuid4())
    logger.error(
        "Unhandled exception [error_id=%s] on %s %s\n%s",
        error_id,
        request.method,
        request.url.path,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error_id": error_id,
            "message": f"An unexpected error occurred. Reference: {error_id}",
        },
    )


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
from app.middleware import SanitisationMiddleware  # noqa: E402

app.add_middleware(SanitisationMiddleware)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
from app.routers import files, query, sessions  # noqa: E402

app.include_router(sessions.router)
app.include_router(files.router)
app.include_router(query.router)


# ---------------------------------------------------------------------------
# Built-in health check (kept alongside the router-level one for convenience)
# ---------------------------------------------------------------------------
@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Liveness probe — always returns 200 OK."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Frontend SPA static file serving (production only)
# When the React build artefacts are present (i.e. inside a Docker container
# built by the multi-stage Dockerfile) the backend serves them directly so
# that a single ``docker run`` is sufficient to start the full application.
# ---------------------------------------------------------------------------
from pathlib import Path  # noqa: E402

from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    # Mount hashed asset bundles (JS/CSS/images) under /assets
    app.mount(
        "/assets",
        StaticFiles(directory=frontend_dist / "assets"),
        name="assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str) -> FileResponse:
        """Catch-all route: serve index.html for any non-API path so that
        the React Router / client-side routing works correctly."""
        return FileResponse(frontend_dist / "index.html")

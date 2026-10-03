"""
Input Sanitisation Middleware.

Inspects all user-supplied string inputs — URL path, query string parameters,
and JSON request body string values — for path traversal sequences and null
bytes.  Any match results in an immediate HTTP 422 response; clean requests
are passed through untouched.

Detected sequences
------------------
- ``../``   — Unix path traversal
- ``..\\``   — Windows path traversal
- ``\\x00``  — Null byte injection

URL-encoded variants (e.g. ``%2e%2e%2f``) are also detected because the
path is decoded with ``urllib.parse.unquote`` before inspection.

Implementation note
-------------------
Uses a pure ASGI middleware class rather than Starlette's ``BaseHTTPMiddleware``
to avoid a known streaming-response incompatibility in Starlette 0.37.x where
``wrapped_receive`` raises ``RuntimeError: Unexpected message received:
http.request`` when the response calls ``listen_for_disconnect``.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import unquote

from starlette.datastructures import Headers, QueryParams
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Sequences that indicate path traversal or null-byte injection attempts.
_BANNED_SEQUENCES: tuple[str, ...] = ("../", "..\\" , "\x00")

_ERROR_BODY = json.dumps(
    {"detail": "Invalid input: path traversal or null byte detected"}
).encode("utf-8")

_ERROR_HEADERS = [
    (b"content-type", b"application/json"),
    (b"content-length", str(len(_ERROR_BODY)).encode()),
]


def _contains_traversal(value: str) -> bool:
    """Return True if *value* contains any banned sequence."""
    lowered = value.lower()
    decoded = unquote(lowered)
    for seq in _BANNED_SEQUENCES:
        if seq.lower() in lowered or seq.lower() in decoded:
            return True
    return False


def _scan_json_value(obj: Any) -> bool:
    """Recursively scan a JSON object for banned sequences in string values."""
    if isinstance(obj, str):
        return _contains_traversal(obj)
    if isinstance(obj, dict):
        return any(_scan_json_value(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_scan_json_value(item) for item in obj)
    return False


async def _send_422(send: Send) -> None:
    """Send an HTTP 422 response directly over the ASGI send channel."""
    await send(
        {
            "type": "http.response.start",
            "status": 422,
            "headers": _ERROR_HEADERS,
        }
    )
    await send({"type": "http.response.body", "body": _ERROR_BODY})


async def _read_body(receive: Receive) -> bytes:
    """Read the full HTTP request body from the receive channel."""
    body = b""
    more_body = True
    while more_body:
        message = await receive()
        body += message.get("body", b"")
        more_body = message.get("more_body", False)
    return body


class SanitisationMiddleware:
    """
    Pure-ASGI middleware that blocks requests containing path traversal
    sequences or null bytes in the URL path, query parameters, and JSON body.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Only inspect HTTP requests; pass websockets and lifespan through.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # ------------------------------------------------------------------
        # 1. Check URL path (covers path parameters such as session_id)
        # ------------------------------------------------------------------
        raw_path: str = scope.get("path", "")
        if _contains_traversal(raw_path):
            logger.warning(
                "Blocked request: traversal sequence in path '%s'", raw_path
            )
            await _send_422(send)
            return

        # ------------------------------------------------------------------
        # 2. Check query string parameters
        # ------------------------------------------------------------------
        query_string: bytes = scope.get("query_string", b"")
        if query_string:
            query_params = QueryParams(query_string.decode("latin-1"))
            for key, value in query_params.multi_items():
                if _contains_traversal(key) or _contains_traversal(value):
                    logger.warning(
                        "Blocked request: traversal sequence in query param '%s'", key
                    )
                    await _send_422(send)
                    return

        # ------------------------------------------------------------------
        # 3. Check JSON request body (only for JSON content types)
        # ------------------------------------------------------------------
        headers = Headers(scope=scope)
        content_type = headers.get("content-type", "")

        if "application/json" in content_type:
            # Read the entire body from the receive channel.
            body_bytes = await _read_body(receive)

            if body_bytes:
                try:
                    parsed = json.loads(body_bytes)
                    if _scan_json_value(parsed):
                        logger.warning(
                            "Blocked request: traversal sequence in JSON body "
                            "on %s %s",
                            scope.get("method", ""),
                            raw_path,
                        )
                        await _send_422(send)
                        return
                except json.JSONDecodeError:
                    # Malformed JSON — let FastAPI's validation handle it.
                    pass

            # Replay the buffered body back to the downstream ASGI app.
            # The replay callable must also handle subsequent disconnect polls;
            # returning http.request again would cause Starlette's
            # listen_for_disconnect task to raise RuntimeError.
            _body_consumed = False

            async def replay_receive() -> dict:
                nonlocal _body_consumed
                if not _body_consumed:
                    _body_consumed = True
                    return {
                        "type": "http.request",
                        "body": body_bytes,
                        "more_body": False,
                    }
                # Simulate a disconnected client for subsequent calls.
                return {"type": "http.disconnect"}

            await self.app(scope, replay_receive, send)
            return

        # Non-JSON request (multipart, plain-text, etc.) — pass through.
        await self.app(scope, receive, send)

"""
Files router — POST /sessions/{session_id}/files

Accepts one or more CSV files as multipart/form-data.  For each file:

1. Validates the ``.csv`` extension.
2. Reads the raw bytes.
3. Calls ``validate_csv`` from ``app.validator``.
4. On success: calls ``infer_schema``, stores a ``DatasetRecord`` +
   ``DataFrame`` via ``session_store.add_dataset``, and returns a
   successful ``FileUploadResponse``.
5. On any failure: returns a ``FileUploadResponse`` with the error
   message in ``errors``.

Uploading to a non-existent session raises ``SessionNotFoundError``
which the global handler in ``app.main`` converts to HTTP 404.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, File, UploadFile

from app.models import FileUploadResponse
from app.session_store import session_store
from app.type_inferrer import infer_schema
from app.validator import validate_csv

logger = logging.getLogger(__name__)

router = APIRouter(tags=["files"])


@router.post(
    "/sessions/{session_id}/files",
    response_model=List[FileUploadResponse],
    summary="Upload one or more CSV files to an existing session",
)
async def upload_files(
    session_id: str,
    files: List[UploadFile] = File(...),
) -> List[FileUploadResponse]:
    """Upload CSV files to an existing session.

    Parameters
    ----------
    session_id:
        The UUID of the target session.  Raises ``SessionNotFoundError``
        (→ HTTP 404) if the session does not exist.
    files:
        One or more files sent as ``multipart/form-data``.

    Returns
    -------
    list[FileUploadResponse]
        One entry per uploaded file.  Check the ``errors`` field to
        determine whether each file was accepted or rejected.
    """
    # Validate that the session exists *before* processing any files so
    # that the caller gets 404 immediately rather than a partial response.
    # get_session raises SessionNotFoundError → caught by the global handler.
    session_store.get_session(session_id)

    responses: List[FileUploadResponse] = []

    for upload in files:
        filename: str = upload.filename or "unknown"

        # ------------------------------------------------------------------
        # 1. Validate file extension
        # ------------------------------------------------------------------
        if not filename.lower().endswith(".csv"):
            logger.info(
                "Rejected non-CSV file '%s' for session %s", filename, session_id
            )
            responses.append(
                FileUploadResponse(
                    filename=filename,
                    row_count=0,
                    column_schemas=[],
                    errors=[
                        f"'{filename}' is not a CSV file. "
                        "Only files with a .csv extension are accepted."
                    ],
                )
            )
            continue

        # ------------------------------------------------------------------
        # 2. Read raw bytes
        # ------------------------------------------------------------------
        try:
            content: bytes = await upload.read()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to read upload '%s' for session %s: %s",
                filename,
                session_id,
                exc,
            )
            responses.append(
                FileUploadResponse(
                    filename=filename,
                    row_count=0,
                    column_schemas=[],
                    errors=[f"Could not read '{filename}': {exc}"],
                )
            )
            continue

        # ------------------------------------------------------------------
        # 3. Validate CSV content (size, parsability, structure)
        # ------------------------------------------------------------------
        result = validate_csv(content, filename)

        if not result.ok:
            logger.info(
                "CSV validation failed for '%s' in session %s: %s",
                filename,
                session_id,
                result.error,
            )
            responses.append(
                FileUploadResponse(
                    filename=filename,
                    row_count=0,
                    column_schemas=[],
                    errors=[result.error],  # type: ignore[list-item]
                )
            )
            continue

        # ------------------------------------------------------------------
        # 4. Infer column schema and store in session
        # ------------------------------------------------------------------
        df = result.dataframe  # guaranteed non-None when ok=True
        schema = infer_schema(df)  # type: ignore[arg-type]

        session_store.add_dataset(session_id, filename, df, schema)  # type: ignore[arg-type]

        logger.info(
            "Stored dataset '%s' (%d rows, %d cols) in session %s",
            filename,
            len(df),  # type: ignore[arg-type]
            len(schema),
            session_id,
        )

        responses.append(
            FileUploadResponse(
                filename=filename,
                row_count=len(df),  # type: ignore[arg-type]
                column_schemas=schema,
                errors=[],
            )
        )

    return responses

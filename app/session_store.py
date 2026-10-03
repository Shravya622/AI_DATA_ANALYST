"""
Session store — in-memory session management.

Provides a module-level singleton ``session_store`` instance backed by a
plain dict.

Public interface
----------------
create_session() -> str
    Create a new session and return its UUID string.

get_session(session_id: str) -> Session
    Return the session or raise ``SessionNotFoundError``.

add_dataset(session_id: str, filename: str, df: pd.DataFrame, schema: list[ColumnSchema]) -> None
    Create a ``DatasetRecord`` from *schema* and store both the record and
    the DataFrame, keyed by *filename*, inside the identified session.

append_history(session_id: str, exchange: Exchange) -> None
    Append *exchange* to the session's history and trim to the most recent
    ``settings.max_history_exchanges`` entries (oldest discarded first).

get_history(session_id: str) -> list[Exchange]
    Return the session's current history (at most 20 exchanges).

build_message_history(session_id: str) -> list[dict]
    Map Exchange history to OpenAI message dicts (last 20, chronological).

reset_session_history(session_id: str) -> None
    Clear all conversation history for the given session.

delete_session(session_id: str) -> None
    Remove the session or raise ``SessionNotFoundError``.
"""

import uuid
from typing import Dict

import pandas as pd

from app.config import settings
from app.exceptions import SessionNotFoundError
from app.models import ColumnSchema, DatasetRecord, Exchange, Session


class SessionStore:
    """Thread-safe-enough for single-process development use."""

    def __init__(self, max_history: int | None = None) -> None:
        self._sessions: Dict[str, Session] = {}
        # Allow override for testing; otherwise use the global settings value.
        self._max_history: int = (
            max_history if max_history is not None else settings.max_history_exchanges
        )

    # ------------------------------------------------------------------
    # Core session lifecycle
    # ------------------------------------------------------------------

    def create_session(self) -> str:
        """Create a new blank session and return its UUID string."""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = Session(
            session_id=session_id,
            datasets={},
            dataframes={},
            history=[],
        )
        return session_id

    def get_session(self, session_id: str) -> Session:
        """Return the session with *session_id*, or raise ``SessionNotFoundError``."""
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session '{session_id}' not found.")
        return session

    def delete_session(self, session_id: str) -> None:
        """Remove the session with *session_id*, or raise ``SessionNotFoundError``."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Session '{session_id}' not found.")
        del self._sessions[session_id]

    # ------------------------------------------------------------------
    # Dataset management
    # ------------------------------------------------------------------

    def add_dataset(
        self,
        session_id: str,
        filename: str,
        df: pd.DataFrame,
        schema: list[ColumnSchema],
    ) -> None:
        """Store *df* and a ``DatasetRecord`` derived from *schema* in the session.

        Parameters
        ----------
        session_id:
            The UUID string identifying the target session.
        filename:
            The original filename used as the storage key.
        df:
            The parsed pandas DataFrame.
        schema:
            List of ``ColumnSchema`` entries produced by the type inferrer.

        Raises
        ------
        SessionNotFoundError
            If *session_id* does not exist in the store.
        """
        session = self.get_session(session_id)
        record = DatasetRecord(
            filename=filename,
            row_count=len(df),
            column_schemas=schema,
        )
        session.datasets[filename] = record
        session.dataframes[filename] = df

    # ------------------------------------------------------------------
    # Conversation history management
    # ------------------------------------------------------------------

    def append_history(self, session_id: str, exchange: Exchange) -> None:
        """Append *exchange* to the session history and enforce the cap.

        Once the history length exceeds ``max_history_exchanges``, the
        oldest entries are discarded so that only the most recent
        ``max_history_exchanges`` exchanges are retained.

        Parameters
        ----------
        session_id:
            The UUID string identifying the target session.
        exchange:
            The ``Exchange`` object to append.

        Raises
        ------
        SessionNotFoundError
            If *session_id* does not exist in the store.
        """
        session = self.get_session(session_id)
        session.history.append(exchange)
        if len(session.history) > self._max_history:
            # Trim oldest entries, keeping exactly max_history_exchanges.
            session.history = session.history[-self._max_history :]

    def get_history(self, session_id: str) -> list[Exchange]:
        """Return the conversation history for *session_id* (at most 20 exchanges).

        Parameters
        ----------
        session_id:
            The UUID string identifying the target session.

        Returns
        -------
        list[Exchange]
            Chronological list of up to ``max_history_exchanges`` exchanges.

        Raises
        ------
        SessionNotFoundError
            If *session_id* does not exist in the store.
        """
        session = self.get_session(session_id)
        return list(session.history)

    def build_message_history(self, session_id: str) -> list[dict]:
        """Map the session's Exchange history to OpenAI-compatible message dicts.

        Returns the last ``max_history_exchanges`` exchanges in chronological
        order (oldest first), each formatted as::

            {"role": exchange.role, "content": exchange.content}

        Parameters
        ----------
        session_id:
            The UUID string identifying the target session.

        Returns
        -------
        list[dict]
            At most 20 message dicts, oldest first.

        Raises
        ------
        SessionNotFoundError
            If *session_id* does not exist in the store.
        """
        exchanges = self.get_history(session_id)
        return [{"role": ex.role, "content": ex.content} for ex in exchanges]

    def reset_session_history(self, session_id: str) -> None:
        """Clear all conversation history for the given session.

        Parameters
        ----------
        session_id:
            The UUID string identifying the target session.

        Raises
        ------
        SessionNotFoundError
            If *session_id* does not exist in the store.
        """
        session = self.get_session(session_id)
        session.history = []


# ---------------------------------------------------------------------------
# Module-level singleton — import this in routers and other modules.
# ---------------------------------------------------------------------------
session_store = SessionStore()

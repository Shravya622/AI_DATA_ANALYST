"""
Pydantic data models for the AI-powered Data Analyst application.

All request/response shapes and domain entities are defined here so that
other modules can import from a single, stable location.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Schema / dataset models
# ---------------------------------------------------------------------------


class ColumnSchema(BaseModel):
    """Metadata for a single column inferred from a CSV file."""

    name: str
    dtype: str  # "numeric" | "datetime" | "categorical"
    null_count: int


class DatasetRecord(BaseModel):
    """Lightweight record stored in the session; the DataFrame lives separately."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    filename: str
    row_count: int
    column_schemas: list[ColumnSchema]


# ---------------------------------------------------------------------------
# Conversation / session models
# ---------------------------------------------------------------------------


class Exchange(BaseModel):
    """A single turn in the conversation history."""

    role: str  # "user" | "assistant"
    content: str


class Session(BaseModel):
    """In-memory session state; never serialised to JSON directly."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    datasets: dict[str, DatasetRecord]  # filename → DatasetRecord
    dataframes: dict[str, pd.DataFrame]  # filename → DataFrame (runtime only)
    history: list[Exchange]


# ---------------------------------------------------------------------------
# Anomaly detection models
# ---------------------------------------------------------------------------


class AnomalyRecord(BaseModel):
    """A single flagged outlier row."""

    row_index: int
    column: str
    value: float
    z_score: Optional[float] = None
    explanation: str


class AnomalyReport(BaseModel):
    """Aggregated result of an anomaly detection run."""

    count: int
    flagged_rows: list[AnomalyRecord]
    message: Optional[str] = None  # set when detection could not run (e.g. no numeric columns)


# ---------------------------------------------------------------------------
# Query response models
# ---------------------------------------------------------------------------


class ReasoningTrace(BaseModel):
    """Audit trail of the tools and columns used to answer a query."""

    tools_selected: list[str]
    columns_used: list[str]
    computation_description: str


class QueryResponse(BaseModel):
    """Complete response returned by the query endpoint."""

    answer: str
    chart_base64: Optional[str] = None
    code_snippet: Optional[str] = None
    code_language: Optional[str] = None  # "python" | "sql"
    anomaly_report: Optional[AnomalyReport] = None
    reasoning_trace: ReasoningTrace  # non-optional — always present
    sampled: bool = False  # True when chart data was down-sampled


# ---------------------------------------------------------------------------
# API request / response shapes
# ---------------------------------------------------------------------------


class CreateSessionResponse(BaseModel):
    """Response body for POST /sessions."""

    session_id: str


class FileUploadResponse(BaseModel):
    """Per-file response body for POST /sessions/{id}/files."""

    filename: str
    row_count: int
    column_schemas: list[ColumnSchema]
    errors: list[str]


class QueryRequest(BaseModel):
    """Request body for POST /sessions/{id}/query."""

    query: str  # required — missing field raises pydantic.ValidationError
    dataset_filename: Optional[str] = None  # hint; Analyst may auto-select

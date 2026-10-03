# Architecture Diagram — AI-Powered Data Analyst

## System Overview

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                           BROWSER  (React + Vite)                        ║
║                                                                           ║
║  ┌─────────────────┐   ┌──────────────────────┐   ┌───────────────────┐  ║
║  │   FileUpload    │   │     ChatPanel        │   │  ResponseRenderer  │  ║
║  │                 │   │                      │   │                    │  ║
║  │  drag-and-drop  │   │  scrollable history  │   │  • text answer     │  ║
║  │  .csv only      │   │  textarea + submit   │   │  • base64 chart    │  ║
║  │  multi-file     │   │  Enter to send       │   │  • code block      │  ║
║  │  per-file       │   │  loading indicator   │   │  • anomaly table   │  ║
║  │  feedback       │   │                      │   │  • reasoning trace │  ║
║  └────────┬────────┘   └──────────┬───────────┘   └───────────────────┘  ║
║           │ multipart/form-data   │ JSON POST                             ║
╚═══════════╪═══════════════════════╪═══════════════════════════════════════╝
            │                       │
            ▼                       ▼
╔═══════════════════════════════════════════════════════════════════════════╗
║                         FASTAPI  APPLICATION                              ║
║                                                                           ║
║  ┌──────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  ║
║  │  Session Router  │  │   File Router        │  │   Query Router      │  ║
║  │                  │  │                      │  │                     │  ║
║  │ POST /sessions   │  │ POST …/{id}/files    │  │ POST …/{id}/query   │  ║
║  │ DELETE …/{id}    │  │  • ext validation    │  │  • deserialise      │  ║
║  │ GET /health      │  │  • validate_csv()    │  │  • await analyse()  │  ║
║  └──────────────────┘  │  • infer_schema()    │  │  • 404 / 503 / 422  │  ║
║                         │  • add_dataset()     │  └──────────┬──────────┘  ║
║  ┌──────────────────┐  └──────────────────────┘             │             ║
║  │  Sanitisation    │                                        ▼             ║
║  │  Middleware      │           ┌───────────────────────────────────────┐  ║
║  │                  │           │            Analyst                    │  ║
║  │ blocks ../  ..\  │           │                                       │  ║
║  │ blocks \x00      │           │  1. get_session()                    │  ║
║  │ JSON body scan   │           │  2. build_schema_context()  ◄── no   │  ║
║  └──────────────────┘           │     raw rows ever sent to LLM        │  ║
║                                 │  3. build_message_history()          │  ║
║                                 │  4. call_llm(messages, tool_schemas) │  ║
║                                 │  5. dispatch each tool_call          │  ║
║                                 │  6. assemble QueryResponse           │  ║
║                                 │  7. append_history()                 │  ║
║                                 └──────────────┬────────────────────────┘  ║
║                                                │                           ║
║                                                ▼                           ║
║                         ┌──────────────────────────────────────────────┐  ║
║                         │              Tool Registry                    │  ║
║                         │                                               │  ║
║  ┌──────────────────┐   │  dataset_profile  ──► Pandas .describe()     │  ║
║  │  Session Store   │   │  query_data       ──► filter + select rows   │  ║
║  │                  │   │  aggregate_data   ──► groupby + agg          │  ║
║  │  dict[UUID →     │◄──│  generate_summary ──► profile + insights     │  ║
║  │    Session]      │   │  generate_chart   ──► Matplotlib → PNG       │  ║
║  │                  │   │  detect_anomalies ──► Z-score / IQR          │  ║
║  │  • datasets      │   │  generate_sql     ──► SQL string template    │  ║
║  │  • dataframes    │   │  generate_pandas  ──► Pandas code template   │  ║
║  │  • history[≤20]  │   │                                               │  ║
║  └──────────────────┘   │  dispatch() validates name → UnknownToolError│  ║
║                         │  No eval / exec / subprocess anywhere        │  ║
║                         └──────────────────────────────────────────────┘  ║
║                                                │                           ║
║                                                ▼                           ║
║                              ┌─────────────────────────┐                  ║
║                              │      LLM Client         │                  ║
║                              │  openai.OpenAI(gpt-4o)  │                  ║
║                              │  tool_choice="auto"     │                  ║
║                              │  API key from env only  │                  ║
║                              └───────────┬─────────────┘                  ║
╚══════════════════════════════════════════╪════════════════════════════════╝
                                           │ HTTPS
                                           ▼
                              ┌─────────────────────────┐
                              │   OpenAI API (GPT-4o)   │
                              └─────────────────────────┘
```

## Component Responsibilities

| Component | File | Responsibility |
|---|---|---|
| Validator | `app/validator.py` | CSV parsing, size check, header validation |
| Type Inferrer | `app/type_inferrer.py` | Classify columns as numeric / datetime / categorical |
| Session Store | `app/session_store.py` | In-memory UUID-keyed session state, history cap |
| Context Builder | `app/context_builder.py` | Schema-only system prompt (no raw rows) |
| LLM Client | `app/llm_client.py` | OpenAI GPT-4o wrapper, error handling |
| Tool Registry | `app/tools/registry.py` | Tool registration, dispatch, JSON schemas |
| Analyst | `app/analyst.py` | Full query pipeline orchestration |
| Renderer | `app/renderer.py` | Matplotlib chart → base64 PNG |
| Anomaly Detector | `app/anomaly_detector.py` | Z-score / IQR outlier detection |
| Code Generator | `app/code_generator.py` | SQL + Pandas snippet templates + validation |

## Data Flow

```
User query
    │
    ▼
Sanitisation Middleware (blocks path traversal / null bytes)
    │
    ▼
Query Router → Analyst
    │
    ├── Session Store: fetch session
    ├── Context Builder: schema summary text (NO raw rows)
    ├── Session Store: conversation history (last 20)
    │
    ▼
LLM (GPT-4o) ──── tool_calls returned ────►  Tool Registry
                                                    │
                                         ┌──────────┴──────────┐
                                         │  Deterministic       │
                                         │  Pandas computation  │
                                         └──────────┬──────────┘
                                                    │
                                                    ▼
                                          QueryResponse assembled
                                          (answer + chart + code
                                           + anomaly + trace)
                                                    │
                                                    ▼
                                         Session history updated
                                                    │
                                                    ▼
                                         HTTP 200 → Frontend
```

## Security Properties

- **No code execution**: LLM output is never `eval()`d or `exec()`d. All tool calls go through the pre-defined registry.
- **No raw data to LLM**: Only schema summaries (column names, dtypes, null counts) are in the system prompt.
- **API key safety**: Loaded exclusively from environment variables; never logged or returned in responses.
- **Input sanitisation**: Path traversal (`../`, `..\`) and null bytes (`\x00`) rejected with HTTP 422.
- **Session isolation**: Each session is UUID-keyed; there is no cross-session data access.

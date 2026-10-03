"""
Application configuration module.

Reads environment variables at import time using pydantic-settings.
A module-level singleton `settings` is exported for use throughout the app.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    openai_api_key: str
    app_env: str = "development"
    max_upload_size_mb: int = 50
    max_history_exchanges: int = 20

    # LLM provider configuration.
    # Defaults preserve the original OpenAI GPT-4o behaviour.
    # Override via environment variables to use alternative providers:
    #   Groq:   LLM_BASE_URL=https://api.groq.com/openai/v1
    #           LLM_MODEL=llama-3.3-70b-versatile
    #   Gemini: LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
    #           LLM_MODEL=gemini-2.0-flash
    llm_base_url: str | None = None   # None → OpenAI default endpoint
    llm_model: str = "gpt-4o"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Variable names are matched case-insensitively by pydantic-settings
        case_sensitive = False
        extra = "ignore"  # silently ignore .env variables not declared as fields


# Module-level singleton — imported by other modules.
# Construction is deferred to main.py so a ValidationError can be caught
# and handled gracefully before the singleton is exposed to the rest of
# the application.  For modules that need settings after startup (i.e.,
# after main.py has already validated the environment), importing this
# module will succeed and `settings` will be fully populated.
settings = Settings()

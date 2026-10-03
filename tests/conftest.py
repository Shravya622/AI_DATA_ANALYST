"""
Root conftest.py — sets required environment variables before any app module
is imported, so pydantic-settings can construct the Settings singleton without
raising a ValidationError during test collection.

These are dummy values used only during testing. All tests that exercise the
LLM client mock the openai library so no real API calls are ever made.
"""

import os

# Set a placeholder key so `app.config.Settings()` validates successfully.
# Individual tests that need to test missing/invalid keys patch
# `app.llm_client.settings` directly using unittest.mock.patch.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder-for-unit-tests")

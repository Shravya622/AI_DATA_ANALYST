"""
Unit tests for app/llm_client.py.

All OpenAI API calls are mocked — no real network requests are made.
Tests cover the happy path, tool-call parsing, AuthenticationError → ConfigError
propagation, and generic APIError → structured LLMResponse handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import openai
import pytest

from app.exceptions import ConfigError
from app.llm_client import LLMResponse, ToolCall, call_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_completion(content: str | None = None, tool_calls: list | None = None) -> MagicMock:
    """Build a fake ``openai.ChatCompletion`` response object."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls or []

    choice = MagicMock()
    choice.message = message

    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _make_tool_call_obj(tc_id: str, name: str, arguments: str) -> MagicMock:
    """Build a fake OpenAI tool call object as returned by the SDK."""
    tc = MagicMock()
    tc.id = tc_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


# ---------------------------------------------------------------------------
# ToolCall dataclass
# ---------------------------------------------------------------------------

class TestToolCallDataclass:
    def test_fields_stored_correctly(self):
        tc = ToolCall(id="call_1", name="my_tool", arguments={"key": "value"})
        assert tc.id == "call_1"
        assert tc.name == "my_tool"
        assert tc.arguments == {"key": "value"}


# ---------------------------------------------------------------------------
# LLMResponse dataclass
# ---------------------------------------------------------------------------

class TestLLMResponseDataclass:
    def test_defaults(self):
        resp = LLMResponse(content="hello")
        assert resp.content == "hello"
        assert resp.tool_calls == []
        assert resp.error is None

    def test_error_field(self):
        resp = LLMResponse(content=None, tool_calls=[], error="something went wrong")
        assert resp.error == "something went wrong"


# ---------------------------------------------------------------------------
# call_llm — happy path (content response)
# ---------------------------------------------------------------------------

class TestCallLLMContentResponse:
    @patch("app.llm_client.openai.OpenAI")
    def test_returns_llm_response_with_content(self, mock_openai_cls):
        completion = _make_completion(content="Hello, world!")
        mock_openai_cls.return_value.chat.completions.create.return_value = completion

        result = call_llm(messages=[{"role": "user", "content": "Hi"}])

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello, world!"
        assert result.tool_calls == []
        assert result.error is None

    @patch("app.llm_client.openai.OpenAI")
    def test_content_none_when_tool_calls_present(self, mock_openai_cls):
        tc_obj = _make_tool_call_obj("call_1", "dataset_profile", '{"limit": 10}')
        completion = _make_completion(content=None, tool_calls=[tc_obj])
        mock_openai_cls.return_value.chat.completions.create.return_value = completion

        result = call_llm(
            messages=[{"role": "user", "content": "profile the data"}],
            tools=[{"type": "function", "function": {"name": "dataset_profile"}}],
        )

        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.error is None


# ---------------------------------------------------------------------------
# call_llm — tool call parsing
# ---------------------------------------------------------------------------

class TestCallLLMToolCallParsing:
    @patch("app.llm_client.openai.OpenAI")
    def test_single_tool_call_parsed(self, mock_openai_cls):
        tc_obj = _make_tool_call_obj("call_abc", "query_data", '{"columns": ["region"]}')
        completion = _make_completion(tool_calls=[tc_obj])
        mock_openai_cls.return_value.chat.completions.create.return_value = completion

        result = call_llm(
            messages=[{"role": "user", "content": "get data"}],
            tools=[{"type": "function", "function": {"name": "query_data"}}],
        )

        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.id == "call_abc"
        assert tc.name == "query_data"
        assert tc.arguments == {"columns": ["region"]}

    @patch("app.llm_client.openai.OpenAI")
    def test_multiple_tool_calls_parsed(self, mock_openai_cls):
        tc1 = _make_tool_call_obj("call_1", "tool_a", '{"x": 1}')
        tc2 = _make_tool_call_obj("call_2", "tool_b", '{"y": 2}')
        completion = _make_completion(tool_calls=[tc1, tc2])
        mock_openai_cls.return_value.chat.completions.create.return_value = completion

        result = call_llm(
            messages=[{"role": "user", "content": "do both"}],
            tools=[{"type": "function", "function": {"name": "tool_a"}},
                   {"type": "function", "function": {"name": "tool_b"}}],
        )

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "tool_a"
        assert result.tool_calls[1].name == "tool_b"

    @patch("app.llm_client.openai.OpenAI")
    def test_malformed_json_arguments_defaults_to_empty_dict(self, mock_openai_cls):
        tc_obj = _make_tool_call_obj("call_bad", "some_tool", "NOT_JSON")
        completion = _make_completion(tool_calls=[tc_obj])
        mock_openai_cls.return_value.chat.completions.create.return_value = completion

        result = call_llm(
            messages=[{"role": "user", "content": "go"}],
            tools=[{"type": "function", "function": {"name": "some_tool"}}],
        )

        assert result.tool_calls[0].arguments == {}

    @patch("app.llm_client.openai.OpenAI")
    def test_tool_choice_auto_when_tools_provided(self, mock_openai_cls):
        completion = _make_completion(content="ok")
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create.return_value = completion

        tools = [{"type": "function", "function": {"name": "my_tool"}}]
        call_llm(messages=[{"role": "user", "content": "test"}], tools=tools)

        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs.get("tool_choice") == "auto"
        assert kwargs.get("tools") == tools

    @patch("app.llm_client.openai.OpenAI")
    def test_tool_choice_none_when_no_tools(self, mock_openai_cls):
        completion = _make_completion(content="ok")
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create.return_value = completion

        call_llm(messages=[{"role": "user", "content": "test"}])

        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs.get("tool_choice") == "none"
        assert "tools" not in kwargs

    @patch("app.llm_client.openai.OpenAI")
    def test_model_is_configured_value(self, mock_openai_cls):
        """The model name sent to the API must match settings.llm_model (default: gpt-4o)."""
        completion = _make_completion(content="ok")
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create.return_value = completion

        call_llm(messages=[{"role": "user", "content": "test"}])

        _, kwargs = mock_client.chat.completions.create.call_args
        from app.config import settings
        assert kwargs.get("model") == settings.llm_model


# ---------------------------------------------------------------------------
# call_llm — AuthenticationError → ConfigError
# ---------------------------------------------------------------------------

class TestCallLLMAuthenticationError:
    @patch("app.llm_client.openai.OpenAI")
    def test_authentication_error_raises_config_error(self, mock_openai_cls):
        mock_openai_cls.return_value.chat.completions.create.side_effect = (
            openai.AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401, headers={}),
                body={"error": {"message": "Invalid API key"}},
            )
        )

        with pytest.raises(ConfigError) as exc_info:
            call_llm(messages=[{"role": "user", "content": "test"}])

        assert "Invalid OpenAI API key" in exc_info.value.message

    @patch("app.llm_client.openai.OpenAI")
    def test_authentication_error_message_does_not_contain_key(self, mock_openai_cls):
        """The ConfigError message must never embed the API key value."""
        mock_openai_cls.return_value.chat.completions.create.side_effect = (
            openai.AuthenticationError(
                message="Incorrect API key provided: sk-test",
                response=MagicMock(status_code=401, headers={}),
                body={"error": {"message": "Incorrect API key"}},
            )
        )

        with pytest.raises(ConfigError) as exc_info:
            call_llm(messages=[{"role": "user", "content": "test"}])

        # The key value should not appear in the raised exception message
        from app.config import settings
        assert settings.openai_api_key not in exc_info.value.message


# ---------------------------------------------------------------------------
# call_llm — APIError → structured error LLMResponse
# ---------------------------------------------------------------------------

class TestCallLLMAPIError:
    @patch("app.llm_client.openai.OpenAI")
    def test_api_error_returns_error_llm_response(self, mock_openai_cls):
        mock_openai_cls.return_value.chat.completions.create.side_effect = (
            openai.APIError(
                message="Service unavailable",
                request=MagicMock(),
                body=None,
            )
        )

        result = call_llm(messages=[{"role": "user", "content": "test"}])

        assert isinstance(result, LLMResponse)
        assert result.content is None
        assert result.tool_calls == []
        assert result.error is not None
        assert "LLM API error" in result.error

    @patch("app.llm_client.openai.OpenAI")
    def test_api_error_does_not_raise(self, mock_openai_cls):
        """APIError must be caught and NOT propagate to the caller."""
        mock_openai_cls.return_value.chat.completions.create.side_effect = (
            openai.APIError(
                message="Rate limit exceeded",
                request=MagicMock(),
                body=None,
            )
        )

        # Should not raise
        result = call_llm(messages=[{"role": "user", "content": "test"}])
        assert result.error is not None

    @patch("app.llm_client.openai.OpenAI")
    def test_api_error_message_does_not_contain_key(self, mock_openai_cls):
        """Error messages surfaced to callers must not embed the API key."""
        mock_openai_cls.return_value.chat.completions.create.side_effect = (
            openai.APIError(
                message="Bad request",
                request=MagicMock(),
                body=None,
            )
        )

        result = call_llm(messages=[{"role": "user", "content": "test"}])

        from app.config import settings
        assert settings.openai_api_key not in (result.error or "")


# ---------------------------------------------------------------------------
# call_llm — missing / empty API key
# ---------------------------------------------------------------------------

class TestCallLLMMissingAPIKey:
    @patch("app.llm_client.settings")
    def test_empty_api_key_raises_config_error(self, mock_settings):
        mock_settings.openai_api_key = ""

        with pytest.raises(ConfigError) as exc_info:
            call_llm(messages=[{"role": "user", "content": "test"}])

        assert "Invalid OpenAI API key" in exc_info.value.message

    @patch("app.llm_client.settings")
    def test_none_api_key_raises_config_error(self, mock_settings):
        mock_settings.openai_api_key = None  # type: ignore[assignment]

        with pytest.raises(ConfigError):
            call_llm(messages=[{"role": "user", "content": "test"}])

"""
LLM Client module for the AI-powered Data Analyst application.

Wraps the OpenAI Chat Completions API. Provides a single public function
``call_llm`` that handles authentication, error handling, and response
parsing without ever logging the API key value.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import openai

from app.config import settings
from app.exceptions import ConfigError

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a single tool call returned by the LLM."""

    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Structured response from the LLM."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    error: str | None = None


def call_llm(
    messages: list[dict],
    tools: list[dict] | None = None,
) -> LLMResponse:
    """Call the OpenAI Chat Completions API and return a structured response.

    Args:
        messages: A list of OpenAI message dicts (role/content pairs).
        tools: Optional list of OpenAI tool schema dicts. When provided,
               ``tool_choice`` is set to ``"auto"`` so the model can select
               any registered tool. When omitted/empty, tool calling is
               disabled via ``"none"``.

    Returns:
        An :class:`LLMResponse` with either ``content``, non-empty
        ``tool_calls``, or — on a non-fatal API error — an ``error`` string.

    Raises:
        ConfigError: If the API key is missing or the key is rejected by the
                     OpenAI API (``openai.AuthenticationError``).
    """
    api_key: str = settings.openai_api_key

    # Guard: empty/unset key — raised before any network call so no key value
    # is ever transmitted or logged.
    if not api_key:
        raise ConfigError("Invalid OpenAI API key")

    # Build the client.  When llm_base_url is None the OpenAI SDK uses its
    # default endpoint (api.openai.com).  Setting it to another value (e.g.
    # Groq or Gemini) redirects all requests to that provider while keeping
    # the identical OpenAI SDK interface.
    client_kwargs: dict = {"api_key": api_key}
    if settings.llm_base_url:
        client_kwargs["base_url"] = settings.llm_base_url

    client = openai.OpenAI(**client_kwargs)

    # Build call kwargs — only include tools/tool_choice when tools are given.
    call_kwargs: dict = {
        "model": settings.llm_model,
        "messages": messages,
    }

    if tools:
        call_kwargs["tools"] = tools
        call_kwargs["tool_choice"] = "auto"
    else:
        call_kwargs["tool_choice"] = "none"

    try:
        response = client.chat.completions.create(**call_kwargs)
    except openai.AuthenticationError:
        # Invalid key — this is a configuration problem; bubble up as ConfigError.
        # Do NOT include the key value in the message.
        logger.error("OpenAI authentication failed. Check OPENAI_API_KEY.")
        raise ConfigError("Invalid OpenAI API key")
    except openai.APIError as exc:
        # Any other API-level error (rate limit, service unavailable, etc.) is
        # surfaced as a structured error response so the application keeps running.
        logger.error("OpenAI API error: %s", exc)
        return LLMResponse(
            content=None,
            tool_calls=[],
            error=f"LLM API error: {exc}",
        )

    message = response.choices[0].message

    # Parse tool calls from the response, if any.
    parsed_tool_calls: list[ToolCall] = []
    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, ValueError):
                arguments = {}
            parsed_tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                )
            )

    return LLMResponse(
        content=message.content,
        tool_calls=parsed_tool_calls,
    )

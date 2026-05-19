from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

import requests
from dotenv import load_dotenv

from .config import LLMConfig


class MissingGroqApiKey(RuntimeError):
    """Raised when live Groq narration is requested without credentials."""


class GroqAPIError(RuntimeError):
    """Raised when Groq returns an error or an invalid response."""


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ChatCompletionResult:
    content: str
    model: str
    usage: dict[str, Any]
    raw_response: dict[str, Any]


class GroqChatClient:
    """Minimal production Groq Chat Completions client.

    Uses Groq's OpenAI-compatible `/chat/completions` endpoint with explicit
    retries for transient rate-limit/server failures. It does not fabricate
    responses or fall back to deterministic text.
    """

    def __init__(
        self,
        api_key: str,
        config: LLMConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise MissingGroqApiKey(
                "GROQ_API_KEY is required for Phase 4 live LLM narration"
            )
        self.api_key = api_key.strip()
        self.config = config or LLMConfig()
        self.session = session or requests.Session()

    @classmethod
    def from_env(cls, config: LLMConfig | None = None) -> "GroqChatClient":
        load_dotenv()
        return cls(api_key=os.getenv("GROQ_API_KEY", ""), config=config)

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        temperature: float | None = None,
    ) -> ChatCompletionResult:
        payload = {
            "model": self.config.model,
            "messages": [message.as_dict() for message in messages],
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.session.post(
                    self.config.chat_completions_url,
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout_seconds,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise GroqAPIError(
                        f"Groq transient error {response.status_code}: {response.text[:500]}"
                    )
                if response.status_code >= 400:
                    raise GroqAPIError(
                        f"Groq request failed {response.status_code}: {response.text[:1000]}"
                    )
                data = response.json()
                content = _extract_message_content(data)
                result = ChatCompletionResult(
                    content=content,
                    model=str(data.get("model", self.config.model)),
                    usage=data.get("usage", {}),
                    raw_response=data,
                )
                logger.info(
                    "Groq completion: model=%s, prompt_tokens=%s, completion_tokens=%s",
                    result.model,
                    result.usage.get("prompt_tokens", "?"),
                    result.usage.get("completion_tokens", "?"),
                )
                return result
            except (requests.RequestException, GroqAPIError, ValueError) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                time.sleep(self.config.retry_backoff_seconds * (attempt + 1))

        raise GroqAPIError(f"Groq completion failed after retries: {last_error}")


def _extract_message_content(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GroqAPIError(f"Groq response did not contain message content: {data}") from exc
    if not isinstance(content, str) or not content.strip():
        raise GroqAPIError("Groq response content was empty")
    return content.strip()

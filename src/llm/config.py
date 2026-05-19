from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    base_url: str = "https://api.groq.com/openai/v1"
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    rate_limit_sleep_seconds: float = 2.0
    risk_narrative_max_tokens: int = 520
    executive_brief_max_tokens: int = 420
    temperature: float = 0.2

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

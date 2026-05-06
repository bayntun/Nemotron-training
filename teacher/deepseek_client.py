"""
DeepSeek V3.2 chat-completions client for Phase 2 teacher CoT generation.

Design goals:
- Async-first via httpx for high-throughput batched generation.
- Tenacity-based retry with exponential backoff on transient failures.
- Off-peak-aware: simply caller's responsibility -- DeepSeek's 50% off
  discount window is 16:30-00:30 UTC.
- Compatible with OpenRouter as a drop-in fallback (set DEEPSEEK_BASE_URL).

Endpoint reference:
- Base URL: https://api.deepseek.com (OpenAI-compatible /v1/chat/completions)
- Models: deepseek-chat (V3.2-Exp, non-thinking by default)
          deepseek-reasoner (R1-style chain-of-thought)
- Auth: Bearer token via Authorization header.

For Phase 2:
- Use deepseek-chat for solver-guided CoT (rule prefixed in user message).
- Reserve deepseek-reasoner for the hardest residual puzzles (slower, costlier).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


@dataclass
class DeepSeekResponse:
    content: str
    reasoning: str | None  # populated for deepseek-reasoner; None for deepseek-chat
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    model: str


class DeepSeekClient:
    """Thin async OpenAI-compatible client for DeepSeek V3.2."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 5,
    ):
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Add it to .env or export it before running."
            )
        base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout_s,
        )
        self._max_retries = max_retries

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def chat(
        self,
        messages: list[dict],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
    ) -> DeepSeekResponse:
        """
        Single chat completion. `messages` follows OpenAI format:
            [{"role": "system" | "user" | "assistant", "content": "..."}]
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False,
        }

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=2, min=2, max=60),
            retry=retry_if_exception_type(
                (httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError)
            ),
            reraise=True,
        ):
            with attempt:
                resp = await self._client.post("/v1/chat/completions", json=payload)
                # Retry on 429 / 5xx; surface 4xx (auth, etc.) immediately.
                if resp.status_code == 429 or resp.status_code >= 500:
                    resp.raise_for_status()
                resp.raise_for_status()
                data = resp.json()

        choice = data["choices"][0]
        message = choice.get("message", {})
        return DeepSeekResponse(
            content=message.get("content", "") or "",
            reasoning=message.get("reasoning_content"),
            prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", ""),
            model=data.get("model", model),
        )

"""OpenAI-compatible LLM backend.

Works with any OpenAI-compatible `/v1/chat/completions` endpoint:

    - Real OpenAI          (base_url: https://api.openai.com/v1)
    - OpenRouter           (base_url: https://openrouter.ai/api/v1)
    - DeepSeek             (base_url: https://api.deepseek.com/v1)
    - Azure OpenAI (proxy) / local LLM servers, etc.

Uses ``httpx`` directly so no extra SDK dependency is required.
"""
from __future__ import annotations

import httpx
import structlog

from lit_inspector.core.config import LLMSettings
from lit_inspector.core.exceptions import LLMError
from lit_inspector.llm.base import LLMBackend

logger = structlog.get_logger(__name__)

_DEFAULT_BASE = "https://api.openai.com/v1"


class OpenAIBackend(LLMBackend):
    """LLM backend for any OpenAI-compatible endpoint.

    Args:
        settings: LLM configuration (api_key, model, temperature,
            base_url, etc.). ``base_url`` is optional and defaults
            to OpenAI's official URL; set it to point at OpenRouter,
            DeepSeek, Azure, local servers, etc.
    """

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        if not settings.api_key:
            raise LLMError("OpenAI backend requires an api_key in config.")
        self._base_url = (settings.base_url or _DEFAULT_BASE).rstrip("/")
        self._model = settings.model or "gpt-4o-mini"

    @property
    def model_id(self) -> str:
        return self._model

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        """Send a prompt and return the response text.

        Endpoint: POST ``{base_url}/chat/completions``
        """
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
            "seed": seed,
        }

        logger.info(
            "openai_request",
            model=self._model,
            base_url=self._base_url,
            prompt_len=len(prompt),
        )

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=30.0)
            ) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()

            choices = data.get("choices", [])
            if not choices:
                raise LLMError(f"OpenAI returned no choices: {data}")

            text = choices[0].get("message", {}).get("content", "")
            finish_reason = choices[0].get("finish_reason", "")
            logger.info(
                "openai_response",
                response_len=len(text),
                finish_reason=finish_reason,
            )
            if finish_reason == "length":
                logger.warning(
                    "openai_truncated",
                    msg="Response hit max_tokens — raise max_tokens in config.",
                )
            return text

        except httpx.HTTPStatusError as exc:
            error_body = exc.response.text[:500]
            raise LLMError(
                f"OpenAI API error (HTTP {exc.response.status_code}): {error_body}"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError(f"OpenAI network error: {exc}") from exc

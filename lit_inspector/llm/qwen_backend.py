"""Qwen (通义千问) LLM backend — connects via DashScope OpenAI-compatible API.

Uses the OpenAI-compatible endpoint provided by Alibaba Cloud DashScope.
Docs: https://help.aliyun.com/zh/model-studio/developer-reference/compatibility-of-openai-with-dashscope
"""
from __future__ import annotations

import httpx
import structlog

from lit_inspector.core.config import LLMSettings
from lit_inspector.core.exceptions import LLMError
from lit_inspector.llm.base import LLMBackend

logger = structlog.get_logger(__name__)

_DASHSCOPE_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class QwenBackend(LLMBackend):
    """LLM backend using the Qwen / DashScope OpenAI-compatible API.

    Args:
        settings: LLM configuration (api_key, model, temperature, etc.).
    """

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        if not settings.api_key:
            raise LLMError("Qwen backend requires an api_key in config.")
        configured_base = (settings.base_url or "").rstrip("/")
        if configured_base and configured_base != _DASHSCOPE_BASE:
            logger.warning(
                "qwen_base_url_overridden",
                configured_base=configured_base,
                forced_base_url=_DASHSCOPE_BASE,
            )
        self._base_url = _DASHSCOPE_BASE
        self._model = settings.model or "qwen-plus"

    @property
    def model_id(self) -> str:
        return self._model

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        """Send a prompt to Qwen via the OpenAI-compatible endpoint.

        Endpoint: POST /v1/chat/completions
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
            "qwen_request",
            model=self._model,
            base_url=self._base_url,
            prompt_len=len(prompt),
        )

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()

            # OpenAI-compatible response format
            choices = data.get("choices", [])
            if not choices:
                raise LLMError(f"Qwen returned no choices: {data}")

            text = choices[0].get("message", {}).get("content", "")
            logger.info("qwen_response", response_len=len(text))
            return text

        except httpx.HTTPStatusError as exc:
            error_body = exc.response.text[:500]
            raise LLMError(
                f"Qwen API error (HTTP {exc.response.status_code}): {error_body}"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError(f"Qwen network error: {exc}") from exc

"""Google Gemini LLM backend — connects via REST API (no SDK needed).

Uses the generativelanguage.googleapis.com endpoint directly with httpx.
Docs: https://ai.google.dev/gemini-api/docs/text-generation
"""
from __future__ import annotations

import httpx
import structlog

from lit_inspector.core.config import LLMSettings
from lit_inspector.core.exceptions import LLMError
from lit_inspector.llm.base import LLMBackend

logger = structlog.get_logger(__name__)

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiBackend(LLMBackend):
    """LLM backend using the Google Gemini REST API.

    Args:
        settings: LLM configuration (api_key, model, temperature, etc.).
    """

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        if not settings.api_key:
            raise LLMError("Gemini backend requires an api_key in config.")
        self._model = settings.model or "gemini-2.5-flash"

    @property
    def model_id(self) -> str:
        return self._model

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        """Send a prompt to Gemini and return the response text.

        Endpoint: POST /v1beta/models/{model}:generateContent?key={key}
        """
        url = f"{_GEMINI_BASE}/models/{self._model}:generateContent"
        params = {"key": self._settings.api_key}

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": self._settings.temperature,
                "maxOutputTokens": self._settings.max_tokens,
            },
        }

        logger.info(
            "gemini_request",
            model=self._model,
            prompt_len=len(prompt),
        )

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
                resp = await client.post(url, params=params, json=payload)
                resp.raise_for_status()
                data = resp.json()

            # Extract text from Gemini response
            candidates = data.get("candidates", [])
            if not candidates:
                raise LLMError(f"Gemini returned no candidates: {data}")

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise LLMError(f"Gemini candidate has no parts: {candidates[0]}")

            text = parts[0].get("text", "")
            logger.info("gemini_response", response_len=len(text))
            return text

        except httpx.HTTPStatusError as exc:
            error_body = exc.response.text[:500]
            raise LLMError(
                f"Gemini API error (HTTP {exc.response.status_code}): {error_body}"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError(f"Gemini network error: {exc}") from exc

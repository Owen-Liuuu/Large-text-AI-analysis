"""OpenAI LLM backend — connects to GPT-4o / GPT-4-turbo / etc.

TODO: Fill in the actual API call implementation.
"""
from __future__ import annotations

from lit_inspector.core.config import LLMSettings
from lit_inspector.llm.base import LLMBackend


class OpenAIBackend(LLMBackend):
    """LLM backend using the OpenAI API.

    Args:
        settings: LLM configuration (api_key, model, temperature, etc.).
    """

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        # TODO: Initialise the OpenAI client here
        # Example:
        #   from openai import AsyncOpenAI
        #   self._client = AsyncOpenAI(api_key=settings.api_key)

    @property
    def model_id(self) -> str:
        return self._settings.model

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        """Send a prompt to OpenAI and return the response text.

        TODO: Implement the actual API call. Example:

            response = await self._client.chat.completions.create(
                model=self._settings.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
                seed=seed,
            )
            return response.choices[0].message.content or ""
        """
        raise NotImplementedError(
            "OpenAI backend not yet implemented. "
            "Fill in the complete() method with your OpenAI API call."
        )

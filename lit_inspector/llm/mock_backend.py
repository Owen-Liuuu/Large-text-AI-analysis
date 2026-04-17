"""Mock LLM backend for testing without API keys."""
from __future__ import annotations

import json

from lit_inspector.llm.base import LLMBackend


class MockLLMBackend(LLMBackend):
    """Returns pre-configured JSON responses for testing.

    Args:
        responses: Optional mapping of prompt substrings to response dicts.
                   If no match is found, returns a generic acknowledgement.
    """

    def __init__(self, responses: dict[str, dict] | None = None) -> None:
        self._responses = responses or {}

    @property
    def model_id(self) -> str:
        return "mock-llm-v1"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        # Check if any registered key appears in the prompt
        for key, response in self._responses.items():
            if key in prompt:
                return json.dumps(response)

        # Default response
        return json.dumps({"status": "ok", "message": "mock response"})

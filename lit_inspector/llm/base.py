"""LLM backend abstract base class and response parsing utilities.

Pattern adapted from metascreener.llm.base (小组项目/SRC/).
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from lit_inspector.core.exceptions import LLMError


class LLMBackend(ABC):
    """Abstract interface for any LLM provider.

    Subclass this to integrate OpenAI, Anthropic, local models, etc.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the identifier of the underlying model."""

    @abstractmethod
    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        """Send a prompt and return the raw text response.

        Args:
            prompt: The full prompt string.
            seed: Reproducibility seed (provider-dependent).

        Returns:
            Raw text response from the model.
        """


def parse_llm_response(raw: str, model_id: str) -> dict:
    """Extract a JSON object from a raw LLM response.

    Handles common cases where the model wraps JSON in markdown
    code fences or adds extra text around it.

    Args:
        raw: Raw text response from the LLM.
        model_id: Model identifier (for error messages).

    Returns:
        Parsed dictionary.

    Raises:
        LLMError: If no valid JSON can be extracted.
    """
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise LLMError(
        f"Failed to parse JSON from {model_id} response. "
        f"Raw output (first 200 chars): {raw[:200]}"
    )

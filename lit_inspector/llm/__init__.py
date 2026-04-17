"""LLM abstraction layer: backend ABC and utilities."""

from lit_inspector.llm.base import LLMBackend, parse_llm_response

__all__ = ["LLMBackend", "parse_llm_response"]

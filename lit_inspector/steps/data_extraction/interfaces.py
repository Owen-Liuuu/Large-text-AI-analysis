"""Abstract interface for step 3: data extraction."""
from __future__ import annotations

from abc import ABC, abstractmethod

from lit_inspector.steps.data_extraction.schemas import ExtractedTable, PaperDocument


class Extractor(ABC):
    """Interface for extracting structured data from a paper.

    Multiple extractor implementations (e.g. using different LLMs)
    can be run in parallel for cross-validation.
    """

    @property
    @abstractmethod
    def extractor_id(self) -> str:
        """Unique identifier for this extractor."""

    @abstractmethod
    async def extract(
        self,
        document: PaperDocument,
        fields: list[str],
        *,
        research_context: str = "",
    ) -> ExtractedTable:
        """Extract specified fields from a paper document.

        Args:
            document: The paper to extract from.
            fields: List of field names to extract (dynamically identified).
            research_context: One-sentence description of the review's topic,
                e.g. "EAT thickness/volume in T1DM vs controls". Helps the
                LLM understand what data to look for.

        Returns:
            Extraction results as a structured table.
        """

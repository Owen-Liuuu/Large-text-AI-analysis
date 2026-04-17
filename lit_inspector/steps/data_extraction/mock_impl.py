"""Mock implementations for step 3: data extraction."""
from __future__ import annotations

from lit_inspector.steps.data_extraction.interfaces import Extractor
from lit_inspector.steps.data_extraction.schemas import (
    ExtractedField,
    ExtractedTable,
    PaperDocument,
)


class MockExtractorA(Extractor):
    """Mock extractor A: returns deterministic fake data."""

    @property
    def extractor_id(self) -> str:
        return "mock-extractor-a"

    async def extract(
        self,
        document: PaperDocument,
        fields: list[str],
        *,
        research_context: str = "",
    ) -> ExtractedTable:
        extracted_fields = []
        for field_name in fields:
            extracted_fields.append(
                ExtractedField(
                    field_name=field_name,
                    value=f"value-a-{field_name}",
                    evidence=f"Mock evidence A for {field_name} from {document.paper_id}.",
                    confidence=0.90,
                )
            )
        return ExtractedTable(
            paper_id=document.paper_id,
            fields=extracted_fields,
            extractor_id=self.extractor_id,
        )


class MockExtractorB(Extractor):
    """Mock extractor B: returns slightly different fake data for cross-validation."""

    @property
    def extractor_id(self) -> str:
        return "mock-extractor-b"

    async def extract(
        self,
        document: PaperDocument,
        fields: list[str],
        *,
        research_context: str = "",
    ) -> ExtractedTable:
        extracted_fields = []
        for field_name in fields:
            extracted_fields.append(
                ExtractedField(
                    field_name=field_name,
                    value=f"value-b-{field_name}",
                    evidence=f"Mock evidence B for {field_name} from {document.paper_id}.",
                    confidence=0.85,
                )
            )
        return ExtractedTable(
            paper_id=document.paper_id,
            fields=extracted_fields,
            extractor_id=self.extractor_id,
        )

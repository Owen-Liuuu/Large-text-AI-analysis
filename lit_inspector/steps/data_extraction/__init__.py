"""Step 3: Data extraction and table generation."""

from lit_inspector.steps.data_extraction.interfaces import Extractor
from lit_inspector.steps.data_extraction.schemas import (
    ExtractedField,
    ExtractedTable,
    PaperDocument,
)

__all__ = ["Extractor", "ExtractedField", "ExtractedTable", "PaperDocument"]

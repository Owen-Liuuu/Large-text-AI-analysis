"""PDF parser: extracts StudentReviewInput from a systematic review PDF.

Flow:
  1. Open PDF with PyMuPDF (fitz) and extract plain text from all pages
  2. Send the text (truncated) to an LLM with an extraction prompt
  3. Parse the LLM's JSON response into a StudentReviewInput object

This replaces the need to hand-write YAML input files for each submission.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import structlog

from lit_inspector.llm.base import LLMBackend, parse_llm_response
from lit_inspector.pipeline.schemas import StudentReviewInput
from lit_inspector.steps.data_extraction.schemas import ExtractedField, ExtractedTable
from lit_inspector.steps.paper_verification.schemas import ReferenceEntry

logger = structlog.get_logger(__name__)


_PARSING_PROMPT = """You are an expert assistant reading a systematic review / meta-analysis paper.
Your job is to extract three things from the paper text below:

1. The SEARCH STRATEGY the authors used:
   - search_database: which database(s) (e.g. "PubMed", "PubMed, Embase, Cochrane")
   - search_strategy_text: the actual query / keywords (e.g. "epicardial adipose tissue AND type 1 diabetes")
   - reported_result_count: total number of records identified in initial search (integer, or null if unclear)

2. The SELECTED PAPERS (the final set of studies INCLUDED in the review, usually in a
   "Characteristics of included studies" table or described in Results). For each one:
   - title: paper title
   - authors: list of author last-name + initials (first 3-4 authors is fine)
   - journal: journal name (or "" if unknown)
   - year: publication year (integer)
   - doi: DOI if available (e.g. "10.1007/s00246-021-02811-x"), or ""

3. The DATA EXTRACTION TABLE (usually called "Table 1" or "Characteristics of included studies").
   This is structured data the authors extracted from each included study.
   - extraction_fields: the list of COLUMN names in the table (e.g. ["country", "sample_size", "age_mean", "bmi", "measurement_tool"])
   - submitted_tables: one entry per included paper, containing the values the authors extracted for each column

## PAPER TEXT
{paper_text}

## OUTPUT FORMAT
Return ONLY a valid JSON object with this exact structure (no extra text):

```json
{{
  "student_id": "auto-extracted",
  "review_title": "the title of this review paper",
  "search_strategy_text": "the search query used",
  "search_database": "database name(s)",
  "reported_result_count": 150,
  "selected_papers": [
    {{
      "title": "Paper title",
      "authors": ["Lastname A", "Lastname B"],
      "journal": "Journal Name",
      "year": 2023,
      "doi": "10.xxxx/xxxxx"
    }}
  ],
  "extraction_fields": ["field1", "field2"],
  "submitted_tables": [
    {{
      "paper_id": "10.xxxx/xxxxx",
      "extractor_id": "student",
      "fields": [
        {{"field_name": "field1", "value": "...", "confidence": 1.0}},
        {{"field_name": "field2", "value": 100, "confidence": 1.0}}
      ]
    }}
  ]
}}
```

Rules:
- Extract EXACTLY what is in the paper. Do NOT invent data.
- For DOI: extract from the References section if present.
- For `value`: use numbers when the cell is numeric, strings otherwise.
- If a field is missing for a paper, use null.
- paper_id in submitted_tables MUST match the doi of the corresponding selected_paper.
- Return VALID JSON only, no commentary."""


class PDFParser:
    """Parses a systematic review PDF into a StudentReviewInput.

    Uses PyMuPDF for text extraction and an LLM backend to identify the
    search strategy, included studies, and data extraction table.

    Args:
        llm_backend: The LLM backend to use for structured extraction.
        max_chars: Maximum characters of paper text to send to the LLM
            (defaults to 30000 — most reviews fit within this).
    """

    def __init__(self, llm_backend: LLMBackend, max_chars: int = 50000) -> None:
        self._backend = llm_backend
        self._max_chars = max_chars

    async def parse(self, pdf_path: Path) -> StudentReviewInput:
        """Parse a PDF and return a populated StudentReviewInput.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            A StudentReviewInput ready to feed into the pipeline.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            ImportError: If PyMuPDF is not installed.
            LLMError: If the LLM response cannot be parsed as JSON.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("pdf_parse_start", path=str(pdf_path))

        # Step A: Extract plain text from the PDF
        paper_text = self._extract_text(pdf_path)
        logger.info(
            "pdf_text_extracted",
            chars=len(paper_text),
            truncated=len(paper_text) > self._max_chars,
        )

        # Step A2: Regex-extract all real DOIs from the full text.
        # These form a whitelist used to catch / replace any DOIs the LLM
        # may hallucinate in the next step.
        real_dois = self._extract_dois(paper_text)
        logger.info("pdf_dois_extracted", n_dois=len(real_dois))

        # Step B: Send to LLM for structured extraction
        prompt = _PARSING_PROMPT.format(paper_text=paper_text[: self._max_chars])
        raw_response = await self._backend.complete(prompt)
        parsed = parse_llm_response(raw_response, self._backend.model_id)

        # Step C: Build the StudentReviewInput (with DOI validation)
        student_input = self._build_input(
            parsed, pdf_path, real_dois=real_dois,
            review_full_text=paper_text,
        )
        logger.info(
            "pdf_parse_done",
            n_selected=len(student_input.selected_papers),
            n_tables=len(student_input.submitted_tables),
            n_fields=len(student_input.extraction_fields),
        )
        return student_input

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # Matches DOIs like "10.1007/s00246-021-02811-x" or "10.1016/S0140-6736(18)31320-5".
    # Allows parentheses inside the suffix (some publishers embed them).
    _DOI_REGEX = re.compile(r"\b10\.\d{4,9}/[^\s\"<>\]]+", re.IGNORECASE)

    @classmethod
    def _extract_dois(cls, text: str) -> list[str]:
        """Extract unique real DOIs from the paper text using a regex.

        PyMuPDF often inserts line breaks, soft hyphens, and zero-width spaces
        in the middle of DOIs (e.g. "10.1016/j.numecd.\\n2013.11.001"), so we
        normalise the text first by:
          1. Removing soft hyphens and zero-width chars
          2. Removing line breaks (and surrounding whitespace) so DOIs that
             were split across two lines reassemble correctly
        """
        # Strip hidden characters PDFs often insert mid-token
        cleaned = text.replace("\u00ad", "").replace("\u200b", "").replace("\u200c", "")

        # Rejoin lines that were split mid-token: remove a bare \n that sits
        # between two non-whitespace characters (e.g. "10.1038/\ns41598" →
        # "10.1038/s41598"), but keep \n that has surrounding whitespace
        # (e.g. "9478-x\n\t21." stays separate → word boundary).
        normalized = re.sub(r"(?<=\S)\n(?=\S)", "", cleaned)

        raw_matches = cls._DOI_REGEX.findall(normalized)
        dois: list[str] = []
        seen: set[str] = set()
        for m in raw_matches:
            doi = m.rstrip(".,);:]")
            key = doi.lower()
            if key not in seen:
                seen.add(key)
                dois.append(doi)
        return dois

    @staticmethod
    def _resolve_doi(
        llm_doi: str,
        llm_title: str,
        real_dois: list[str],
    ) -> str:
        """Validate / correct an LLM-provided DOI against the real DOI whitelist.

        Strategy:
          1. If the LLM's DOI is already in the whitelist, accept it.
          2. Otherwise, try to match by title keywords: look at the DOI's
             suffix and path — if no match, try matching title words against
             the DOI's slug. This is best-effort.
          3. If nothing matches, return "" (leave it empty rather than keep
             a hallucinated DOI).
        """
        if not real_dois:
            return llm_doi  # nothing to validate against
        whitelist_lower = {d.lower(): d for d in real_dois}
        if llm_doi and llm_doi.lower() in whitelist_lower:
            return whitelist_lower[llm_doi.lower()]
        # LLM DOI is not real — drop it. We don't attempt fuzzy slug matching
        # because that risks introducing new wrong DOIs. Downstream CrossRef
        # verification can still find the paper by title search.
        return ""

    @staticmethod
    def _extract_text(pdf_path: Path) -> str:
        """Extract plain text from all pages of a PDF using PyMuPDF."""
        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is required for PDF parsing. Install with: pip install pymupdf"
            ) from exc

        doc = fitz.open(str(pdf_path))
        pages_text: list[str] = []
        try:
            for i in range(len(doc)):
                pages_text.append(doc[i].get_text())
        finally:
            doc.close()

        return "\n\n".join(pages_text)

    @classmethod
    def _build_input(
        cls,
        parsed: dict[str, Any],
        pdf_path: Path,
        real_dois: list[str] | None = None,
        review_full_text: str = "",
    ) -> StudentReviewInput:
        """Convert the parsed LLM JSON into a StudentReviewInput.

        This handles missing fields, type coercion, and normalizes the
        submitted_tables into ExtractedTable/ExtractedField objects.

        If ``real_dois`` is provided (extracted via regex from the PDF text),
        any LLM-generated DOI that is NOT in that whitelist is dropped —
        preventing hallucinated DOIs from polluting the pipeline.
        """
        real_dois = real_dois or []

        # Track title -> validated DOI so we can later rewrite submitted_tables
        # whose paper_id used the old (hallucinated) DOI.
        doi_rewrite: dict[str, str] = {}

        # Selected papers
        selected_papers: list[ReferenceEntry] = []
        for p in parsed.get("selected_papers", []) or []:
            llm_doi = p.get("doi") or ""
            title = p.get("title") or ""
            validated_doi = cls._resolve_doi(llm_doi, title, real_dois)
            if llm_doi and llm_doi != validated_doi:
                logger.warning(
                    "pdf_doi_dropped",
                    title=title[:60],
                    llm_doi=llm_doi,
                    reason="not found in PDF text — likely LLM hallucination",
                )
                doi_rewrite[llm_doi] = validated_doi
            selected_papers.append(
                ReferenceEntry(
                    title=title,
                    authors=p.get("authors") or [],
                    journal=p.get("journal") or "",
                    year=p.get("year"),
                    doi=validated_doi,
                )
            )

        # Submitted tables
        submitted_tables: list[ExtractedTable] = []
        for t in parsed.get("submitted_tables", []) or []:
            fields: list[ExtractedField] = []
            for f in t.get("fields", []) or []:
                fields.append(
                    ExtractedField(
                        field_name=f.get("field_name") or "unknown",
                        value=f.get("value"),
                        evidence=f.get("evidence") or "",
                        confidence=float(f.get("confidence") or 1.0),
                    )
                )
            paper_id = t.get("paper_id") or ""
            # If the LLM used a hallucinated DOI as paper_id, rewrite it
            # to the validated one (or "") so the Step 4 comparison still
            # lines up with selected_papers.
            if paper_id in doi_rewrite:
                paper_id = doi_rewrite[paper_id]
            submitted_tables.append(
                ExtractedTable(
                    paper_id=paper_id,
                    fields=fields,
                    extractor_id=t.get("extractor_id") or "student",
                )
            )

        # Student ID defaults to the PDF filename if not provided
        student_id = parsed.get("student_id") or pdf_path.stem

        return StudentReviewInput(
            student_id=student_id,
            review_title=parsed.get("review_title") or pdf_path.stem,
            search_strategy_text=parsed.get("search_strategy_text") or "",
            search_database=parsed.get("search_database") or "PubMed",
            reported_result_count=parsed.get("reported_result_count"),
            selected_papers=selected_papers,
            extraction_fields=parsed.get("extraction_fields") or [],
            submitted_tables=submitted_tables,
            review_full_text=review_full_text,
        )

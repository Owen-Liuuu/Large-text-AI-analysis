"""Real implementation for step 3: LLM-based data extraction.

Uses an injected LLMBackend to extract structured fields from paper text.
Works with any LLM provider (OpenAI, Anthropic, etc.) — the specific
provider is determined by which backend is injected at construction.

Key design: the fields to extract are DYNAMIC — they are identified by
the LLM in Step 1 based on the review's actual topic, not hardcoded.
"""
from __future__ import annotations

import structlog

from lit_inspector.llm.base import LLMBackend, parse_llm_response
from lit_inspector.steps.data_extraction.interfaces import Extractor
from lit_inspector.steps.data_extraction.schemas import (
    ExtractedField,
    ExtractedTable,
    PaperDocument,
)

logger = structlog.get_logger(__name__)

_EXTRACTION_PROMPT_TEMPLATE = """You are an expert data extractor for systematic reviews in medical research.

## RESEARCH CONTEXT
{research_context}

## TASK
Extract the following fields from the provided paper text.
For each field, return the extracted value and a brief supporting quote from the text.
If a field cannot be found, set its value to null.

## FIELDS TO EXTRACT
{field_list}

## INTERPRETING FIELD NAMES
The field names above may be raw column headers from a student's data
extraction table. Interpret them flexibly using domain knowledge of the
research context above. Examples (general — apply the same reasoning
to ANY abbreviation, with or without slashes / units):

  "Age"             → mean age of participants (years), e.g. "12.90 +/- 1.30"
  "BMI" / "BMI Kg/m2" → body mass index (kg/m^2), e.g. "20.57 +/- 1.7"
  "N"               → sample size, total number of participants
  "EFT" / "EAT" / "EFT/ EAT" → epicardial fat thickness OR epicardial
                      adipose tissue thickness/volume (mm or cm^3),
                      whichever the paper reports
  "HR"              → hazard ratio (with 95% CI if available)
  "OS"              → overall survival (median months or HR)
  "PFS"             → progression-free survival
  "p" / "P-value"   → statistical significance value
  "Group"           → study group / cohort description (e.g. "T1DM patients
                      and healthy controls")
  "Quality"         → quality assessment score (NOS, Cochrane RoB, etc.)

For values reported as "mean +/- SD" or "median (IQR)", return the raw
string verbatim (e.g. "12.90 +/- 1.30"), do NOT split it into separate
numbers — the comparator handles the parsing.

Search the ENTIRE paper text below — including Abstract, Methods,
Results, AND any embedded table rows — for the requested values.
Tables in the text may contain numeric values that the prose does not.

## PAPER TEXT
{paper_text}

## OUTPUT FORMAT
Return a JSON object with exactly this structure:

```json
{{
  "fields": [
    {{
      "field_name": "name_of_field",
      "value": "extracted value or null",
      "evidence": "brief quote from the text supporting this value"
    }}
  ]
}}
```

Rules:
- Extract values ONLY from the provided text. Do NOT infer or hallucinate.
- A valid value is a NUMBER, a "mean +/- SD" string, a phrase, or a
  short description. The following are NOT valid values — return null
  instead of any of these:
    * The field name itself (e.g. value="Age" for field "Age")
    * An abbreviation or alias of the field name (e.g. value="EFT" for
      field "EFT/ EAT" — that's just echoing the term, not extracting a value)
    * Placeholder strings: "N/A", "n/a", "NA", "not reported", "not
      available", "see table", "not mentioned", "-", "--", "N.R.", "NR"
    * Empty string ""
  In ALL the above cases, the correct answer is value=null.
- For "mean +/- SD" style values, return the full string verbatim
  (e.g. "12.90 +/- 1.30"). The string MUST contain at least one digit.
- For pure numeric values (sample size, year), return a number.
- For evidence, quote the relevant sentence or table row that contains
  the actual value. If no such sentence exists, value MUST be null.
- Return the field_name EXACTLY as listed in "FIELDS TO EXTRACT" — do
  not rename, abbreviate, or normalise them.
- Return valid JSON only, no additional text."""


class LLMExtractor(Extractor):
    """Extracts structured data from papers using an LLM backend.

    Args:
        backend: The LLM backend to use (OpenAI, Claude, etc.).
        extractor_name: A unique name for this extractor instance
                        (e.g. "gpt4o-extractor", "claude-extractor").
    """

    def __init__(self, backend: LLMBackend, extractor_name: str) -> None:
        self._backend = backend
        self._extractor_name = extractor_name

    @property
    def extractor_id(self) -> str:
        return self._extractor_name

    async def extract(
        self,
        document: PaperDocument,
        fields: list[str],
        *,
        research_context: str = "",
    ) -> ExtractedTable:
        """Extract fields from a paper document using the LLM.

        Args:
            document: The paper to extract from.
            fields: List of field names to extract (dynamically identified).
            research_context: Topic of the review (e.g. "EAT in T1DM").

        Returns:
            Extraction results as a structured table.
        """
        logger.info(
            "llm_extraction_start",
            paper_id=document.paper_id,
            extractor=self._extractor_name,
            n_fields=len(fields),
        )

        # Build the prompt — select the most relevant text sections
        field_list = "\n".join(f"- {f}" for f in fields)
        paper_text = self._select_relevant_text(document, max_chars=25000)
        context_line = (
            research_context
            if research_context
            else "Systematic review — extract data relevant to the review's topic."
        )

        prompt = _EXTRACTION_PROMPT_TEMPLATE.format(
            research_context=context_line,
            field_list=field_list,
            paper_text=paper_text,
        )

        # Call the LLM. If anything goes wrong at this stage, return a
        # table of *extractor_failed* fields — this is an extractor
        # failure and must not be blamed on the student.
        try:
            raw_response = await self._backend.complete(prompt)
            parsed = parse_llm_response(raw_response, self._backend.model_id)
        except Exception as exc:
            logger.warning(
                "llm_extraction_failed",
                paper_id=document.paper_id,
                extractor=self._extractor_name,
                error=str(exc),
            )
            return ExtractedTable(
                paper_id=document.paper_id,
                fields=[
                    ExtractedField(
                        field_name=f,
                        value=None,
                        evidence="",
                        confidence=0.0,
                        extractor_failed=True,
                    )
                    for f in fields
                ],
                extractor_id=self._extractor_name,
            )

        # Parse the response into ExtractedField objects
        extracted_fields: list[ExtractedField] = []
        raw_fields = parsed.get("fields", [])

        # Build a lookup from normalised requested name → original
        # so we can restore the exact field name the caller asked for,
        # even if the LLM renamed / normalised it in its response.
        def _norm(n: str) -> str:
            return "".join(c for c in n.lower() if c.isalnum())

        requested_by_norm = {_norm(f): f for f in fields}

        for raw_field in raw_fields:
            raw_name = raw_field.get("field_name", "unknown")
            # Restore original requested name if the LLM normalised it
            name = requested_by_norm.get(_norm(raw_name), raw_name)
            raw_value = raw_field.get("value")
            cleaned_value = self._clean_value(raw_value, name)

            extracted_fields.append(
                ExtractedField(
                    field_name=name,
                    value=cleaned_value,
                    evidence=raw_field.get("evidence") or "",
                    confidence=0.8 if cleaned_value is not None else 0.0,
                )
            )

        # Fill in any missing fields (requested but not returned by LLM)
        returned_names = {f.field_name for f in extracted_fields}
        for field_name in fields:
            if field_name not in returned_names:
                extracted_fields.append(
                    ExtractedField(
                        field_name=field_name,
                        value=None,
                        evidence="",
                        confidence=0.0,
                    )
                )

        logger.info(
            "llm_extraction_done",
            paper_id=document.paper_id,
            extractor=self._extractor_name,
            n_extracted=len(extracted_fields),
        )

        return ExtractedTable(
            paper_id=document.paper_id,
            fields=extracted_fields,
            extractor_id=self._extractor_name,
        )

    # Strings that LLMs commonly return when they couldn't find a value.
    # All of these get nullified by ``_clean_value``.
    _GARBAGE_VALUES: frozenset[str] = frozenset({
        "", "n/a", "na", "n.a.", "n.a", "nr", "n.r.", "n/r",
        "not reported", "not available", "not mentioned", "not specified",
        "not stated", "not given", "unknown", "unclear", "none",
        "see table", "see text", "see paper", "see figure",
        "in table", "in figure", "in text",
        "-", "--", "—", "–", "...", "?", ".",
        "null", "none reported", "no data", "no value",
    })

    @classmethod
    def _clean_value(cls, value: object, field_name: str) -> object:
        """Reject garbage LLM outputs (echoed field names, placeholders).

        Returns:
            The original value if it looks legitimate, otherwise ``None``.

        Rejection rules:
          1. Non-string values pass through (numbers, lists, etc.)
          2. Empty / placeholder strings → None
          3. Value identical to the field name (case-insensitive,
             alphanumeric-only) → None  (e.g. value="EFT" for field
             "EFT/ EAT" — the LLM just echoed the abbreviation back)
          4. Value is a substring of the field name AND ≤4 characters
             → None  (catches "EFT" for "EFT/ EAT", "BMI" for "BMI Kg/m2")
          5. String must contain at least one alphanumeric character
        """
        if value is None:
            return None
        if not isinstance(value, str):
            return value  # numbers, bools, etc.

        v = value.strip()
        if not v:
            return None

        # Normalised forms for comparison
        v_lower = v.lower().strip(" .,;:!?\"'()-_/")
        if v_lower in cls._GARBAGE_VALUES:
            return None

        # Echo of the field name itself (alphanumeric-only comparison)
        v_alnum = "".join(c for c in v.lower() if c.isalnum())
        f_alnum = "".join(c for c in field_name.lower() if c.isalnum())
        if v_alnum == f_alnum:
            return None

        # Short value that's a substring of the field name (echo of an
        # abbreviation, e.g. "EFT" inside "EFT/ EAT"; "BMI" inside "BMI Kg/m2")
        if len(v_alnum) <= 4 and v_alnum and v_alnum in f_alnum:
            return None

        # Must contain at least one alphanumeric character
        if not any(c.isalnum() for c in v):
            return None

        return value

    @staticmethod
    def _select_relevant_text(document: PaperDocument, max_chars: int = 25000) -> str:
        """Select the most data-rich portions of the paper for extraction.

        For papers with named sections (e.g. from PMC XML), prioritises:
          1. Abstract (always included — gives overview)
          2. Results / Tables (where quantitative data lives)
          3. Methods (sample sizes, measurement tools, study design)
          4. Discussion (sometimes reports key numbers)

        For short papers (abstracts only), returns the full text.
        """
        # Short text — just return as-is
        if len(document.full_text) <= max_chars:
            return document.full_text

        # If we have named sections, build a priority selection
        if document.sections:
            # Priority order for data extraction
            priority_keys = [
                # Results and tables — highest priority
                "results", "result", "findings",
                # Abstract — always useful
                "abstract",
                # Methods — sample size, measurement tools
                "methods", "method", "materials_and_methods",
                "patients_and_methods", "study_design", "participants",
                "subjects", "study_population",
                # Statistical analysis
                "statistical_analysis", "statistics", "data_analysis",
                # Discussion sometimes has summary numbers
                "discussion",
            ]

            parts: list[str] = []
            used_keys: set[str] = set()
            total_chars = 0

            for pkey in priority_keys:
                for sec_name, sec_text in document.sections.items():
                    if sec_name in used_keys:
                        continue
                    # Match section name (case-insensitive, partial)
                    if pkey in sec_name.lower():
                        # Don't exceed budget
                        if total_chars + len(sec_text) > max_chars:
                            remaining = max_chars - total_chars
                            if remaining > 200:
                                parts.append(
                                    f"[{sec_name}]\n{sec_text[:remaining]}..."
                                )
                                total_chars += remaining
                            break
                        parts.append(f"[{sec_name}]\n{sec_text}")
                        total_chars += len(sec_text)
                        used_keys.add(sec_name)

            if parts:
                selected = "\n\n".join(parts)
                logger.info(
                    "text_selection",
                    source="sections",
                    selected_sections=list(used_keys),
                    chars=len(selected),
                )
                return selected

        # Fallback: for plain text without sections, try to find the
        # data-rich middle part (skip intro, include methods/results)
        text = document.full_text
        # Try to find "Results" or "Methods" section header in plain text
        import re
        results_match = re.search(
            r"\n\s*(Results|RESULTS|Methods|METHODS|Patients|PATIENTS)",
            text,
        )
        if results_match:
            start = max(0, results_match.start() - 200)  # a bit before
            return text[start : start + max_chars]

        # Last resort: take the first max_chars
        return text[:max_chars]

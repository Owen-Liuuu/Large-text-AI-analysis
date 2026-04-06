from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import logging
import re

import fitz  # PyMuPDF
from docx import Document
from pydantic import BaseModel, Field, ValidationError


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -----------------------------------------------------------------------------
# Regex patterns for enrichment
# -----------------------------------------------------------------------------
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
PMID_PATTERN = re.compile(r"\bPMID:\s*(\d+)\b", re.I)
PMCID_PATTERN = re.compile(r"\bPMCID:\s*(PMC\d+)\b", re.I)
NCT_PATTERN = re.compile(r"\bNCT\d{8}\b", re.I)

SECTION_HEADING_PATTERN = re.compile(
    r"(?im)^\s*(abstract|introduction|background|methods?|methodology|materials and methods|"
    r"search strategy|literature search|data sources|information sources|results?|discussion|"
    r"conclusion|references|bibliography)\s*$"
)


DATABASE_MAP = {
    "pubmed": "pubmed",
    "medline": "medline",
    "clinicaltrials.gov": "clinicaltrials_gov",
    "clinical trials.gov": "clinicaltrials_gov",
    "clinicaltrials gov": "clinicaltrials_gov",
    "cochrane library": "cochrane_library",
    "cochrane": "cochrane_library",
    "pubmed central": "pubmed_central",
    "pmc": "pubmed_central",
    "embase": "embase",
    "scopus": "scopus",
    "web of science": "web_of_science",
}


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class ReferenceEntry(BaseModel):
    reference_id: int
    raw_text: str
    title: str | None = None
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    nct_id: str | None = None


class DatabaseEntry(BaseModel):
    name: str
    normalized_name: str


class SearchStrategy(BaseModel):
    raw_text: str = ""
    keywords: list[str] = Field(default_factory=list)
    boolean_operators: list[str] = Field(default_factory=list)
    query_blocks: list[str] = Field(default_factory=list)


class ParsingConfidence(BaseModel):
    databases_confidence: float = 0.0
    search_strategy_confidence: float = 0.0
    references_confidence: float = 0.0


class Module1Output(BaseModel):
    document_id: str
    source_file: str
    parser_version: str
    databases: list[DatabaseEntry] = Field(default_factory=list)
    search_strategy: SearchStrategy = Field(default_factory=SearchStrategy)
    references: list[ReferenceEntry] = Field(default_factory=list)
    student_reported_claims: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    parsing_confidence: ParsingConfidence = Field(default_factory=ParsingConfidence)


# -----------------------------------------------------------------------------
# Optional LLM interface
# -----------------------------------------------------------------------------
class BaseMetadataLLMClient:
    """
    Minimal interface for a metadata extraction LLM client.
    Replace this with your team's Gemini/OpenAI-compatible wrapper if needed.
    """

    def extract_metadata(self, prompt: str) -> dict[str, Any]:
        raise NotImplementedError


class MockMetadataLLMClient(BaseMetadataLLMClient):
    """
    Safe fallback so the parser can still run end-to-end without a real LLM.
    This is useful while wiring the module into the repo.
    """

    def extract_metadata(self, prompt: str) -> dict[str, Any]:
        logger.warning("Using MockMetadataLLMClient. Returning heuristic placeholder output.")
        return {
            "databases": [],
            "search_strategy": {
                "raw_text": "",
                "keywords": [],
                "boolean_operators": [],
                "query_blocks": [],
            },
            "references": [],
            "student_reported_claims": [],
            "warnings": ["Mock LLM used; metadata extraction is incomplete."],
            "parsing_confidence": {
                "databases_confidence": 0.2,
                "search_strategy_confidence": 0.2,
                "references_confidence": 0.2,
            },
        }


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def extract_first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(0).strip() if match else None


def extract_first_group(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def normalize_database_name(name: str) -> str:
    cleaned_name = re.sub(r"\s+", " ", name.strip().lower())
    return DATABASE_MAP.get(cleaned_name, re.sub(r"\W+", "_", cleaned_name).strip("_"))


def clean_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def split_reference_block_into_entries(reference_text: str) -> list[str]:
    """
    Very simple fallback splitter for references if the LLM misses them.
    It assumes references often appear as numbered lines or line-separated entries.
    """
    lines = [line.strip() for line in reference_text.splitlines() if line.strip()]
    if not lines:
        return []

    entries: list[str] = []
    current_entry: list[str] = []

    for line in lines:
        starts_new_reference = bool(re.match(r"^\[?\d+\]?[\.\)]?\s+", line))
        if starts_new_reference and current_entry:
            entries.append(" ".join(current_entry).strip())
            current_entry = [line]
        else:
            current_entry.append(line)

    if current_entry:
        entries.append(" ".join(current_entry).strip())

    return entries


# -----------------------------------------------------------------------------
# Main parser
# -----------------------------------------------------------------------------
class DocumentParser:
    """
    Module 1 - Document Parser

    Responsibilities:
    - Read PDF or DOCX review files
    - Extract raw text
    - Detect useful sections (methods/search strategy/references/results)
    - Use an LLM to extract structured metadata
    - Validate and enrich the metadata
    - Save standardized JSON for downstream modules
    """

    def __init__(
        self,
        llm_client: BaseMetadataLLMClient | None = None,
        parser_version: str = "1.0.0",
    ) -> None:
        self.llm_client = llm_client or MockMetadataLLMClient()
        self.parser_version = parser_version

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def parse(
        self,
        file_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Module1Output:
        input_path = Path(file_path)
        logger.info("Parsing document: %s", input_path)

        raw_text = self.extract_text(input_path)
        sections = self.split_sections(raw_text)
        llm_output = self.llm_extract_metadata(input_path, sections)
        enriched_output = self.enrich_output(llm_output, raw_text, sections)

        enriched_output["document_id"] = self.build_document_id(input_path)
        enriched_output["source_file"] = input_path.name
        enriched_output["parser_version"] = self.parser_version

        try:
            validated_output = Module1Output.model_validate(enriched_output)
        except ValidationError as exc:
            logger.exception("Validation failed for parsed output.")
            raise ValueError(f"Parsed metadata failed schema validation: {exc}") from exc

        if output_path is not None:
            self.save_output(validated_output, output_path)

        return validated_output

    # -------------------------------------------------------------------------
    # File text extraction
    # -------------------------------------------------------------------------
    def extract_text(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self.extract_pdf_text(file_path)
        if suffix == ".docx":
            return self.extract_docx_text(file_path)

        raise ValueError(f"Unsupported file type: {suffix}")

    def extract_pdf_text(self, file_path: Path) -> str:
        document = fitz.open(file_path)
        page_texts: list[str] = []

        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text")
            if page_text.strip():
                page_texts.append(f"\n--- PAGE {page_number} ---\n{page_text}")

        full_text = "\n".join(page_texts).strip()
        if not full_text:
            logger.warning("No text was extracted from PDF: %s", file_path)

        return full_text

    def extract_docx_text(self, file_path: Path) -> str:
        document = Document(file_path)
        paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        full_text = "\n".join(paragraphs).strip()

        if not full_text:
            logger.warning("No text was extracted from DOCX: %s", file_path)

        return full_text

    # -------------------------------------------------------------------------
    # Section splitting
    # -------------------------------------------------------------------------
    def split_sections(self, full_text: str) -> dict[str, str]:
        """
        Lightweight heading-based splitter.
        Falls back to heuristic slices if headings are weak or absent.
        """
        normalized_lines = [line.rstrip() for line in full_text.splitlines()]
        heading_indices: list[tuple[int, str]] = []

        for index, line in enumerate(normalized_lines):
            if SECTION_HEADING_PATTERN.match(line.strip()):
                heading_indices.append((index, line.strip().lower()))

        sections: dict[str, str] = {}

        if heading_indices:
            for position, (start_index, heading) in enumerate(heading_indices):
                end_index = heading_indices[position + 1][0] if position + 1 < len(heading_indices) else len(normalized_lines)
                section_text = "\n".join(normalized_lines[start_index:end_index]).strip()

                canonical_name = self.map_heading_to_section_name(heading)
                if canonical_name in sections:
                    sections[canonical_name] += "\n\n" + section_text
                else:
                    sections[canonical_name] = section_text

        # Fallbacks if useful sections were not found.
        lowered_text = full_text.lower()

        if "methods" not in sections and "methodology" not in sections:
            methods_fallback = self.extract_keyword_window(
                full_text,
                keywords=["pubmed", "clinicaltrials.gov", "cochrane", "search strategy", "information sources"],
                window_chars=6000,
            )
            if methods_fallback:
                sections["methods"] = methods_fallback

        if "references" not in sections:
            last_chunk = full_text[int(len(full_text) * 0.7):]
            if "pmid" in last_chunk.lower() or "doi" in last_chunk.lower() or "references" in last_chunk.lower():
                sections["references"] = last_chunk

        if "results" not in sections:
            results_fallback = self.extract_keyword_window(
                full_text,
                keywords=["sample size", "participants", "outcome", "intervention", "results"],
                window_chars=5000,
            )
            if results_fallback:
                sections["results"] = results_fallback

        sections["full_text"] = full_text
        return sections

    def map_heading_to_section_name(self, heading: str) -> str:
        normalized_heading = heading.strip().lower()

        if normalized_heading in {"methods", "method", "methodology", "materials and methods"}:
            return "methods"
        if normalized_heading in {"search strategy", "literature search", "data sources", "information sources"}:
            return "search_strategy"
        if normalized_heading in {"references", "bibliography"}:
            return "references"
        if normalized_heading in {"results", "result"}:
            return "results"
        if normalized_heading in {"abstract"}:
            return "abstract"
        if normalized_heading in {"introduction", "background"}:
            return "introduction"
        if normalized_heading in {"discussion"}:
            return "discussion"
        if normalized_heading in {"conclusion"}:
            return "conclusion"

        return re.sub(r"\W+", "_", normalized_heading).strip("_")

    def extract_keyword_window(
        self,
        full_text: str,
        keywords: list[str],
        window_chars: int = 4000,
    ) -> str:
        lowered_text = full_text.lower()

        for keyword in keywords:
            keyword_index = lowered_text.find(keyword.lower())
            if keyword_index != -1:
                start_index = max(0, keyword_index - window_chars // 3)
                end_index = min(len(full_text), keyword_index + window_chars)
                return full_text[start_index:end_index].strip()

        return ""

    # -------------------------------------------------------------------------
    # LLM extraction
    # -------------------------------------------------------------------------
    def llm_extract_metadata(
        self,
        file_path: Path,
        sections: dict[str, str],
    ) -> dict[str, Any]:
        prompt = self.build_prompt(file_path, sections)

        try:
            llm_result = self.llm_client.extract_metadata(prompt)
        except Exception as exc:
            logger.exception("LLM metadata extraction failed.")
            llm_result = {
                "databases": [],
                "search_strategy": {
                    "raw_text": "",
                    "keywords": [],
                    "boolean_operators": [],
                    "query_blocks": [],
                },
                "references": [],
                "student_reported_claims": [],
                "warnings": [f"LLM extraction failed: {exc}"],
                "parsing_confidence": {
                    "databases_confidence": 0.0,
                    "search_strategy_confidence": 0.0,
                    "references_confidence": 0.0,
                },
            }

        return llm_result

    def build_prompt(self, file_path: Path, sections: dict[str, str]) -> str:
        methods_text = sections.get("methods", "")
        search_text = sections.get("search_strategy", "")
        references_text = sections.get("references", "")
        results_text = sections.get("results", "")

        return f"""
You are extracting structured metadata from a student systematic/scoping review.

Return VALID JSON ONLY. Do not include markdown. Do not include explanations.

Extract:
1. The bibliographic databases searched by the student.
2. The search keywords and Boolean operators exactly as documented in the methodology.
3. The references cited in the review.
4. Optionally, short factual claims reported by the student in the results/tables that may later be validated.

Rules:
- Preserve Boolean query structure where possible.
- Use exact database names from the document when possible.
- For references, include the raw reference text.
- If PMID, PMCID, DOI, or NCT ID is visible, extract it.
- If a field is missing, use null or an empty list.
- Do not invent information.

Return JSON in this format:
{{
  "databases": [
    {{"name": "...", "normalized_name": "..."}}
  ],
  "search_strategy": {{
    "raw_text": "...",
    "keywords": ["..."],
    "boolean_operators": ["AND", "OR"],
    "query_blocks": ["..."]
  }},
  "references": [
    {{
      "reference_id": 1,
      "raw_text": "...",
      "title": null,
      "doi": null,
      "pmid": null,
      "pmcid": null,
      "nct_id": null
    }}
  ],
  "student_reported_claims": ["..."],
  "warnings": [],
  "parsing_confidence": {{
    "databases_confidence": 0.0,
    "search_strategy_confidence": 0.0,
    "references_confidence": 0.0
  }}
}}

Document name: {file_path.name}

METHODS SECTION:
{methods_text[:12000]}

SEARCH STRATEGY SECTION:
{search_text[:12000]}

REFERENCES SECTION:
{references_text[:20000]}

RESULTS SECTION:
{results_text[:12000]}
""".strip()

    # -------------------------------------------------------------------------
    # Post-processing / enrichment
    # -------------------------------------------------------------------------
    def enrich_output(
        self,
        llm_output: dict[str, Any],
        raw_text: str,
        sections: dict[str, str],
    ) -> dict[str, Any]:
        llm_output = dict(llm_output) if llm_output else {}

        llm_output.setdefault("databases", [])
        llm_output.setdefault("search_strategy", {})
        llm_output.setdefault("references", [])
        llm_output.setdefault("student_reported_claims", [])
        llm_output.setdefault("warnings", [])
        llm_output.setdefault("parsing_confidence", {})

        search_strategy = llm_output["search_strategy"]
        if not isinstance(search_strategy, dict):
            search_strategy = {}

        search_strategy.setdefault("raw_text", "")
        search_strategy.setdefault("keywords", [])
        search_strategy.setdefault("boolean_operators", [])
        search_strategy.setdefault("query_blocks", [])

        # Fallback search strategy extraction if LLM missed it.
        if not search_strategy["raw_text"]:
            fallback_strategy_text = sections.get("search_strategy") or sections.get("methods", "")
            search_strategy["raw_text"] = fallback_strategy_text[:4000].strip()

        if not search_strategy["boolean_operators"]:
            detected_operators = sorted(set(re.findall(r"\b(AND|OR|NOT)\b", search_strategy["raw_text"], flags=re.I)))
            search_strategy["boolean_operators"] = [operator.upper() for operator in detected_operators]

        if not search_strategy["keywords"]:
            quoted_terms = re.findall(r'"([^"]+)"', search_strategy["raw_text"])
            search_strategy["keywords"] = list(dict.fromkeys([term.strip() for term in quoted_terms if term.strip()]))

        llm_output["search_strategy"] = search_strategy

        # Normalize database names and add heuristic fallback.
        normalized_databases: list[dict[str, str]] = []
        seen_database_names: set[str] = set()

        for database in llm_output["databases"]:
            if not isinstance(database, dict):
                continue
            raw_name = str(database.get("name", "")).strip()
            if not raw_name:
                continue

            normalized_name = normalize_database_name(raw_name)
            if normalized_name in seen_database_names:
                continue

            seen_database_names.add(normalized_name)
            normalized_databases.append(
                {"name": raw_name, "normalized_name": normalized_name}
            )

        if not normalized_databases:
            heuristic_databases = self.detect_databases_from_text(raw_text)
            for raw_name in heuristic_databases:
                normalized_name = normalize_database_name(raw_name)
                if normalized_name not in seen_database_names:
                    seen_database_names.add(normalized_name)
                    normalized_databases.append(
                        {"name": raw_name, "normalized_name": normalized_name}
                    )
            if heuristic_databases:
                llm_output["warnings"].append("Databases recovered heuristically from document text.")

        llm_output["databases"] = normalized_databases

        # Clean and enrich references.
        cleaned_references = self.clean_references(
            references=llm_output["references"],
            references_section=sections.get("references", ""),
        )
        llm_output["references"] = cleaned_references

        # Fallback claims extraction.
        claims = llm_output.get("student_reported_claims", [])
        if not claims:
            claims = self.extract_student_reported_claims(sections.get("results", ""))
            if claims:
                llm_output["warnings"].append("Student reported claims recovered heuristically from results text.")

        llm_output["student_reported_claims"] = claims[:50]

        # Populate warnings if key fields are missing.
        if not llm_output["databases"]:
            llm_output["warnings"].append("No bibliographic databases were confidently extracted.")
        if not llm_output["references"]:
            llm_output["warnings"].append("No references were confidently extracted.")
        if not llm_output["search_strategy"]["raw_text"]:
            llm_output["warnings"].append("No search strategy text was confidently extracted.")

        # Set default confidence values if missing.
        parsing_confidence = llm_output["parsing_confidence"]
        if not isinstance(parsing_confidence, dict):
            parsing_confidence = {}

        parsing_confidence.setdefault("databases_confidence", 0.5 if llm_output["databases"] else 0.0)
        parsing_confidence.setdefault("search_strategy_confidence", 0.5 if llm_output["search_strategy"]["raw_text"] else 0.0)
        parsing_confidence.setdefault("references_confidence", 0.5 if llm_output["references"] else 0.0)
        llm_output["parsing_confidence"] = parsing_confidence

        return llm_output

    def detect_databases_from_text(self, raw_text: str) -> list[str]:
        detected_databases: list[str] = []
        lowered_text = raw_text.lower()

        database_candidates = [
            "PubMed",
            "ClinicalTrials.gov",
            "Cochrane Library",
            "PubMed Central",
            "Embase",
            "Scopus",
            "Web of Science",
            "MEDLINE",
        ]

        for candidate in database_candidates:
            if candidate.lower() in lowered_text:
                detected_databases.append(candidate)

        return detected_databases

    def clean_references(
        self,
        references: list[Any],
        references_section: str,
    ) -> list[dict[str, Any]]:
        cleaned_references: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        # Use LLM references if present.
        for reference_index, reference in enumerate(references, start=1):
            if not isinstance(reference, dict):
                continue

            raw_text = clean_whitespace(str(reference.get("raw_text", "")))
            if not raw_text:
                continue

            dedupe_key = re.sub(r"\s+", " ", raw_text.lower())
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)

            enriched_reference = {
                "reference_id": reference_index,
                "raw_text": raw_text,
                "title": reference.get("title"),
                "doi": reference.get("doi") or extract_first_match(DOI_PATTERN, raw_text),
                "pmid": reference.get("pmid") or extract_first_group(PMID_PATTERN, raw_text),
                "pmcid": reference.get("pmcid") or extract_first_group(PMCID_PATTERN, raw_text),
                "nct_id": reference.get("nct_id") or extract_first_match(NCT_PATTERN, raw_text),
            }
            cleaned_references.append(enriched_reference)

        # Fallback: recover reference entries heuristically if LLM returned nothing.
        if not cleaned_references and references_section.strip():
            fallback_entries = split_reference_block_into_entries(references_section)
            for reference_index, raw_entry in enumerate(fallback_entries, start=1):
                raw_entry = clean_whitespace(raw_entry)
                if not raw_entry:
                    continue

                dedupe_key = re.sub(r"\s+", " ", raw_entry.lower())
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)

                cleaned_references.append(
                    {
                        "reference_id": reference_index,
                        "raw_text": raw_entry,
                        "title": None,
                        "doi": extract_first_match(DOI_PATTERN, raw_entry),
                        "pmid": extract_first_group(PMID_PATTERN, raw_entry),
                        "pmcid": extract_first_group(PMCID_PATTERN, raw_entry),
                        "nct_id": extract_first_match(NCT_PATTERN, raw_entry),
                    }
                )

        return cleaned_references

    def extract_student_reported_claims(self, results_text: str) -> list[str]:
        """
        Lightweight heuristic claim extraction.
        This is intentionally simple and mainly exists to help downstream Module 5.
        """
        if not results_text.strip():
            return []

        candidate_sentences = re.split(r"(?<=[.!?])\s+", results_text)
        claims: list[str] = []

        claim_patterns = [
            r"\bsample size\b",
            r"\bparticipants?\b",
            r"\bintervention\b",
            r"\boutcome\b",
            r"\bprimary outcome\b",
            r"\bsecondary outcome\b",
            r"\bduration\b",
            r"\bweeks?\b",
            r"\bmonths?\b",
            r"\bmean\b",
            r"\bmedian\b",
            r"\bp\s*[<=>]\s*0\.\d+\b",
        ]

        for sentence in candidate_sentences:
            cleaned_sentence = clean_whitespace(sentence)
            if len(cleaned_sentence) < 20 or len(cleaned_sentence) > 300:
                continue

            if any(re.search(pattern, cleaned_sentence, flags=re.I) for pattern in claim_patterns):
                claims.append(cleaned_sentence)

        # Deduplicate while preserving order.
        deduplicated_claims = list(dict.fromkeys(claims))
        return deduplicated_claims[:50]

    # -------------------------------------------------------------------------
    # Output helpers
    # -------------------------------------------------------------------------
    def build_document_id(self, file_path: Path) -> str:
        file_hash = hashlib.md5(str(file_path.resolve()).encode("utf-8")).hexdigest()
        return file_hash[:12]

    def save_output(
        self,
        parsed_output: Module1Output,
        output_path: str | Path,
    ) -> None:
        destination_path = Path(output_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(
            parsed_output.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info("Saved parsed JSON to %s", destination_path)


# -----------------------------------------------------------------------------
# Example local usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = DocumentParser()

    # Replace with a real review path in your repo.
    example_input = "example_review.pdf"
    example_output = "outputs/module1_output.json"

    if Path(example_input).exists():
        parsed = parser.parse(example_input, example_output)
        print(json.dumps(parsed.model_dump(), indent=2))
    else:
        print(
            "Parser ready. Update 'example_input' with a real PDF/DOCX path to test locally."
        )
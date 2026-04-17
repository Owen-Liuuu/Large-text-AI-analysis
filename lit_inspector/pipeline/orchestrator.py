"""Pipeline orchestrator: runs the 4-step pipeline sequentially.

Correct flow:
  Step 1: LLM analyses the review → identifies research features + search terms
          → reproduces search on PubMed → checks count + cross-checks papers
  Step 2: Verify each selected paper exists via CrossRef
  Step 3: Retrieve + extract data from selected papers using LLM,
          using the dynamically identified features from Step 1
  Step 4: Compare student's tables vs AI-generated tables → report
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from difflib import SequenceMatcher

import structlog

from lit_inspector.core.enums import PipelineStep, ValidationSeverity
from lit_inspector.llm.base import LLMBackend, parse_llm_response
from lit_inspector.steps.search_validation.interfaces import SearchProvider
from lit_inspector.steps.search_validation.schemas import SearchStrategy
from lit_inspector.steps.paper_verification.interfaces import (
    PaperRetriever,
    ReferenceVerifier,
)
from lit_inspector.steps.data_extraction.interfaces import Extractor
from lit_inspector.steps.table_comparison.interfaces import (
    ReportGenerator,
    TableComparator,
)
from lit_inspector.pipeline.schemas import (
    PipelineRunResult,
    StudentReviewInput,
    ValidationFlag,
)

logger = structlog.get_logger(__name__)

# ======================================================================
# Prompt: Step 1 — LLM analyses the review to identify features + query
# ======================================================================

_STEP1_ANALYSIS_PROMPT = """You are an expert in systematic review methodology.

Read the following systematic review text and extract:

1. **research_context**: A one-sentence description of what this review studies.
   Example: "EAT thickness and volume in T1DM patients vs healthy controls"

2. **search_query**: The PubMed search query you would construct to reproduce the
   authors' search. Use Boolean operators (AND, OR). Keep it focused.
   Example: "epicardial adipose tissue AND type 1 diabetes mellitus"

3. **extraction_features**: The list of DATA FIELDS that the authors extracted from
   each included study for their evidence table / meta-analysis. These are the
   column headers of their data extraction table (e.g. Table 1 or forest plot
   data). Each feature should be a short, snake_case name.

   IMPORTANT: Include BOTH descriptive features (country, sample_size, imaging
   modality) AND quantitative outcome data (mean, sd, n for each group). These
   are the numbers that go into forest plots.

   Example for an EAT / T1DM review:
   ["author", "country", "sample_size", "measurement_tool", "age_mean",
    "bmi", "intervention_group_mean", "intervention_group_sd",
    "intervention_group_n", "control_group_mean", "control_group_sd",
    "control_group_n", "outcome_measure"]

## REVIEW TEXT (may be truncated)
{review_text}

## OUTPUT FORMAT
Return ONLY valid JSON:
```json
{{
  "research_context": "...",
  "search_query": "...",
  "extraction_features": ["feature1", "feature2", ...]
}}
```"""


class PipelineOrchestrator:
    """Coordinates the 4-step literature integrity checking pipeline.

    Args:
        search_provider: Step 1 implementation.
        reference_verifier: Step 2 verifier implementation.
        paper_retriever: Step 2/3 retriever implementation.
        extractors: Step 3 extractor implementations (1 or more).
        table_comparator: Step 4 comparator implementation.
        report_generator: Step 4 report generator implementation.
        llm_backend: LLM backend for Step 1 review analysis (optional;
            if None, skips LLM analysis and uses student-provided fields).
        enabled_steps: Which steps to run. Defaults to all.
    """

    def __init__(
        self,
        search_provider: SearchProvider,
        reference_verifier: ReferenceVerifier,
        paper_retriever: PaperRetriever,
        extractors: list[Extractor],
        table_comparator: TableComparator,
        report_generator: ReportGenerator,
        llm_backend: LLMBackend | None = None,
        enabled_steps: list[str] | None = None,
    ) -> None:
        self._search_provider = search_provider
        self._reference_verifier = reference_verifier
        self._paper_retriever = paper_retriever
        self._extractors = extractors
        self._table_comparator = table_comparator
        self._report_generator = report_generator
        self._llm = llm_backend
        self._enabled_steps = set(enabled_steps or [s.value for s in PipelineStep])

    async def run(self, student_input: StudentReviewInput) -> PipelineRunResult:
        """Execute the pipeline on a student's submission."""
        run_id = uuid.uuid4().hex[:12]
        logger.info(
            "pipeline_start",
            run_id=run_id,
            student_id=student_input.student_id,
            n_selected=len(student_input.selected_papers),
        )

        result = PipelineRunResult(run_id=run_id, student_input=student_input)

        # Step 1: Search Validation (includes LLM analysis if available)
        if PipelineStep.SEARCH_VALIDATION.value in self._enabled_steps:
            result = await self._run_search_validation(result, student_input)

        # Step 2: Paper Verification (verify selected papers exist)
        if PipelineStep.PAPER_VERIFICATION.value in self._enabled_steps:
            result = await self._run_paper_verification(result, student_input)

        # Step 3: Data Extraction (uses dynamic features from Step 1)
        if PipelineStep.DATA_EXTRACTION.value in self._enabled_steps:
            result = await self._run_data_extraction(result, student_input)

        # Step 4: Table Comparison
        if PipelineStep.TABLE_COMPARISON.value in self._enabled_steps:
            result = await self._run_table_comparison(result)

        result.completed_at = datetime.now()
        logger.info("pipeline_complete", run_id=run_id, flags=len(result.all_flags))
        return result

    # ==================================================================
    # Step 1: LLM Review Analysis + Search Validation
    # ==================================================================

    async def _run_search_validation(
        self, result: PipelineRunResult, student_input: StudentReviewInput
    ) -> PipelineRunResult:
        """Step 1: Analyse the review with LLM, then reproduce the search.

        Sub-steps:
          1a. LLM reads the review text → identifies research context,
              optimal search query, and extraction features
          1b. Reproduce the search on PubMed using the LLM's query
              (falls back to student-provided query if no LLM)
          1c. Cross-check: do selected papers appear in search results?
        """
        logger.info("step_start", step="search_validation")
        try:
            # Step 1a: LLM analysis of the review
            await self._analyse_review(result, student_input)

            # Step 1b: Reproduce search on PubMed
            # Use LLM-constructed query if available, else student's text
            search_text = result.llm_search_query or student_input.search_strategy_text
            strategy = SearchStrategy(
                database=student_input.search_database,
                raw_strategy_text=search_text,
                reported_result_count=student_input.reported_result_count,
            )
            result.search_result = await self._search_provider.validate_strategy(
                strategy
            )

            # Convert step-level flags to pipeline-level flags
            for flag in result.search_result.flags:
                result.all_flags.append(
                    ValidationFlag(
                        step=PipelineStep.SEARCH_VALIDATION,
                        severity=flag.severity,
                        code=flag.code,
                        message=flag.message,
                    )
                )

            # Step 1c: Cross-check selected papers in search results
            result.papers_in_search = self._check_papers_in_search(
                student_input, result
            )

        except Exception as exc:
            logger.error("step_failed", step="search_validation", error=str(exc))
            result.all_flags.append(
                ValidationFlag(
                    step=PipelineStep.SEARCH_VALIDATION,
                    severity=ValidationSeverity.ERROR,
                    code="STEP_FAILED",
                    message=f"Search validation failed: {exc}",
                )
            )
        return result

    async def _analyse_review(
        self, result: PipelineRunResult, student_input: StudentReviewInput
    ) -> None:
        """Step 1a: Use LLM to analyse the review and dynamically identify
        research context, search query, and extraction features.

        Results are stored in ``result.research_context``,
        ``result.identified_features``, and ``result.llm_search_query``.
        """
        if self._llm is None:
            logger.info("step1_llm_skip", reason="no LLM backend configured")
            return
        if not student_input.review_full_text:
            logger.info(
                "step1_llm_skip", reason="no review_full_text (YAML input?)"
            )
            return

        logger.info("step1_llm_analysis_start")
        try:
            review_text = student_input.review_full_text[:20000]
            prompt = _STEP1_ANALYSIS_PROMPT.format(review_text=review_text)
            raw = await self._llm.complete(prompt)
            parsed = parse_llm_response(raw, self._llm.model_id)

            result.research_context = parsed.get("research_context", "")
            result.llm_search_query = parsed.get("search_query", "")
            features = parsed.get("extraction_features", [])
            if isinstance(features, list):
                result.identified_features = features

            logger.info(
                "step1_llm_analysis_done",
                context=result.research_context[:80],
                query=result.llm_search_query[:80],
                n_features=len(result.identified_features),
                features=result.identified_features,
            )

        except Exception as exc:
            logger.warning(
                "step1_llm_analysis_failed",
                error=str(exc),
                fallback="using student-provided fields",
            )

    def _check_papers_in_search(
        self, student_input: StudentReviewInput, result: PipelineRunResult
    ) -> dict[str, bool]:
        """Check if each selected paper appears in the PubMed search results.

        Matches by DOI (exact) or title similarity (fuzzy).
        Returns a dict of paper_title -> found_in_search.
        """
        papers_found: dict[str, bool] = {}

        if not result.search_result or not result.search_result.sample_results:
            return papers_found

        # Collect DOIs and titles from search results
        search_dois = {
            s.doi.lower() for s in result.search_result.sample_results if s.doi
        }
        search_titles = [
            s.title.lower() for s in result.search_result.sample_results
        ]

        for paper in student_input.selected_papers:
            found = False
            reason = ""

            # Check by DOI
            if paper.doi and paper.doi.lower() in search_dois:
                found = True
                reason = "DOI match"

            # Check by title similarity
            if not found:
                for st in search_titles:
                    sim = SequenceMatcher(None, paper.title.lower(), st).ratio()
                    if sim > 0.85:
                        found = True
                        reason = f"title match (sim={sim:.2f})"
                        break

            papers_found[paper.title] = found

            if not found:
                result.all_flags.append(
                    ValidationFlag(
                        step=PipelineStep.SEARCH_VALIDATION,
                        severity=ValidationSeverity.WARNING,
                        code="PAPER_NOT_IN_SEARCH",
                        message=(
                            f"Selected paper not found in reproduced search results: "
                            f"'{paper.title[:60]}...'"
                        ),
                        details=(
                            "This paper was not in the first 20 PubMed results. "
                            "It may appear later in the full result set, or "
                            "the search strategy may not capture it."
                        ),
                    )
                )
            else:
                logger.info(
                    "paper_in_search",
                    title=paper.title[:50],
                    reason=reason,
                )

        return papers_found

    # ==================================================================
    # Step 2: Paper Verification
    # ==================================================================

    async def _run_paper_verification(
        self, result: PipelineRunResult, student_input: StudentReviewInput
    ) -> PipelineRunResult:
        """Step 2: Verify each selected paper exists via CrossRef."""
        logger.info(
            "step_start",
            step="paper_verification",
            n_papers=len(student_input.selected_papers),
        )
        try:
            for ref in student_input.selected_papers:
                verification = await self._reference_verifier.verify(ref)
                result.verification_results.append(verification)

                for flag in verification.flags:
                    result.all_flags.append(
                        ValidationFlag(
                            step=PipelineStep.PAPER_VERIFICATION,
                            severity=flag.severity,
                            code=flag.code,
                            message=flag.message,
                        )
                    )
        except Exception as exc:
            logger.error("step_failed", step="paper_verification", error=str(exc))
            result.all_flags.append(
                ValidationFlag(
                    step=PipelineStep.PAPER_VERIFICATION,
                    severity=ValidationSeverity.ERROR,
                    code="STEP_FAILED",
                    message=f"Paper verification failed: {exc}",
                )
            )
        return result

    # ==================================================================
    # Step 3: Data Extraction (dynamic features from Step 1)
    # ==================================================================

    async def _run_data_extraction(
        self, result: PipelineRunResult, student_input: StudentReviewInput
    ) -> PipelineRunResult:
        """Step 3: Retrieve full text and extract data from selected papers.

        The extraction fields are the UNION of three sources, so the LLM
        is always asked for everything the student claims to have extracted
        (otherwise comparison would show N/A for fields the LLM was never
        requested to find):

          - ``result.identified_features``        (Step 1 LLM analysis of the review)
          - ``student_input.extraction_fields``   (declared fields from PDF parser / YAML)
          - field_names from ``student_input.submitted_tables``  (actual columns)

        Uses PaperRetriever to get full text, then runs multiple Extractors
        (different LLMs) on each paper for cross-validation.
        """
        # ----- Build the union of requested fields -----
        # Use a dict-as-ordered-set to preserve order and de-duplicate
        # (case-insensitive). This guarantees that any column the student
        # actually submitted gets requested from the LLM.
        seen: dict[str, str] = {}
        sources_used: list[str] = []

        def _add_fields(field_names: list[str], source: str) -> None:
            added = False
            for name in field_names:
                if not name:
                    continue
                key = name.strip().lower()
                if key and key not in seen:
                    seen[key] = name.strip()
                    added = True
            if added and source not in sources_used:
                sources_used.append(source)

        _add_fields(result.identified_features, "llm_identified")
        _add_fields(student_input.extraction_fields, "student_declared")
        # Pull column headers from every submitted table
        student_table_columns: list[str] = []
        for st in student_input.submitted_tables:
            for f in st.fields:
                if f.field_name:
                    student_table_columns.append(f.field_name)
        _add_fields(student_table_columns, "student_table_columns")

        fields = list(seen.values())

        if not fields:
            logger.warning(
                "step_skip",
                step="data_extraction",
                reason="no extraction features from any source",
            )
            return result

        logger.info(
            "step_start",
            step="data_extraction",
            sources=sources_used,
            n_fields=len(fields),
            n_papers=len(student_input.selected_papers),
            n_extractors=len(self._extractors),
            fields=fields,
        )

        # Record extractor IDs upfront so the report always knows which
        # LLM columns to render even if some extractors fail.
        result.extractor_ids = [e.extractor_id for e in self._extractors]

        try:
            for ref in student_input.selected_papers:
                # Retrieve full text / accessible content
                doc = await self._paper_retriever.retrieve(ref)
                if doc is None:
                    logger.warning(
                        "paper_not_retrievable",
                        title=ref.title[:50],
                    )
                    result.all_flags.append(
                        ValidationFlag(
                            step=PipelineStep.DATA_EXTRACTION,
                            severity=ValidationSeverity.WARNING,
                            code="PAPER_NOT_RETRIEVABLE",
                            message=f"Could not retrieve full text: '{ref.title[:60]}...'",
                        )
                    )
                    continue

                # Run each extractor independently for cross-validation
                for extractor in self._extractors:
                    table = await extractor.extract(
                        doc,
                        fields,
                        research_context=result.research_context,
                    )
                    result.extracted_tables.append(table)
                    logger.info(
                        "extraction_done",
                        paper=ref.title[:40],
                        extractor=extractor.extractor_id,
                        fields=len(table.fields),
                    )

        except Exception as exc:
            logger.error("step_failed", step="data_extraction", error=str(exc))
            result.all_flags.append(
                ValidationFlag(
                    step=PipelineStep.DATA_EXTRACTION,
                    severity=ValidationSeverity.ERROR,
                    code="STEP_FAILED",
                    message=f"Data extraction failed: {exc}",
                )
            )
        return result

    # ==================================================================
    # Step 4: Table Comparison + Report
    # ==================================================================

    @staticmethod
    def _normalise_doi(doi: str) -> str:
        """Normalise a DOI for comparison: lowercase, strip URL prefix."""
        d = doi.strip().lower()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if d.startswith(prefix):
                d = d[len(prefix):]
        return d

    def _find_matching_tables(
        self,
        paper_id: str,
        paper_title: str,
        tables: list,
    ) -> list:
        """Find tables whose paper_id matches by normalised DOI or title."""
        norm_id = self._normalise_doi(paper_id) if paper_id else ""
        matched = []
        for t in tables:
            t_norm = self._normalise_doi(t.paper_id) if t.paper_id else ""
            # Exact normalised DOI match
            if norm_id and t_norm and norm_id == t_norm:
                matched.append(t)
                continue
            # Title-based match (if paper_id contains the title substring)
            if paper_title and t.paper_id:
                t_lower = t.paper_id.lower()
                title_lower = paper_title.lower()[:40]
                if title_lower in t_lower or t_lower in title_lower:
                    matched.append(t)
        return matched

    async def _run_table_comparison(
        self, result: PipelineRunResult
    ) -> PipelineRunResult:
        """Step 4: Compare student's tables with AI-generated tables.

        Iterates over ALL selected papers (not just student-submitted tables)
        so every paper appears in the report. Uses DOI normalisation and
        title fallback to match student tables ↔ AI tables.
        """
        logger.info("step_start", step="table_comparison")
        try:
            comparison_results = []

            # Build the list of ALL papers to compare
            all_papers = result.student_input.selected_papers

            # Pre-assign orphaned student tables (paper_id = "") by index.
            # This handles the case where the PDF parser dropped DOIs.
            orphaned_student_tables = [
                t for t in result.student_input.submitted_tables if not t.paper_id.strip()
            ]
            orphan_idx = 0  # cursor into orphaned tables

            for ref in all_papers:
                paper_doi = ref.doi or ""
                paper_title = ref.title or ""

                # Find matching student table (by DOI or title)
                student_tables = self._find_matching_tables(
                    paper_doi, paper_title, result.student_input.submitted_tables
                )
                student_table = student_tables[0] if student_tables else None

                # Fallback: if no match and orphaned tables remain, assign next one
                if student_table is None and orphan_idx < len(orphaned_student_tables):
                    student_table = orphaned_student_tables[orphan_idx]
                    orphan_idx += 1
                    logger.info(
                        "student_table_orphan_assigned",
                        paper=paper_title[:50],
                        table_paper_id=student_table.paper_id,
                    )

                # Find matching AI-generated tables
                model_tables = self._find_matching_tables(
                    paper_doi, paper_title, result.extracted_tables
                )

                # Canonical paper_id for comparison: prefer validated DOI
                canonical_id = paper_doi or paper_title[:60]

                if student_table and model_tables:
                    # Ensure the student table uses the canonical paper_id
                    # so the comparison result's paper_id is always the DOI.
                    # IMPORTANT: we do NOT synthesise empty rows here —
                    # empty values on the student side must remain empty
                    # so the comparator can correctly assign MISSING_STUDENT.
                    from lit_inspector.steps.data_extraction.schemas import (
                        ExtractedTable,
                    )
                    normalised_student = ExtractedTable(
                        paper_id=canonical_id,
                        fields=student_table.fields,
                        extractor_id=student_table.extractor_id,
                    )
                    comp = await self._table_comparator.compare(
                        normalised_student, model_tables
                    )
                    comparison_results.append(comp)
                elif model_tables:
                    # AI extracted data but no student table. Feed a
                    # truly empty student table to the comparator so it
                    # can mark every field as MISSING_STUDENT — we no
                    # longer synthesise placeholder student rows.
                    from lit_inspector.steps.data_extraction.schemas import (
                        ExtractedTable,
                    )
                    empty_student = ExtractedTable(
                        paper_id=canonical_id,
                        fields=[],
                        extractor_id="student",
                    )
                    comp = await self._table_comparator.compare(
                        empty_student, model_tables
                    )
                    comparison_results.append(comp)
                    result.all_flags.append(
                        ValidationFlag(
                            step=PipelineStep.TABLE_COMPARISON,
                            severity=ValidationSeverity.INFO,
                            code="NO_STUDENT_TABLE",
                            message=(
                                f"No student table for '{paper_title[:60]}'. "
                                "Showing AI extraction only."
                            ),
                        )
                    )
                elif student_table:
                    # Student submitted a table but AI couldn't extract
                    result.all_flags.append(
                        ValidationFlag(
                            step=PipelineStep.TABLE_COMPARISON,
                            severity=ValidationSeverity.WARNING,
                            code="NO_MODEL_TABLE",
                            message=(
                                f"No AI-generated table for '{paper_title[:60]}'. "
                                "Cannot compare."
                            ),
                        )
                    )
                else:
                    # Neither side has data for this paper
                    result.all_flags.append(
                        ValidationFlag(
                            step=PipelineStep.TABLE_COMPARISON,
                            severity=ValidationSeverity.WARNING,
                            code="NO_TABLES",
                            message=(
                                f"No data for '{paper_title[:60]}' "
                                "from student or AI."
                            ),
                        )
                    )

            result.comparison_results = comparison_results

            # Generate final report
            result.report = await self._report_generator.generate(
                comparison_results, result.run_id
            )

            for flag in result.report.overall_flags:
                result.all_flags.append(
                    ValidationFlag(
                        step=PipelineStep.TABLE_COMPARISON,
                        severity=flag.severity,
                        code=flag.code,
                        message=flag.message,
                    )
                )
        except Exception as exc:
            logger.error("step_failed", step="table_comparison", error=str(exc))
            result.all_flags.append(
                ValidationFlag(
                    step=PipelineStep.TABLE_COMPARISON,
                    severity=ValidationSeverity.ERROR,
                    code="STEP_FAILED",
                    message=f"Table comparison failed: {exc}",
                )
            )
        return result

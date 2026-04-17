"""Data models for the pipeline orchestration layer.

Key concept: The student does a systematic review where they:
  1. Define a search strategy and run it
  2. Screen results and select 6-10 papers for inclusion
  3. Extract data from those selected papers into tables
  4. Submit everything for integrity checking

This pipeline checks each of those steps.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from lit_inspector.core.enums import PipelineStep, ValidationSeverity
from lit_inspector.steps.data_extraction.schemas import ExtractedTable
from lit_inspector.steps.paper_verification.schemas import ReferenceEntry
from lit_inspector.steps.search_validation.schemas import SearchValidationResult
from lit_inspector.steps.paper_verification.schemas import ReferenceVerificationResult
from lit_inspector.steps.table_comparison.schemas import (
    EvaluationReport,
    TableComparisonResult,
)


class StudentReviewInput(BaseModel):
    """Input submitted by a student for integrity checking.

    Attributes:
        student_id: Student identifier.
        review_title: Title of the systematic review.
        search_strategy_text: Free-text description of the search strategy
            (e.g. "We searched PubMed using immunotherapy AND lung cancer...").
        search_database: Which database was searched (default: PubMed).
        reported_result_count: How many search results the student claims
            they found (e.g. 150). Used by Step 1 to compare.
        selected_papers: The final 6-10 papers the student selected after
            screening. These are the papers that go through Steps 2-4.
        extraction_fields: Which fields the student extracted from each paper
            (e.g. sample_size, study_design, outcome, p_value).
        submitted_tables: The student's own data extraction tables
            (one per selected paper). Used by Step 4 for comparison.
    """

    student_id: str
    review_title: str
    search_strategy_text: str = ""
    search_database: str = "PubMed"
    reported_result_count: int | None = None
    selected_papers: list[ReferenceEntry] = Field(default_factory=list)
    extraction_fields: list[str] = Field(default_factory=list)
    submitted_tables: list[ExtractedTable] = Field(default_factory=list)
    # --- Populated by Step 0 (PDF parser) or by the user ---
    review_full_text: str = ""  # full text of the review for LLM analysis


class ValidationFlag(BaseModel):
    """A flag raised at any point in the pipeline.

    Attributes:
        step: Which pipeline step raised this flag.
        severity: How serious the issue is.
        code: Machine-readable code (e.g. COUNT_MISMATCH).
        message: Human-readable description.
        details: Optional extra context.
    """

    step: PipelineStep
    severity: ValidationSeverity
    code: str
    message: str
    details: str = ""


class PipelineRunResult(BaseModel):
    """Complete result of a single pipeline run.

    Attributes:
        run_id: Unique identifier for this run.
        student_input: The original input.
        search_result: Step 1 output — search reproducibility check.
        papers_in_search: Step 1 check — which selected papers appear
            in the reproduced search results.
        verification_results: Step 2 output — one per selected paper.
        extracted_tables: Step 3 output — multiple extractors × papers.
        comparison_results: Step 4 comparison output.
        report: Step 4 final report.
        all_flags: Aggregated flags from all steps.
        started_at: When the pipeline run started.
        completed_at: When the pipeline run completed.
    """

    run_id: str
    student_input: StudentReviewInput
    # Step 1 LLM analysis: dynamically identified features
    research_context: str = ""  # e.g. "EAT thickness/volume in T1DM vs controls"
    identified_features: list[str] = Field(default_factory=list)  # e.g. ["mean", "sd", "n", "imaging_modality"]
    llm_search_query: str = ""  # LLM-constructed search query
    search_result: SearchValidationResult | None = None
    papers_in_search: dict[str, bool] = Field(default_factory=dict)
    verification_results: list[ReferenceVerificationResult] = Field(
        default_factory=list
    )
    extracted_tables: list[ExtractedTable] = Field(default_factory=list)
    # Names of all LLM extractors used (populated by Step 3 regardless of success)
    # Used by the report generator to ensure all LLM columns always appear.
    extractor_ids: list[str] = Field(default_factory=list)
    comparison_results: list[TableComparisonResult] = Field(default_factory=list)
    report: EvaluationReport | None = None
    all_flags: list[ValidationFlag] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None

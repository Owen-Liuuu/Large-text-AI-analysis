"""Mock implementations for step 4: table comparison and reporting."""
from __future__ import annotations

from lit_inspector.core.enums import (
    FieldStatus,
    ReportVerdict,
    ValidationSeverity,
)
from lit_inspector.steps.data_extraction.schemas import ExtractedTable
from lit_inspector.steps.table_comparison.interfaces import (
    ReportGenerator,
    TableComparator,
)
from lit_inspector.steps.table_comparison.schemas import (
    ComparisonFlag,
    EvaluationReport,
    FieldDiff,
    TableComparisonResult,
)


class MockTableComparator(TableComparator):
    """Returns fake comparison results for testing.

    Does NOT synthesise student fields from model data — if the student
    submits nothing, the paper is reported as skipped so the orchestrator
    and report can distinguish "student did not submit" from
    "student submitted and matched".
    """

    async def compare(
        self,
        student_table: ExtractedTable,
        model_tables: list[ExtractedTable],
    ) -> TableComparisonResult:
        student_fields = list(student_table.fields)

        if not student_fields and not model_tables:
            return TableComparisonResult(
                paper_id=student_table.paper_id,
                field_diffs=[],
                agreement_rate=0.0,
                coverage_rate=0.0,
                compared_count=0,
                total_count=0,
                flags=[],
                skipped=True,
            )

        diffs: list[FieldDiff] = []
        compared = 0
        coverable = 0
        agree = 0

        for field in student_fields:
            model_vals = []
            model_evidence = []
            for mt in model_tables:
                for f in mt.fields:
                    if f.field_name == field.field_name:
                        model_vals.append(f.value)
                        model_evidence.append(f.evidence or "")
            if field.value is not None and model_vals and any(v is not None for v in model_vals):
                status = FieldStatus.MATCH
                compared += 1
                coverable += 1
                agree += 1
                explanation = "Mock comparison: values match."
            elif field.value is None and not model_vals:
                status = FieldStatus.NOT_COMPARABLE
                explanation = "Mock comparison: both empty."
            elif field.value is None:
                status = FieldStatus.MISSING_STUDENT
                explanation = "Mock: student empty, model has value."
            else:
                status = FieldStatus.MISSING_MODEL
                explanation = "Mock: model empty, student has value."

            diffs.append(
                FieldDiff(
                    field_name=field.field_name,
                    student_value=field.value,
                    student_value_normalized=field.value,
                    model_values=model_vals,
                    model_values_normalized=list(model_vals),
                    status=status,
                    is_consistent=status in (FieldStatus.MATCH, FieldStatus.PARTIAL_MATCH),
                    explanation=explanation,
                    model_evidence=model_evidence,
                )
            )

        total = len(diffs)
        agreement = (agree / compared) if compared else 0.0
        coverage = (coverable / total) if total else 0.0

        return TableComparisonResult(
            paper_id=student_table.paper_id,
            field_diffs=diffs,
            agreement_rate=agreement if compared else 0.85,
            coverage_rate=coverage,
            compared_count=compared,
            total_count=total,
            flags=[
                ComparisonFlag(
                    code="MOCK_MINOR_DIFF",
                    severity=ValidationSeverity.INFO,
                    message="Minor differences detected in mock comparison.",
                )
            ],
            skipped=total == 0,
        )


class MockReportGenerator(ReportGenerator):
    """Returns a fake evaluation report for testing."""

    async def generate(
        self,
        comparison_results: list[TableComparisonResult],
        run_id: str,
    ) -> EvaluationReport:
        all_flags = []
        for cr in comparison_results:
            all_flags.extend(cr.flags)

        total = len(comparison_results)
        compared_results = [cr for cr in comparison_results if not cr.skipped]
        skipped = total - len(compared_results)

        avg_agreement = (
            sum(cr.agreement_rate for cr in compared_results) / len(compared_results)
            if compared_results else 0.0
        )
        avg_coverage = (
            sum(cr.coverage_rate for cr in comparison_results) / total
            if total else 0.0
        )

        return EvaluationReport(
            run_id=run_id,
            comparison_results=comparison_results,
            overall_flags=all_flags,
            summary=(
                f"Mock evaluation complete. "
                f"Reviewed {total} paper(s). "
                f"Average agreement rate: {avg_agreement:.1%}."
            ),
            verdict=ReportVerdict.PASS if compared_results else ReportVerdict.INCOMPLETE,
            avg_agreement=avg_agreement,
            avg_coverage=avg_coverage,
            compared_papers=len(compared_results),
            skipped_papers=skipped,
        )

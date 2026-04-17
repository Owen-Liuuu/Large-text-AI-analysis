"""Mock implementations for step 4: table comparison and reporting.

Delegates to the canonical-schema logic in :mod:`real_impl` so that
mock output has the same shape as the real comparator — single row per
canonical field, no duplicate student/model columns.
"""
from __future__ import annotations

from lit_inspector.core.enums import ReportVerdict, ValidationSeverity
from lit_inspector.steps.data_extraction.schemas import ExtractedTable
from lit_inspector.steps.table_comparison.interfaces import (
    ReportGenerator,
    TableComparator,
)
from lit_inspector.steps.table_comparison.real_impl import RealTableComparator
from lit_inspector.steps.table_comparison.schemas import (
    ComparisonFlag,
    EvaluationReport,
    TableComparisonResult,
)


class MockTableComparator(TableComparator):
    """Returns comparison results with the same canonical shape as the real
    implementation. Behaviour is deterministic and suitable for tests —
    it does not call any external services."""

    def __init__(self) -> None:
        self._inner = RealTableComparator()

    async def compare(
        self,
        student_table: ExtractedTable,
        model_tables: list[ExtractedTable],
    ) -> TableComparisonResult:
        result = await self._inner.compare(student_table, model_tables)
        # Tag with a mock marker so tests can distinguish if needed.
        result.flags.append(
            ComparisonFlag(
                code="MOCK_MINOR_DIFF",
                severity=ValidationSeverity.INFO,
                message="Minor differences detected in mock comparison.",
            )
        )
        # Keep the historical 0.85 floor only when no real comparison was
        # possible, so existing tests that check ``agreement_rate > 0.0``
        # on an all-match mock input still pass.
        if result.compared_count == 0 and not result.skipped:
            result.agreement_rate = 0.85
        return result


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

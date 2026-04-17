"""Real implementations for step 4: table comparison and reporting.

Compares student-submitted extraction tables with AI-generated tables
on a field-by-field basis.

Key design properties (post-refactor):

  * Extractor failures are never blamed on the student. When the model
    has no value, the field status becomes MISSING_MODEL and the flag
    code is EXTRACTOR_GAP rather than FIELD_MISMATCH.
  * Values are normalised before comparison: author strings, measurement
    tool synonyms, numeric formatting ("52.3±8.1", "52.3 (36.1-65.5)")
    and general text.
  * Group-aware: fields like ``age_t1dm`` / ``age_control`` are only
    compared against model fields for the same group; they do not cross
    over into the wrong cohort.
  * Coverage and agreement are counted separately. "How many fields
    could we compare at all?" is independent of "and of those, how many
    agreed?".
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from lit_inspector.core.enums import (
    ComparisonFlagCode,
    FieldStatus,
    ReportVerdict,
    ValidationSeverity,
)
from lit_inspector.steps.data_extraction.schemas import ExtractedField, ExtractedTable
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


# ======================================================================
# Field name normalisation & matching
# ======================================================================

# Common abbreviations / synonyms found in systematic review tables
_FIELD_ALIASES: dict[str, list[str]] = {
    "author": ["author", "authors", "first_author", "first author"],
    "country": ["country", "location", "region", "nation"],
    "sample_size": ["n", "sample_size", "sample size", "total_n", "total n",
                    "participants", "number of participants", "subjects"],
    "age_mean": ["age", "age_mean", "mean_age", "age mean", "mean age",
                 "age (years)", "age(years)"],
    "bmi": ["bmi", "bmi_kg_m2", "bmi kg/m2", "bmi (kg/m2)", "body_mass_index",
            "body mass index"],
    "measurement_tool": ["measurement_tool", "measurement tool", "imaging",
                         "imaging_modality", "modality", "method"],
    "group": ["group", "groups", "study_group", "study group"],
    "eat_measurement": ["eft", "eat", "eft_eat", "eft/ eat", "eft/eat",
                        "eat_thickness", "eat_volume", "epicardial_fat",
                        "epicardial adipose tissue", "eat measurement",
                        "eft/ eat measurement"],
    "overall_quality": ["overall_quality", "overall quality", "quality",
                        "quality_score", "nos", "nos score", "risk_of_bias",
                        "risk of bias", "quality assessment"],
    "study_design": ["study_design", "study design", "design", "study_type",
                     "study type"],
    "intervention": ["intervention", "treatment", "drug", "exposure"],
    "primary_outcome": ["primary_outcome", "primary outcome", "outcome",
                        "main_outcome", "endpoint", "primary_endpoint"],
    "p_value": ["p_value", "p value", "p-value", "pvalue", "significance"],
    "hazard_ratio": ["hr", "hazard_ratio", "hazard ratio"],
    "overall_survival_hr": ["overall_survival_hr", "os_hr", "overall survival hr"],
    "confidence_interval": ["ci", "confidence_interval", "confidence interval",
                            "95_ci", "95% ci"],
    "intervention_group_mean": ["intervention_group_mean", "t1dm_mean",
                                 "case_mean", "patient_mean", "disease_mean",
                                 "experimental_mean"],
    "intervention_group_sd": ["intervention_group_sd", "t1dm_sd",
                               "case_sd", "patient_sd", "disease_sd",
                               "experimental_sd"],
    "intervention_group_n": ["intervention_group_n", "t1dm_n",
                              "case_n", "patient_n", "disease_n",
                              "experimental_n"],
    "control_group_mean": ["control_group_mean", "control_mean",
                            "healthy_mean", "normal_mean"],
    "control_group_sd": ["control_group_sd", "control_sd",
                          "healthy_sd", "normal_sd"],
    "control_group_n": ["control_group_n", "control_n",
                         "healthy_n", "normal_n"],
    # Group-aware fields
    "age_t1dm": ["age_t1dm", "age_case", "age_patient", "age_disease"],
    "age_control": ["age_control", "age_healthy", "age_normal"],
    "bmi_t1dm": ["bmi_t1dm", "bmi_case", "bmi_patient"],
    "bmi_control": ["bmi_control", "bmi_healthy", "bmi_normal"],
    "eft_t1dm": ["eft_t1dm", "eat_t1dm", "eft_case", "eat_case"],
    "eft_control": ["eft_control", "eat_control", "eft_healthy", "eat_healthy"],
}

# Group tags that a field name might carry. Order matters — longer /
# more-specific tokens first so "t1dm" isn't accidentally matched as "t1".
_GROUP_TAGS: dict[str, list[str]] = {
    "t1dm": ["t1dm", "case", "patient", "patients", "disease", "intervention"],
    "control": ["control", "controls", "healthy", "normal"],
}


def _normalise_field_name(name: str) -> str:
    """Normalise a field name to lowercase snake_case for matching."""
    s = name.lower().strip()
    s = re.sub(r"[/\-–—()（）\[\]]", "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s


def _extract_group(normalised_name: str) -> str | None:
    """Return the group tag ("t1dm" / "control") if the name has one."""
    tokens = set(normalised_name.split("_"))
    for group, markers in _GROUP_TAGS.items():
        if any(m in tokens for m in markers):
            return group
    return None


def _match_field_name(
    student_name: str,
    model_names: list[str],
    threshold: float = 0.60,
) -> str | None:
    """Find the best matching model field name for a student field name.

    Group-aware: a student field tagged ``t1dm`` will never match a
    model field tagged ``control`` and vice versa.
    """
    s_norm = _normalise_field_name(student_name)
    s_group = _extract_group(s_norm)

    # Pre-filter model names by group compatibility
    def _compatible(m_norm: str) -> bool:
        m_group = _extract_group(m_norm)
        if s_group is None or m_group is None:
            return True
        return s_group == m_group

    m_norms = {
        _normalise_field_name(m): m
        for m in model_names
        if _compatible(_normalise_field_name(m))
    }

    # 1. Exact normalised match
    if s_norm in m_norms:
        return m_norms[s_norm]

    # 2. Alias lookup
    s_canonical = None
    for canon, aliases in _FIELD_ALIASES.items():
        normed_aliases = [_normalise_field_name(a) for a in aliases]
        if s_norm in normed_aliases or s_norm == canon:
            s_canonical = canon
            break

    if s_canonical:
        if s_canonical in m_norms:
            return m_norms[s_canonical]
        canon_aliases = [_normalise_field_name(a) for a in _FIELD_ALIASES.get(s_canonical, [])]
        for m_norm, m_orig in m_norms.items():
            if m_norm in canon_aliases:
                return m_orig

    for canon, aliases in _FIELD_ALIASES.items():
        normed_aliases = [_normalise_field_name(a) for a in aliases]
        for m_norm, m_orig in m_norms.items():
            if m_norm in normed_aliases or m_norm == canon:
                if s_norm in normed_aliases or s_norm == canon:
                    return m_orig

    # 3. Substring / containment match — but only within compatible groups
    for m_norm, m_orig in m_norms.items():
        if len(s_norm) >= 3 and len(m_norm) >= 3:
            if s_norm in m_norm or m_norm in s_norm:
                return m_orig
        s_tokens = set(s_norm.split("_"))
        m_tokens = set(m_norm.split("_"))
        trivial = {"", "kg", "m2", "mm", "cm", "ml", "mean", "sd", "median"}
        s_key = s_tokens - trivial
        m_key = m_tokens - trivial
        if s_key and m_key and s_key & m_key:
            return m_orig

    # 4. Fuzzy
    best_sim = 0.0
    best_match: str | None = None
    for m_norm, m_orig in m_norms.items():
        sim = SequenceMatcher(None, s_norm, m_norm).ratio()
        if sim > best_sim:
            best_sim = sim
            best_match = m_orig
    if best_sim >= threshold:
        return best_match

    return None


# ======================================================================
# Value normalisation
# ======================================================================

_AUTHOR_ETAL_RE = re.compile(
    r"^\s*([A-Za-zÀ-ÿ'\-]+)\s+(?:et\s*al\.?|and\s+colleagues|&\s*al)",
    re.IGNORECASE,
)
_YEAR_TRAILER_RE = re.compile(r"[\[\(]\s*(19|20)\d{2}\s*[\]\)]\s*$")

_TOOL_SYNONYMS: dict[str, set[str]] = {
    "echocardiography": {
        "echo", "echocardiography", "echocardiogram", "echocardiographic",
        "tte", "transthoracic echocardiography", "2d echo", "2d echocardiography",
    },
    "ccta": {
        "ccta", "cardiac ct", "coronary ct angiography", "cardiac cta",
        "ct angiography", "cardiac computed tomography angiography",
    },
    "cmr": {
        "cmr", "mri", "cardiac mri", "cardiovascular magnetic resonance",
        "cardiac magnetic resonance", "cardiac mr",
    },
    "ct": {"ct", "computed tomography", "cardiac ct scan"},
}


def _normalise_author(value: str) -> str:
    """Collapse "Ahmad et al. [2022]" / full author lists to a canonical key.

    We extract the first author's surname (lower-cased) and optionally
    a trailing year if present. Full author lists like
    "Ahmad, Iacobellis, Hussain" reduce to "ahmad".
    """
    s = value.strip()
    # Strip enclosing quotes
    s = s.strip("\"' ")
    # Year trailer
    year_match = re.search(r"(19|20)\d{2}", s)
    year = year_match.group(0) if year_match else ""
    # "Ahmad et al." — take the word before "et al"
    m = _AUTHOR_ETAL_RE.match(s)
    if m:
        surname = m.group(1).lower()
    else:
        # Full list: "Ahmad, Iacobellis, Hussain" → first token
        first = re.split(r"[,;&/]|\band\b", s, maxsplit=1)[0]
        # If comma order was "Surname, First", take first token before space
        # If no comma, take first alphabetic token
        tokens = re.findall(r"[A-Za-zÀ-ÿ'\-]+", first)
        surname = tokens[0].lower() if tokens else s.lower()
    key = surname
    if year:
        key = f"{surname} {year}"
    return key.strip()


def _normalise_tool(value: str) -> str:
    """Map a measurement tool string to its canonical synonym."""
    s = value.strip().lower()
    # Strip units / parentheses tails like "Echocardiography (2D)"
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    for canonical, syns in _TOOL_SYNONYMS.items():
        if s in syns:
            return canonical
        for syn in syns:
            if syn in s:
                return canonical
    return s


def _normalise_numeric(value: str) -> str | None:
    """Canonicalise numeric formats.

    Accepts: "52.3±8.1", "52.3 ± 8.1", "52.3+/-8.1", "52.3 (36.1-65.5)",
             "52.3", "52,3", "52.3%" → "52.3".
    Returns the leading numeric token as a canonical string, or None if
    no number is present.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Normalise unicode / typographic variants
    s = s.replace(",", ".").replace("±", "+-").replace("–", "-").replace("—", "-")
    s = s.replace("+/-", "+-")
    # Find the first numeric token
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    num = m.group(0)
    # Strip trailing ".0" to stabilise "52" vs "52.0"
    try:
        f = float(num)
    except ValueError:
        return num
    if f == int(f):
        return str(int(f))
    return f"{f:.6g}"


def _normalise_text(value: str) -> str:
    """General text normalisation: lowercase, collapse whitespace, strip
    punctuation edges."""
    s = value.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,;:!?\"'()-_/[]")
    return s


def _normalise_value(value: object, field_name: str) -> object:
    """Dispatch to the right normaliser based on the field name."""
    if value is None:
        return None
    # Bools / lists pass through
    if isinstance(value, (bool, list)):
        return value
    s = str(value)
    f_norm = _normalise_field_name(field_name)

    if "author" in f_norm:
        return _normalise_author(s)
    if "tool" in f_norm or "modal" in f_norm or "imaging" in f_norm or "method" in f_norm:
        return _normalise_tool(s)
    # Numeric-ish fields
    if any(tok in f_norm for tok in (
        "age", "bmi", "size", "_n", "mean", "sd",
        "eft", "eat", "year", "duration",
        "hr", "ratio", "p_value", "pvalue",
    )):
        n = _normalise_numeric(s)
        if n is not None:
            return n
    # Pure-numeric input
    if isinstance(value, (int, float)):
        n = _normalise_numeric(str(value))
        if n is not None:
            return n
    return _normalise_text(s)


def _try_float(v: object) -> float | None:
    """Try to parse a value as a float."""
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        n = _normalise_numeric(v)
        if n is None:
            return None
        try:
            return float(n)
        except ValueError:
            return None
    return None


def _compare_normalised(
    s_norm: object,
    m_norm: object,
    *,
    numeric_tolerance: float = 0.01,
    strong_text_threshold: float = 0.90,
    partial_text_threshold: float = 0.70,
) -> tuple[FieldStatus, str]:
    """Compare two already-normalised values. Caller handles None semantics."""
    # Numeric comparison first
    s_num = _try_float(s_norm)
    m_num = _try_float(m_norm)
    if s_num is not None and m_num is not None:
        denom = max(abs(s_num), abs(m_num), 1.0)
        rel = abs(s_num - m_num) / denom
        if rel <= numeric_tolerance:
            return FieldStatus.MATCH, f"numeric match ({s_num}≈{m_num})"
        if rel <= 10 * numeric_tolerance:
            return FieldStatus.PARTIAL_MATCH, (
                f"numeric close ({s_num} vs {m_num}, rel={rel:.2%})"
            )
        return FieldStatus.DIFF, f"numeric DIFF ({s_num} vs {m_num})"

    # Text comparison
    s_str = str(s_norm) if s_norm is not None else ""
    m_str = str(m_norm) if m_norm is not None else ""
    if s_str == m_str and s_str:
        return FieldStatus.MATCH, "exact match"
    sim = SequenceMatcher(None, s_str, m_str).ratio() if (s_str or m_str) else 0.0
    if sim >= strong_text_threshold:
        return FieldStatus.MATCH, f"text match (sim={sim:.2f})"
    if sim >= partial_text_threshold:
        return FieldStatus.PARTIAL_MATCH, f"text close (sim={sim:.2f})"
    return FieldStatus.DIFF, (
        f"text DIFF (sim={sim:.2f}): '{s_str}' vs '{m_str}'"
    )


# ======================================================================
# Comparator
# ======================================================================


class RealTableComparator(TableComparator):
    """Compare a student table against model-extracted tables."""

    async def compare(
        self,
        student_table: ExtractedTable,
        model_tables: list[ExtractedTable],
    ) -> TableComparisonResult:
        flags: list[ComparisonFlag] = []
        diffs: list[FieldDiff] = []

        # Collect distinct model field names for matching.
        all_model_field_names: list[str] = []
        for mt in model_tables:
            for mf in mt.fields:
                if mf.field_name not in all_model_field_names:
                    all_model_field_names.append(mf.field_name)

        # Decide the universe of fields to report on: union of what the
        # student submitted and what any model extracted. We do NOT fill
        # in empty student fields here — empty on the student side must
        # remain empty so MISSING_STUDENT can be detected downstream.
        student_by_name: dict[str, ExtractedField] = {
            f.field_name: f for f in student_table.fields
        }
        universe: list[str] = []
        for name in student_by_name.keys():
            if name not in universe:
                universe.append(name)
        for name in all_model_field_names:
            if name not in universe:
                universe.append(name)

        # If the comparator has literally nothing to work with, mark as
        # skipped so the report verdict can factor this in.
        if not universe:
            return TableComparisonResult(
                paper_id=student_table.paper_id,
                field_diffs=[],
                agreement_rate=0.0,
                coverage_rate=0.0,
                compared_count=0,
                total_count=0,
                flags=[
                    ComparisonFlag(
                        code=ComparisonFlagCode.NO_TABLES.value,
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"Paper {student_table.paper_id}: "
                            "no student or model fields available."
                        ),
                    )
                ],
                skipped=True,
            )

        missing_model_fields: list[str] = []
        mismatch_fields: list[str] = []
        compared_count = 0
        coverable_count = 0  # fields where both sides have a value
        agree_count = 0

        for field_name in universe:
            s_field = student_by_name.get(field_name)

            # Gather model values for the matched field name.
            matched_model_name: str | None
            if field_name in all_model_field_names:
                matched_model_name = field_name
            elif s_field is not None:
                matched_model_name = _match_field_name(
                    s_field.field_name, all_model_field_names
                )
            else:
                matched_model_name = None

            model_vals: list[object] = []
            model_evidence: list[str] = []
            any_extractor_failed = False
            if matched_model_name:
                for mt in model_tables:
                    for mf in mt.fields:
                        if mf.field_name == matched_model_name:
                            model_vals.append(mf.value)
                            model_evidence.append(mf.evidence or "")
                            if mf.extractor_failed:
                                any_extractor_failed = True

            student_val = s_field.value if s_field is not None else None
            student_has = student_val is not None and (
                not isinstance(student_val, str) or student_val.strip() != ""
            )
            model_non_null_vals = [v for v in model_vals if v is not None
                                   and (not isinstance(v, str) or v.strip() != "")]
            model_has = bool(model_non_null_vals)

            # Status / accounting
            if not student_has and not model_has:
                status = FieldStatus.NOT_COMPARABLE
                explanation = "both sides empty"
                student_norm: object = None
                model_norms: list[object] = [None] * len(model_vals)
            elif not model_has:
                # Extractor gap — never blame the student
                status = FieldStatus.MISSING_MODEL
                if any_extractor_failed:
                    explanation = "extractor failure — value not produced by model"
                else:
                    explanation = "model has no value (extractor gap)"
                missing_model_fields.append(field_name)
                student_norm = _normalise_value(student_val, field_name)
                model_norms = [None] * len(model_vals)
            elif not student_has:
                status = FieldStatus.MISSING_STUDENT
                explanation = "student has no value; model provided one"
                student_norm = None
                model_norms = [_normalise_value(v, field_name) for v in model_vals]
            else:
                student_norm = _normalise_value(student_val, field_name)
                model_norms = [_normalise_value(v, field_name) for v in model_vals]
                # Pick the best comparison across all model values
                best_status = FieldStatus.DIFF
                best_expl = ""
                status_rank = {
                    FieldStatus.MATCH: 3,
                    FieldStatus.PARTIAL_MATCH: 2,
                    FieldStatus.DIFF: 1,
                }
                for mn in model_norms:
                    if mn is None:
                        continue
                    st, expl = _compare_normalised(student_norm, mn)
                    if status_rank.get(st, 0) > status_rank.get(best_status, 0):
                        best_status = st
                        best_expl = expl
                    elif not best_expl:
                        best_expl = expl
                status = best_status
                explanation = best_expl
                coverable_count += 1
                compared_count += 1
                if status in (FieldStatus.MATCH, FieldStatus.PARTIAL_MATCH):
                    agree_count += 1
                else:
                    mismatch_fields.append(field_name)

            is_consistent = status in (FieldStatus.MATCH, FieldStatus.PARTIAL_MATCH)

            diffs.append(
                FieldDiff(
                    field_name=field_name,
                    student_value=student_val,
                    student_value_normalized=student_norm,
                    model_values=model_vals,
                    model_values_normalized=model_norms,
                    status=status,
                    is_consistent=is_consistent,
                    explanation=explanation,
                    model_evidence=model_evidence,
                )
            )

        total = len(diffs)
        agreement_rate = (agree_count / compared_count) if compared_count > 0 else 0.0
        coverage_rate = (coverable_count / total) if total > 0 else 0.0

        # FIELD_MISMATCH: only fields where BOTH sides had values and still differ.
        # This must exclude MISSING_MODEL fields — those go under EXTRACTOR_GAP.
        if mismatch_fields:
            names = ", ".join(mismatch_fields[:5])
            flags.append(
                ComparisonFlag(
                    code=ComparisonFlagCode.FIELD_MISMATCH.value,
                    severity=(
                        ValidationSeverity.ERROR
                        if compared_count > 0 and agreement_rate < 0.5
                        else ValidationSeverity.WARNING
                    ),
                    message=(
                        f"Paper {student_table.paper_id}: "
                        f"{len(mismatch_fields)}/{compared_count} fields differ ({names})"
                    ),
                )
            )

        if missing_model_fields:
            names = ", ".join(missing_model_fields[:5])
            flags.append(
                ComparisonFlag(
                    code=ComparisonFlagCode.EXTRACTOR_GAP.value,
                    severity=ValidationSeverity.INFO,
                    message=(
                        f"Paper {student_table.paper_id}: "
                        f"model missing {len(missing_model_fields)} field(s) "
                        f"({names}) — not a student error."
                    ),
                )
            )

        # If the paper couldn't be compared at all (no overlapping values),
        # mark it as skipped.
        skipped = compared_count == 0

        return TableComparisonResult(
            paper_id=student_table.paper_id,
            field_diffs=diffs,
            agreement_rate=agreement_rate,
            coverage_rate=coverage_rate,
            compared_count=compared_count,
            total_count=total,
            flags=flags,
            skipped=skipped,
        )


# ======================================================================
# Report generator
# ======================================================================


class RealReportGenerator(ReportGenerator):
    """Generates a final evaluation report from comparison results.

    Verdict rules (applied top-down, first match wins):
      - skipped > 0 AND compared == 0 → INCOMPLETE
      - skipped > 0 OR avg_coverage < 0.5 → PARTIAL
      - avg_agreement < 0.6 → FAIL
      - else PASS
    """

    async def generate(
        self,
        comparison_results: list[TableComparisonResult],
        run_id: str,
    ) -> EvaluationReport:
        all_flags: list[ComparisonFlag] = []
        for cr in comparison_results:
            all_flags.extend(cr.flags)

        if not comparison_results:
            return EvaluationReport(
                run_id=run_id,
                comparison_results=[],
                overall_flags=[],
                summary="No papers to compare. Step 3 data not available.",
                verdict=ReportVerdict.INCOMPLETE,
            )

        compared_results = [cr for cr in comparison_results if not cr.skipped]
        skipped_count = len(comparison_results) - len(compared_results)
        total = len(comparison_results)

        avg_agreement = (
            sum(cr.agreement_rate for cr in compared_results) / len(compared_results)
            if compared_results else 0.0
        )
        avg_coverage = sum(cr.coverage_rate for cr in comparison_results) / total

        # Verdict
        if skipped_count > 0 and len(compared_results) == 0:
            verdict = ReportVerdict.INCOMPLETE
            detail = (
                f"{skipped_count}/{total} paper(s) could not be compared "
                "(extractor produced nothing comparable)."
            )
        elif skipped_count > 0 or avg_coverage < 0.5:
            verdict = ReportVerdict.PARTIAL
            detail = (
                f"Coverage is limited (avg={avg_coverage:.0%}, "
                f"skipped={skipped_count}/{total}). Agreement where "
                f"comparable: {avg_agreement:.0%}."
            )
        elif avg_agreement < 0.6:
            verdict = ReportVerdict.FAIL
            detail = f"Average agreement {avg_agreement:.0%} is below 60%."
        else:
            verdict = ReportVerdict.PASS
            detail = (
                f"All comparable papers show acceptable agreement "
                f"(avg={avg_agreement:.0%}, coverage={avg_coverage:.0%})."
            )

        summary = (
            f"[{verdict.value}] Reviewed {total} paper(s) "
            f"(compared={len(compared_results)}, skipped={skipped_count}). "
            f"Average agreement: {avg_agreement:.0%}. "
            f"Average coverage: {avg_coverage:.0%}. {detail}"
        )

        return EvaluationReport(
            run_id=run_id,
            comparison_results=comparison_results,
            overall_flags=all_flags,
            summary=summary,
            verdict=verdict,
            avg_agreement=avg_agreement,
            avg_coverage=avg_coverage,
            compared_papers=len(compared_results),
            skipped_papers=skipped_count,
        )

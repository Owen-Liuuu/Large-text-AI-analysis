"""Real implementations for step 4: table comparison and reporting.

Compares student-submitted extraction tables with AI-generated tables
on a field-by-field basis using a *canonical schema*.

Post-refactor properties:

  * One FieldDiff per canonical (base, group). Student raw names like
    "N", "EFT/ EAT", "Age" are mapped to canonical names like
    "sample_size", "eat_or_eft_t1dm", "age_t1dm". No duplicate rows.
  * Extractor failures do not become student errors (MISSING_MODEL +
    EXTRACTOR_GAP, never FIELD_MISMATCH).
  * Author normalisation handles particles ("de Gonzalo-Calvo") and
    year-optional matching.
  * Measurement tool synonyms include verbose spellings
    ("Coronary computed tomography angiography" → "ccta").
  * Group-aware: single-group student values are reconciled against
    dual-group model outputs. When the assignment is ambiguous, status
    is NEEDS_REVIEW rather than MATCH/DIFF.
  * Coverage and agreement are tracked separately; the summary makes
    the limitation explicit when coverage is low.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from dataclasses import dataclass, field

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
# Canonical field schema
# ======================================================================
#
# Each raw field name (from student headers or model outputs) maps to a
# canonical (base, group) pair. The comparator builds one FieldDiff per
# canonical key so the report shows one unified row per logical field.

# Group markers. Order matters: longer / more-specific tokens first.
_GROUP_TAGS: dict[str, list[str]] = {
    "t1dm": ["t1dm", "t1d", "case", "cases", "patient", "patients",
             "disease", "intervention"],
    "control": ["control", "controls", "healthy", "normal"],
}

# Normalised alias → canonical base
_ALIAS_TO_BASE: dict[str, str] = {
    # author
    "author": "author", "authors": "author", "first_author": "author",
    # country
    "country": "country", "location": "country", "region": "country",
    "nation": "country",
    # year
    "year": "year", "publication_year": "year",
    # sample size
    "n": "sample_size", "sample_size": "sample_size",
    "total_n": "sample_size", "participants": "sample_size",
    "subjects": "sample_size", "number_of_participants": "sample_size",
    # age
    "age": "age", "age_mean": "age", "mean_age": "age",
    "age_years": "age",
    # bmi
    "bmi": "bmi", "bmi_kg_m2": "bmi", "bmi_mean": "bmi",
    "body_mass_index": "bmi", "bmi_kg": "bmi",
    # eat / eft (unified canonical base)
    "eft": "eat_or_eft", "eat": "eat_or_eft", "eft_eat": "eat_or_eft",
    "eat_thickness": "eat_or_eft", "eat_volume": "eat_or_eft",
    "eat_volume_mean": "eat_or_eft", "eat_measurement": "eat_or_eft",
    "eat_or_eft": "eat_or_eft", "epicardial_fat": "eat_or_eft",
    "epicardial_adipose_tissue": "eat_or_eft",
    "eft_measurement": "eat_or_eft",
    # measurement tool
    "measurement_tool": "measurement_tool",
    "imaging_modality": "measurement_tool",
    "modality": "measurement_tool", "method": "measurement_tool",
    "imaging": "measurement_tool",
    "imaging_method": "measurement_tool",
    # study design
    "study_design": "study_design", "design": "study_design",
    "study_type": "study_design",
    # quality
    "overall_quality": "overall_quality", "quality": "overall_quality",
    "nos": "overall_quality", "nos_score": "overall_quality",
    "risk_of_bias": "overall_quality",
    "quality_score": "overall_quality",
    "quality_assessment": "overall_quality",
    # group label (string describing cohort)
    "group": "group", "groups": "group", "study_group": "group",
    # stats (pass-through canonicals)
    "p_value": "p_value", "p": "p_value", "pvalue": "p_value",
    "hazard_ratio": "hazard_ratio", "hr": "hazard_ratio",
    "confidence_interval": "confidence_interval",
    "ci": "confidence_interval",
}


def _normalise_field_name(name: str) -> str:
    """Normalise a field name to lowercase snake_case."""
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


def _strip_group_tokens(normalised_name: str) -> str:
    """Remove group-tag tokens so we can look up the base alias."""
    all_markers: set[str] = set()
    for markers in _GROUP_TAGS.values():
        all_markers.update(markers)
    tokens = [t for t in normalised_name.split("_") if t and t not in all_markers]
    return "_".join(tokens)


def _canonical_key(raw_name: str) -> tuple[str, str | None]:
    """Map a raw field name to ``(canonical_base, group_tag)``.

    - "N" → ("sample_size", None)
    - "Age" → ("age", None)
    - "age_t1dm" → ("age", "t1dm")
    - "eat_volume_mean" → ("eat_or_eft", None)
    - "Measurement tool" → ("measurement_tool", None)
    """
    s_norm = _normalise_field_name(raw_name)
    group = _extract_group(s_norm)
    stripped = _strip_group_tokens(s_norm)

    # Direct alias lookup (group-stripped first, then full)
    if stripped in _ALIAS_TO_BASE:
        return (_ALIAS_TO_BASE[stripped], group)
    if s_norm in _ALIAS_TO_BASE:
        return (_ALIAS_TO_BASE[s_norm], group)

    # Partial / containment alias match
    for alias, base in _ALIAS_TO_BASE.items():
        if len(alias) >= 3 and len(stripped) >= 3:
            if alias in stripped or stripped in alias:
                return (base, group)

    # Token-overlap fallback
    s_tokens = set(stripped.split("_")) - {""}
    trivial = {"kg", "m2", "mm", "cm", "ml", "mean", "sd", "median", "value"}
    s_key = s_tokens - trivial
    if s_key:
        for alias, base in _ALIAS_TO_BASE.items():
            a_tokens = set(alias.split("_")) - trivial
            if a_tokens and s_key & a_tokens:
                return (base, group)

    # Unknown: use the stripped (or full) name as its own canonical base.
    return (stripped if stripped else s_norm, group)


def _canonical_display_name(base: str, group: str | None) -> str:
    """Build the display field name for the report table."""
    if group:
        return f"{base}_{group}"
    return base


# ======================================================================
# Value normalisation
# ======================================================================

_TOOL_SYNONYMS: dict[str, set[str]] = {
    # Order matters for substring matching (first match wins).
    "echocardiography": {
        "echo", "echocardiography", "echocardiogram", "echocardiographic",
        "tte", "transthoracic echocardiography", "2d echo",
        "2d echocardiography", "transthoracic echo", "transthoracic",
    },
    "ccta": {
        "ccta", "cardiac cta",
        "cardiac computed tomography angiography",
        "coronary computed tomography angiography",
        "coronary ct angiography",
        "ct angiography", "computed tomography angiography",
        "cardiac ct", "coronary ct",
    },
    "cmr": {
        "cmr", "mri", "cardiac mri", "cardiovascular magnetic resonance",
        "cardiac magnetic resonance", "cardiac mr",
    },
    # Plain CT last — only used when nothing more specific matches.
    "ct": {"cardiac ct scan", "computed tomography"},
}


def _normalise_author(value: str) -> str:
    """Collapse various author-string formats to a surname (+ optional year).

    Keeps multi-token surnames intact (so "de Gonzalo-Calvo" stays
    "de gonzalo-calvo"), strips first initials, "et al" suffixes, and
    bracketed / parenthesised years.
    """
    if not value:
        return ""
    s = str(value).strip().strip("\"' ")
    # Year
    year_match = re.search(r"(19|20)\d{2}", s)
    year = year_match.group(0) if year_match else ""
    # Strip (year) / [year]
    s = re.sub(r"[\[\(]\s*(?:19|20)\d{2}\s*[\]\)]", "", s).strip(" .,")
    # Strip " et al" / "and colleagues" and everything after
    s = re.sub(
        r"\s*(?:et\s*al\.?|and\s+colleagues|&\s*al\.?).*$",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()
    # Strip trailing ", F." / ", F. M." (first-name initials)
    s = re.sub(r",\s*[A-Z]\.?(?:[\s\-][A-Z]\.?)*\s*$", "", s).strip()
    # If comma-separated list (coauthors), take the first segment
    if "," in s:
        s = s.split(",")[0].strip()
    # Drop trailing bare initials ("Smith J" → "Smith")
    tokens = s.split()
    while tokens and re.fullmatch(r"[A-Za-z]\.?", tokens[-1]):
        tokens.pop()
    surname = " ".join(tokens).lower() if tokens else s.lower()
    surname = surname.strip(" .,")
    if year:
        return f"{surname} {year}"
    return surname


def _normalise_tool(value: str) -> str:
    """Map a measurement tool string to its canonical synonym."""
    s = value.strip().lower()
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".,;")
    for canonical, syns in _TOOL_SYNONYMS.items():
        if s in syns:
            return canonical
        for syn in syns:
            if syn in s:
                return canonical
    return s


def _normalise_numeric(value: object) -> str | None:
    """Canonicalise numeric formats.

    "52.3±8.1", "52.3 ± 8.1", "52.3+/-8.1", "52.3 (36.1-65.5)",
    "52,3", "52.3%" → "52.3".  "52" → "52".  Returns None if no number.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.replace(",", ".").replace("±", "+-").replace("–", "-").replace("—", "-")
    s = s.replace("+/-", "+-")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    num = m.group(0)
    try:
        f = float(num)
    except ValueError:
        return num
    if f == int(f):
        return str(int(f))
    return f"{f:.6g}"


def _normalise_text(value: str) -> str:
    """General text normalisation."""
    s = value.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,;:!?\"'()-_/[]")
    return s


# Bases where we normalise numerically (the group suffix is stripped
# before dispatch, so we look at the base).
_NUMERIC_BASES: set[str] = {
    "age", "bmi", "sample_size", "eat_or_eft", "year",
    "hazard_ratio", "p_value",
}


def _normalise_value(value: object, base: str) -> object:
    """Dispatch normalisation based on the canonical base."""
    if value is None:
        return None
    if isinstance(value, (bool, list)):
        return value
    s = str(value)

    if base == "author":
        return _normalise_author(s)
    if base == "measurement_tool":
        return _normalise_tool(s)
    if base in _NUMERIC_BASES:
        n = _normalise_numeric(s)
        if n is not None:
            return n
        # Fall through to text for non-numeric outlier strings
    if isinstance(value, (int, float)) and not isinstance(value, bool):
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


# ======================================================================
# Pair-wise comparison primitives
# ======================================================================


def _compare_numeric(s_num: float, m_num: float) -> tuple[FieldStatus, str]:
    denom = max(abs(s_num), abs(m_num), 1.0)
    rel = abs(s_num - m_num) / denom
    if rel <= 0.01:
        return FieldStatus.MATCH, f"numeric match ({s_num}≈{m_num})"
    if rel <= 0.10:
        return FieldStatus.PARTIAL_MATCH, (
            f"numeric close ({s_num} vs {m_num}, rel={rel:.2%})"
        )
    return FieldStatus.DIFF, f"numeric DIFF ({s_num} vs {m_num})"


def _compare_text(s: str, m: str) -> tuple[FieldStatus, str]:
    if not s and not m:
        return FieldStatus.NOT_COMPARABLE, "both empty"
    if s == m and s:
        return FieldStatus.MATCH, "exact match"
    sim = SequenceMatcher(None, s, m).ratio() if (s or m) else 0.0
    if sim >= 0.90:
        return FieldStatus.MATCH, f"text match (sim={sim:.2f})"
    if sim >= 0.70:
        return FieldStatus.PARTIAL_MATCH, f"text close (sim={sim:.2f})"
    return FieldStatus.DIFF, f"text DIFF (sim={sim:.2f}): '{s}' vs '{m}'"


def _compare_authors(s_raw: object, m_raw: object) -> tuple[FieldStatus, str]:
    """Author comparison: surname is decisive, year is secondary.

    "de Gonzalo-Calvo, D. et al."  vs  "de Gonzalo-Calvo et al. [2018]"
    → MATCH on surname "de gonzalo-calvo"; year only present on one side.
    """
    s = _normalise_author(str(s_raw) if s_raw is not None else "")
    m = _normalise_author(str(m_raw) if m_raw is not None else "")

    def _split(a: str) -> tuple[str, str]:
        toks = a.split()
        if toks and re.fullmatch(r"(?:19|20)\d{2}", toks[-1]):
            return " ".join(toks[:-1]), toks[-1]
        return a, ""

    s_sur, s_yr = _split(s)
    m_sur, m_yr = _split(m)
    if not s_sur or not m_sur:
        return FieldStatus.NOT_COMPARABLE, "empty author surname"
    if s_sur == m_sur:
        if s_yr and m_yr and s_yr != m_yr:
            return FieldStatus.PARTIAL_MATCH, (
                f"surname matches ({s_sur}) but year differs ({s_yr} vs {m_yr})"
            )
        return FieldStatus.MATCH, f"author match ({s_sur})"
    sim = SequenceMatcher(None, s_sur, m_sur).ratio()
    if sim >= 0.90:
        return FieldStatus.MATCH, (
            f"author near-match (sim={sim:.2f}: '{s_sur}' ~ '{m_sur}')"
        )
    if sim >= 0.75:
        return FieldStatus.PARTIAL_MATCH, f"author partial (sim={sim:.2f})"
    return FieldStatus.DIFF, f"author DIFF: '{s_sur}' vs '{m_sur}'"


def _compare_pair(
    base: str, s_val: object, m_val: object
) -> tuple[FieldStatus, str]:
    """Compare one (student_value, model_value) pair for a given base.

    Handles None semantics; dispatches to the right comparator.
    """
    s_empty = s_val is None or (isinstance(s_val, str) and not s_val.strip())
    m_empty = m_val is None or (isinstance(m_val, str) and not m_val.strip())
    if s_empty and m_empty:
        return FieldStatus.NOT_COMPARABLE, "both empty"
    if s_empty:
        return FieldStatus.MISSING_STUDENT, "student empty, model has value"
    if m_empty:
        return FieldStatus.MISSING_MODEL, "model empty (extractor gap)"

    if base == "author":
        return _compare_authors(s_val, m_val)

    sn = _normalise_value(s_val, base)
    mn = _normalise_value(m_val, base)
    s_num = _try_float(sn)
    m_num = _try_float(mn)
    if s_num is not None and m_num is not None:
        return _compare_numeric(s_num, m_num)
    return _compare_text(str(sn) if sn is not None else "",
                        str(mn) if mn is not None else "")


_STATUS_RANK: dict[FieldStatus, int] = {
    FieldStatus.MATCH: 5,
    FieldStatus.PARTIAL_MATCH: 4,
    FieldStatus.NEEDS_REVIEW: 3,
    FieldStatus.DIFF: 2,
    FieldStatus.MISSING_STUDENT: 1,
    FieldStatus.MISSING_MODEL: 1,
    FieldStatus.NOT_COMPARABLE: 0,
}


# ======================================================================
# Canonical entry buffer used during compare()
# ======================================================================


@dataclass
class _CanonicalEntry:
    student_field: ExtractedField | None = None
    student_raw_name: str = ""
    model_fields: list[ExtractedField] = field(default_factory=list)
    model_raw_names: list[str] = field(default_factory=list)
    student_ambiguous: bool = False  # student value resolved from ungrouped→t1dm


# ======================================================================
# Comparator
# ======================================================================


class RealTableComparator(TableComparator):
    """Compare a student table against model-extracted tables using a
    canonical schema. Emits one FieldDiff per (base, group)."""

    async def compare(
        self,
        student_table: ExtractedTable,
        model_tables: list[ExtractedTable],
    ) -> TableComparisonResult:
        flags: list[ComparisonFlag] = []

        # 1. Bucket fields by canonical key
        entries: dict[tuple[str, str | None], _CanonicalEntry] = {}

        for sf in student_table.fields:
            key = _canonical_key(sf.field_name)
            ent = entries.setdefault(key, _CanonicalEntry())
            # If multiple student fields collide on the same key, keep the
            # first one with a non-null value and record both raw names.
            if ent.student_field is None or (
                (ent.student_field.value is None) and (sf.value is not None)
            ):
                ent.student_field = sf
                ent.student_raw_name = sf.field_name

        for mt in model_tables:
            for mf in mt.fields:
                key = _canonical_key(mf.field_name)
                ent = entries.setdefault(key, _CanonicalEntry())
                ent.model_fields.append(mf)
                if mf.field_name not in ent.model_raw_names:
                    ent.model_raw_names.append(mf.field_name)

        if not entries:
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

        # 2. Resolve ungrouped-student ↔ grouped-model ambiguity.
        # When a base has grouped model variants and the student provided
        # only an ungrouped value, attach the student's value to the t1dm
        # variant (the typical student focus) and mark it as ambiguous so
        # the final status becomes NEEDS_REVIEW.
        bases_with_groups: dict[str, set[str]] = {}
        for (b, g) in entries:
            if g is not None:
                bases_with_groups.setdefault(b, set()).add(g)

        drop_keys: list[tuple[str, str | None]] = []
        for (base, group), ent in list(entries.items()):
            if group is not None:
                continue
            if base not in bases_with_groups:
                continue
            if ent.student_field is None or ent.student_field.value is None:
                # Nothing useful on the ungrouped-student side; drop it if
                # the model also has nothing here.
                if not any(
                    mf.value is not None for mf in ent.model_fields
                ):
                    drop_keys.append((base, group))
                continue

            groups_present = bases_with_groups[base]
            preferred = "t1dm" if "t1dm" in groups_present else next(iter(groups_present))
            target_key = (base, preferred)
            target = entries[target_key]
            if target.student_field is None:
                target.student_field = ent.student_field
                target.student_raw_name = ent.student_raw_name
                target.student_ambiguous = True
                drop_keys.append((base, group))
            # else: both grouped and ungrouped student entries — leave
            # the ungrouped one as its own row so the user can see it.

        for k in drop_keys:
            entries.pop(k, None)

        # 3. Emit one FieldDiff per canonical entry.
        diffs: list[FieldDiff] = []
        compared_count = 0
        coverable_count = 0
        agree_count = 0
        missing_model_fields: list[str] = []
        mismatch_fields: list[str] = []
        needs_review_fields: list[str] = []

        for (base, group), ent in entries.items():
            display_name = _canonical_display_name(base, group)

            s_val = ent.student_field.value if ent.student_field else None
            s_evidence = (
                ent.student_field.evidence if ent.student_field else ""
            )
            model_vals = [mf.value for mf in ent.model_fields]
            model_evidence = [mf.evidence or "" for mf in ent.model_fields]
            any_extractor_failed = any(
                getattr(mf, "extractor_failed", False) for mf in ent.model_fields
            )

            s_empty = s_val is None or (
                isinstance(s_val, str) and not s_val.strip()
            )
            model_non_empty = [
                v for v in model_vals
                if v is not None and (not isinstance(v, str) or v.strip())
            ]
            m_empty = not model_non_empty

            # Normalised values (for display/debug)
            student_norm: object = None if s_empty else _normalise_value(s_val, base)
            model_norms: list[object] = [
                None if v is None or (isinstance(v, str) and not v.strip())
                else _normalise_value(v, base)
                for v in model_vals
            ]

            # Decide status
            if s_empty and m_empty:
                status = FieldStatus.NOT_COMPARABLE
                explanation = "both sides empty"
            elif m_empty:
                status = FieldStatus.MISSING_MODEL
                explanation = (
                    "extractor failure — value not produced by model"
                    if any_extractor_failed
                    else "model has no value (extractor gap)"
                )
                missing_model_fields.append(display_name)
            elif s_empty:
                status = FieldStatus.MISSING_STUDENT
                explanation = "student has no value; model provided one"
            else:
                # Both sides have values. Take the best comparison across
                # model values.
                best_status = FieldStatus.NOT_COMPARABLE
                best_expl = ""
                for mv in model_vals:
                    if mv is None or (isinstance(mv, str) and not mv.strip()):
                        continue
                    st, expl = _compare_pair(base, s_val, mv)
                    if _STATUS_RANK[st] > _STATUS_RANK.get(best_status, 0):
                        best_status = st
                        best_expl = expl
                    elif not best_expl:
                        best_expl = expl

                status = best_status
                explanation = best_expl

                # Ambiguous single-group student vs dual-group model →
                # never hard-judge, fall back to NEEDS_REVIEW.
                if ent.student_ambiguous:
                    status = FieldStatus.NEEDS_REVIEW
                    explanation = (
                        f"student value not labeled with cohort; "
                        f"provisionally assigned to {group} — "
                        f"underlying match: {best_expl}"
                    )

                coverable_count += 1
                compared_count += 1
                if status in (FieldStatus.MATCH, FieldStatus.PARTIAL_MATCH):
                    agree_count += 1
                elif status == FieldStatus.NEEDS_REVIEW:
                    needs_review_fields.append(display_name)
                else:
                    mismatch_fields.append(display_name)

            is_consistent = status in (FieldStatus.MATCH, FieldStatus.PARTIAL_MATCH)

            # Source-type summary
            if ent.student_field is not None and ent.model_fields:
                source_type = "student+llm"
            elif ent.model_fields:
                source_type = "llm-only"
            elif ent.student_field is not None:
                source_type = "student-only"
            else:
                source_type = ""

            diffs.append(
                FieldDiff(
                    field_name=display_name,
                    student_raw_name=ent.student_raw_name,
                    model_raw_names=list(ent.model_raw_names),
                    student_value=s_val,
                    student_value_normalized=student_norm,
                    student_evidence=s_evidence,
                    model_values=model_vals,
                    model_values_normalized=model_norms,
                    model_evidence=model_evidence,
                    status=status,
                    is_consistent=is_consistent,
                    explanation=explanation,
                    source_type=source_type,
                )
            )

        total = len(diffs)
        agreement_rate = (agree_count / compared_count) if compared_count > 0 else 0.0
        coverage_rate = (coverable_count / total) if total > 0 else 0.0

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

        if needs_review_fields:
            names = ", ".join(needs_review_fields[:5])
            flags.append(
                ComparisonFlag(
                    code=ComparisonFlagCode.NEEDS_REVIEW.value,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Paper {student_table.paper_id}: "
                        f"{len(needs_review_fields)} field(s) need manual review "
                        f"({names})."
                    ),
                )
            )

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

    Verdict (applied top-down, first match wins):
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

        # Low-coverage health warning — makes the summary honest even
        # when agreement looks high on a tiny comparable subset.
        low_coverage_note = ""
        if avg_coverage < 0.3:
            low_coverage_note = (
                " WARNING: average agreement reflects only the small "
                "comparable subset; overall coverage is very low "
                f"({avg_coverage:.0%}), so overall confidence is limited."
            )
        elif avg_coverage < 0.5:
            low_coverage_note = (
                " Note: average agreement is computed only over the "
                "comparable fields; coverage is below 50%."
            )

        summary = (
            f"[{verdict.value}] Reviewed {total} paper(s) "
            f"(compared={len(compared_results)}, skipped={skipped_count}). "
            f"Average agreement (comparable fields only): {avg_agreement:.0%}. "
            f"Average coverage: {avg_coverage:.0%}. {detail}{low_coverage_note}"
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

from __future__ import annotations

from pathlib import Path
import os
import re
from typing import Any

from .config import PipelineConfig
from .llm_clients import BaseLLMClient, MockLLMClient, OpenAICompatibleClient
from .models import ClaimValidationFinding, DataExtractionValidationResult, PaperDownloadResult


class DualLLMValidator:
    """Module 5: cross-validate extracted claims with two independent LLM evaluators."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.llm_a = self._build_client(
            provider=config.llm_a_provider,
            model=config.llm_a_model,
            name="LLM-A",
            env_prefix="LLM_A",
        )
        self.llm_b = self._build_client(
            provider=config.llm_b_provider,
            model=config.llm_b_model,
            name="LLM-B",
            env_prefix="LLM_B",
        )

    def validate(
        self,
        claims: list[str],
        download_result: PaperDownloadResult,
    ) -> DataExtractionValidationResult:
        evidence_text = self._aggregate_evidence(download_result)
        selected_claims = claims[: self.config.max_claims_to_validate]

        findings: list[ClaimValidationFinding] = []
        for idx, claim in enumerate(selected_claims, start=1):
            a = self._normalize_output(self.llm_a.evaluate_claim(claim, evidence_text))
            b = self._normalize_output(self.llm_b.evaluate_claim(claim, evidence_text))

            same_label = a["label"] == b["label"]
            agreement = 1.0 if same_label else 0.0
            avg_conf = (a["confidence"] + b["confidence"]) / 2
            flagged = (not same_label) or avg_conf < self.config.stage2_pass_threshold

            findings.append(
                ClaimValidationFinding(
                    claim_id=idx,
                    claim_text=claim,
                    llm_a_label=a["label"],
                    llm_b_label=b["label"],
                    llm_a_confidence=a["confidence"],
                    llm_b_confidence=b["confidence"],
                    agreement_score=agreement,
                    final_label=self._final_label(a["label"], b["label"]),
                    flagged=flagged,
                    rationale_a=a["rationale"],
                    rationale_b=b["rationale"],
                )
            )

        total_claims = len(findings)
        flagged_claims = len([item for item in findings if item.flagged])
        confidence_score = self._compute_confidence(findings)

        return DataExtractionValidationResult(
            total_claims=total_claims,
            flagged_claims=flagged_claims,
            confidence_score=round(confidence_score, 3),
            findings=findings,
            stage2_pass=confidence_score >= self.config.stage2_pass_threshold,
        )

    def _aggregate_evidence(self, download_result: PaperDownloadResult) -> str:
        chunks: list[str] = []
        for item in download_result.downloaded:
            if not item.local_path:
                continue
            try:
                text = item.local_path.read_text(encoding="utf-8", errors="ignore")
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    chunks.append(text[:30000])
            except OSError:
                continue

        return "\n".join(chunks)[:120000]

    def _build_client(
        self,
        provider: str,
        model: str,
        name: str,
        env_prefix: str,
    ) -> BaseLLMClient:
        if provider.lower() == "mock":
            return MockLLMClient(name=name)

        base_url = os.getenv(f"{env_prefix}_BASE_URL")
        api_key = os.getenv(f"{env_prefix}_API_KEY")
        if not base_url or not api_key:
            return MockLLMClient(name=f"{name}-mock-fallback")

        return OpenAICompatibleClient(
            name=name,
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=self.config.llm_timeout_seconds,
        )

    @staticmethod
    def _normalize_output(data: dict[str, Any]) -> dict[str, Any]:
        label = str(data.get("label", "insufficient")).lower().strip()
        if label not in {"supported", "partially_supported", "unsupported", "insufficient"}:
            label = "insufficient"

        try:
            confidence = float(data.get("confidence", 0.2))
        except (TypeError, ValueError):
            confidence = 0.2
        confidence = max(0.0, min(confidence, 1.0))

        rationale = str(data.get("rationale", "No rationale provided.")).strip()
        return {"label": label, "confidence": confidence, "rationale": rationale}

    @staticmethod
    def _final_label(label_a: str, label_b: str) -> str:
        if label_a == label_b:
            return label_a
        priority = ["unsupported", "insufficient", "partially_supported", "supported"]
        for label in priority:
            if label in {label_a, label_b}:
                return label
        return "insufficient"

    @staticmethod
    def _compute_confidence(findings: list[ClaimValidationFinding]) -> float:
        if not findings:
            return 0.0
        score = 0.0
        for finding in findings:
            avg_conf = (finding.llm_a_confidence + finding.llm_b_confidence) / 2
            score += avg_conf * finding.agreement_score
        return score / len(findings)

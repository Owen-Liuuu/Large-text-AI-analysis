"""Application configuration: YAML loading + Pydantic validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from lit_inspector.core.exceptions import ConfigError


class LLMSettings(BaseModel):
    """Settings for the LLM backend."""

    provider: str = "mock"
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 4096
    api_key: str = ""
    base_url: str = ""


class PubMedSettings(BaseModel):
    """Settings for PubMed E-utilities API."""

    api_key: str = ""
    base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    rate_limit: float = 3.0  # requests per second (10 with api_key)
    email: str = ""  # recommended by NCBI; also used for Unpaywall API


class UnpaywallSettings(BaseModel):
    """Settings for Unpaywall API (free OA full-text discovery)."""

    email: str = ""  # required by Unpaywall TOS (your email address)
    enabled: bool = True  # set False to skip Unpaywall tier


class CrossRefSettings(BaseModel):
    """Settings for CrossRef API."""

    base_url: str = "https://api.crossref.org"
    mailto: str = ""  # enter email to join the polite pool (faster)
    timeout: float = 30.0


class ThresholdSettings(BaseModel):
    """Numeric thresholds used in verification / comparison."""

    title_similarity: float = 0.85
    author_match_ratio: float = 0.8


class PathSettings(BaseModel):
    """File-system paths used by the application."""

    data_dir: Path = Path("./data")
    output_dir: Path = Path("./output")
    log_file: Path = Path("./logs/lit_inspector.log")


class AppConfig(BaseModel):
    """Top-level application configuration."""

    app_name: str = "lit-inspector"
    environment: str = "development"
    mock_mode: bool = True
    enabled_steps: list[str] = Field(
        default_factory=lambda: [
            "search_validation",
            "paper_verification",
            "data_extraction",
            "table_comparison",
        ]
    )
    llm: LLMSettings = Field(default_factory=LLMSettings)
    llm2: LLMSettings | None = None  # optional second LLM for cross-validation
    pubmed: PubMedSettings = Field(default_factory=PubMedSettings)
    unpaywall: UnpaywallSettings = Field(default_factory=UnpaywallSettings)
    crossref: CrossRefSettings = Field(default_factory=CrossRefSettings)
    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)
    paths: PathSettings = Field(default_factory=PathSettings)


def load_config(path: Path) -> AppConfig:
    """Load application config from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated AppConfig instance.

    Raises:
        ConfigError: If the file cannot be read or parsed.
    """
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return AppConfig(**data)
    except Exception as exc:
        raise ConfigError(f"Failed to load config from {path}: {exc}") from exc

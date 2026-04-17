"""Real PaperRetriever: fetches paper abstracts from PubMed via efetch.

For each paper, tries:
  1. If DOI is available → search PubMed by DOI to get PMID → efetch abstract
  2. If title is available → search PubMed by title → efetch abstract
  3. Falls back to CrossRef abstract if PubMed fails

This gives Step 3 real paper content (abstract) for the LLM to extract from.
Full-text retrieval (via Unpaywall etc.) can be added later.
"""
from __future__ import annotations

import httpx
import structlog

from lit_inspector.core.config import PubMedSettings
from lit_inspector.steps.data_extraction.schemas import PaperDocument
from lit_inspector.steps.paper_verification.interfaces import PaperRetriever
from lit_inspector.steps.paper_verification.schemas import ReferenceEntry

logger = structlog.get_logger(__name__)


class PubMedPaperRetriever(PaperRetriever):
    """Retrieves paper abstracts from PubMed E-utilities.

    Args:
        settings: PubMed API settings (api_key, base_url).
    """

    def __init__(self, settings: PubMedSettings) -> None:
        self._base = settings.base_url.rstrip("/")
        self._api_key = settings.api_key

    async def retrieve(self, reference: ReferenceEntry) -> PaperDocument | None:
        """Attempt to retrieve a paper's abstract from PubMed.

        Strategy:
          1. Search PubMed by DOI (exact) to get PMID
          2. Fallback: search by title
          3. Use efetch to get the abstract text
        """
        paper_id = reference.doi or reference.title[:50]
        logger.info("pubmed_retrieve_start", paper_id=paper_id[:50])

        try:
            pmid = await self._find_pmid(reference)
            if not pmid:
                logger.warning("pubmed_no_pmid", title=reference.title[:50])
                return self._fallback_document(reference)

            abstract = await self._fetch_abstract(pmid)
            if not abstract:
                logger.warning("pubmed_no_abstract", pmid=pmid)
                return self._fallback_document(reference)

            logger.info(
                "pubmed_retrieve_done",
                pmid=pmid,
                abstract_len=len(abstract),
            )
            return PaperDocument(
                paper_id=reference.doi or f"pmid:{pmid}",
                reference=reference,
                full_text=abstract,
                sections={"abstract": abstract},
                metadata={"source": "pubmed", "pmid": pmid},
            )

        except Exception as exc:
            logger.warning("pubmed_retrieve_error", error=str(exc)[:100])
            return self._fallback_document(reference)

    async def _find_pmid(self, reference: ReferenceEntry) -> str | None:
        """Search PubMed to find the PMID for a reference."""
        params: dict[str, str] = {
            "db": "pubmed",
            "rettype": "json",
            "retmode": "json",
            "retmax": "3",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        # Try DOI search first
        if reference.doi:
            params["term"] = f"{reference.doi}[doi]"
            pmid = await self._esearch(params)
            if pmid:
                return pmid

        # Fallback: title search
        if reference.title:
            params["term"] = f"{reference.title}[title]"
            pmid = await self._esearch(params)
            if pmid:
                return pmid

        return None

    async def _esearch(self, params: dict[str, str]) -> str | None:
        """Execute an esearch and return the first PMID, or None."""
        url = f"{self._base}/esearch.fcgi"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        id_list = data.get("esearchresult", {}).get("idlist", [])
        return id_list[0] if id_list else None

    async def _fetch_abstract(self, pmid: str) -> str:
        """Fetch the abstract text for a given PMID using efetch."""
        url = f"{self._base}/efetch.fcgi"
        params: dict[str, str] = {
            "db": "pubmed",
            "id": pmid,
            "rettype": "abstract",
            "retmode": "text",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.text.strip()

    @staticmethod
    def _fallback_document(reference: ReferenceEntry) -> PaperDocument:
        """Create a minimal PaperDocument with just the title/metadata.

        The LLM will receive minimal content and extract what it can.
        """
        text = f"Title: {reference.title}\n"
        if reference.authors:
            text += f"Authors: {', '.join(reference.authors)}\n"
        if reference.journal:
            text += f"Journal: {reference.journal}\n"
        if reference.year:
            text += f"Year: {reference.year}\n"
        text += "\n[Full text not available. Only title and metadata provided.]"

        return PaperDocument(
            paper_id=reference.doi or f"fallback-{reference.title[:20]}",
            reference=reference,
            full_text=text,
            metadata={"source": "fallback-metadata-only"},
        )

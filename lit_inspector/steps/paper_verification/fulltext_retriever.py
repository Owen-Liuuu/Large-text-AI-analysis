"""Full-text paper retriever with 3-tier retrieval chain.

Retrieval priority:
  1. PubMed Central (PMC) — free full-text XML for open-access papers
  2. Unpaywall — discovers OA PDF/HTML links for any DOI
  3. PubMed abstract — fallback (current baseline)

This dramatically improves Step 3 data extraction quality because the
LLM receives full paper text (methods, results, tables) instead of just
a ~300-word abstract.

API references:
  PMC: https://www.ncbi.nlm.nih.gov/pmc/tools/developers/
  Unpaywall: https://unpaywall.org/products/api
  (Unpaywall requires an email, no API key needed)
"""
from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET

import httpx
import structlog

from lit_inspector.core.config import PubMedSettings
from lit_inspector.steps.data_extraction.schemas import PaperDocument
from lit_inspector.steps.paper_verification.interfaces import PaperRetriever
from lit_inspector.steps.paper_verification.schemas import ReferenceEntry

logger = structlog.get_logger(__name__)

# Max chars to keep from full text (avoid blowing up LLM context)
_MAX_FULLTEXT_CHARS = 60_000


class FullTextRetriever(PaperRetriever):
    """Retrieves paper full text via PMC → Unpaywall → PubMed abstract.

    Args:
        pubmed_settings: PubMed/PMC API settings.
        unpaywall_email: Email for Unpaywall API (required by their TOS).
            If empty, Unpaywall tier is skipped.
    """

    def __init__(
        self,
        pubmed_settings: PubMedSettings,
        unpaywall_email: str = "",
    ) -> None:
        self._pubmed_base = pubmed_settings.base_url.rstrip("/")
        self._api_key = pubmed_settings.api_key
        self._unpaywall_email = unpaywall_email or pubmed_settings.email
        self._timeout = httpx.Timeout(60.0, connect=15.0)

    async def retrieve(self, reference: ReferenceEntry) -> PaperDocument | None:
        """Try to get the fullest text available for a reference.

        Chain: PMC full text → Unpaywall OA → PubMed abstract → fallback.
        """
        paper_id = reference.doi or reference.title[:50]
        logger.info("fulltext_retrieve_start", paper_id=paper_id[:50])

        # --- Tier 1: PubMed Central full text ---
        doc = await self._try_pmc(reference)
        if doc:
            return doc

        # --- Tier 2: Unpaywall open-access PDF/HTML ---
        doc = await self._try_unpaywall(reference)
        if doc:
            return doc

        # --- Tier 3: PubMed abstract (baseline) ---
        doc = await self._try_pubmed_abstract(reference)
        if doc:
            return doc

        # --- Fallback: metadata only ---
        logger.warning("fulltext_all_tiers_failed", title=reference.title[:50])
        return self._fallback_document(reference)

    # ==================================================================
    # Tier 1: PubMed Central
    # ==================================================================

    async def _try_pmc(self, reference: ReferenceEntry) -> PaperDocument | None:
        """Try to get full text from PubMed Central.

        Steps:
          1. Find PMC ID via esearch (DOI → PMCID)
          2. Fetch full XML from PMC efetch
          3. Parse XML → plain text
        """
        try:
            pmc_id = await self._find_pmc_id(reference)
            if not pmc_id:
                return None

            full_text = await self._fetch_pmc_fulltext(pmc_id)
            if not full_text or len(full_text) < 200:
                return None

            logger.info(
                "pmc_fulltext_retrieved",
                pmc_id=pmc_id,
                chars=len(full_text),
            )
            return PaperDocument(
                paper_id=reference.doi or f"pmc:{pmc_id}",
                reference=reference,
                full_text=full_text[:_MAX_FULLTEXT_CHARS],
                sections=self._split_sections(full_text),
                metadata={"source": "pmc", "pmc_id": pmc_id},
            )

        except Exception as exc:
            logger.debug("pmc_tier_failed", error=str(exc)[:120])
            return None

    async def _find_pmc_id(self, reference: ReferenceEntry) -> str | None:
        """Search for a PMC ID using the paper's DOI.

        IMPORTANT: Only searches by DOI, NOT by title. Title search in PMC
        is highly unreliable and frequently returns unrelated papers with
        similar keywords (e.g. other diabetes/EAT papers instead of the
        target paper). DOI is the only reliable identifier.
        """
        if not reference.doi:
            return None

        params: dict[str, str] = {
            "db": "pmc",
            "rettype": "json",
            "retmode": "json",
            "retmax": "3",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        params["term"] = f"{reference.doi}[doi]"
        return await self._esearch(params)

    async def _fetch_pmc_fulltext(self, pmc_id: str) -> str:
        """Fetch full text XML from PMC and convert to plain text."""
        url = f"{self._pubmed_base}/efetch.fcgi"
        params: dict[str, str] = {
            "db": "pmc",
            "id": pmc_id,
            "rettype": "xml",
            "retmode": "xml",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            xml_text = resp.text

        return self._pmc_xml_to_text(xml_text)

    @staticmethod
    def _pmc_xml_to_text(xml_text: str) -> str:
        """Parse PMC XML and extract readable text.

        Extracts: title, abstract, body sections, tables (as text).
        """
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return ""

        parts: list[str] = []

        # Article title
        for title_el in root.iter("article-title"):
            if title_el.text:
                parts.append(f"TITLE: {_et_text(title_el)}\n")

        # Abstract
        for abstract_el in root.iter("abstract"):
            text = _et_text(abstract_el)
            if text:
                parts.append(f"ABSTRACT:\n{text}\n")

        # Body sections
        for body_el in root.iter("body"):
            for sec in body_el.iter("sec"):
                # Section title
                title_nodes = sec.findall("title")
                sec_title = _et_text(title_nodes[0]) if title_nodes else ""
                # Section paragraphs
                paragraphs = []
                for p in sec.findall("p"):
                    p_text = _et_text(p)
                    if p_text:
                        paragraphs.append(p_text)
                if paragraphs:
                    header = f"\n## {sec_title}\n" if sec_title else "\n"
                    parts.append(header + "\n".join(paragraphs) + "\n")

        # Tables (as text, for data extraction)
        for table_wrap in root.iter("table-wrap"):
            caption_el = table_wrap.find("caption")
            caption = _et_text(caption_el) if caption_el is not None else ""
            table_el = table_wrap.find(".//table")
            if table_el is not None:
                table_text = _table_to_text(table_el)
                parts.append(f"\nTABLE: {caption}\n{table_text}\n")

        return "\n".join(parts)

    # ==================================================================
    # Tier 2: Unpaywall
    # ==================================================================

    async def _try_unpaywall(self, reference: ReferenceEntry) -> PaperDocument | None:
        """Try to find and download open-access full text via Unpaywall.

        Unpaywall provides free OA locations for papers identified by DOI.
        It can return PDF or HTML links.
        """
        if not reference.doi or not self._unpaywall_email:
            return None

        try:
            oa_url, content_type = await self._unpaywall_find_oa(reference.doi)
            if not oa_url:
                return None

            if content_type == "pdf":
                full_text = await self._download_and_parse_pdf(oa_url)
            else:
                full_text = await self._download_html_text(oa_url)

            if not full_text or len(full_text) < 200:
                return None

            logger.info(
                "unpaywall_fulltext_retrieved",
                doi=reference.doi,
                content_type=content_type,
                chars=len(full_text),
            )
            return PaperDocument(
                paper_id=reference.doi,
                reference=reference,
                full_text=full_text[:_MAX_FULLTEXT_CHARS],
                sections=self._split_sections(full_text),
                metadata={
                    "source": "unpaywall",
                    "oa_url": oa_url,
                    "content_type": content_type,
                },
            )

        except Exception as exc:
            logger.debug("unpaywall_tier_failed", error=str(exc)[:120])
            return None

    async def _unpaywall_find_oa(self, doi: str) -> tuple[str, str]:
        """Query Unpaywall API for OA location.

        Returns (url, content_type) where content_type is 'pdf' or 'html'.
        Returns ('', '') if no OA version found.
        """
        url = f"https://api.unpaywall.org/v2/{doi}"
        params = {"email": self._unpaywall_email}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 404:
                logger.debug("unpaywall_not_found", doi=doi)
                return "", ""
            resp.raise_for_status()
            data = resp.json()

        # Check best_oa_location first, then oa_locations list
        best = data.get("best_oa_location") or {}
        if not best and data.get("oa_locations"):
            best = data["oa_locations"][0]

        if not best:
            return "", ""

        # Prefer PDF
        pdf_url = best.get("url_for_pdf", "")
        if pdf_url:
            return pdf_url, "pdf"

        # Fallback to landing page / HTML
        landing_url = best.get("url_for_landing_page", "") or best.get("url", "")
        if landing_url:
            return landing_url, "html"

        return "", ""

    async def _download_and_parse_pdf(self, url: str) -> str:
        """Download a PDF from a URL and extract text using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("pymupdf_not_installed", msg="pip install PyMuPDF")
            return ""

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=30.0),
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            pdf_bytes = resp.content

        # Parse PDF in memory
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        parts: list[str] = []
        for page in doc:
            text = page.get_text("text")
            if text:
                parts.append(text)
        doc.close()

        return "\n".join(parts)

    async def _download_html_text(self, url: str) -> str:
        """Download an HTML page and extract the main text content.

        Uses a simple approach: strip tags, keep text.
        """
        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "lit-inspector/1.0 (academic research tool; "
                    f"mailto:{self._unpaywall_email})"
                )
            },
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text

        # Basic HTML → text (strip tags)
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        # Decode HTML entities
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&nbsp;", " ").replace("&#39;", "'").replace("&quot;", '"')
        return text.strip()

    # ==================================================================
    # Tier 3: PubMed abstract (same as original retriever)
    # ==================================================================

    async def _try_pubmed_abstract(
        self, reference: ReferenceEntry
    ) -> PaperDocument | None:
        """Fallback: fetch just the abstract from PubMed."""
        try:
            pmid = await self._find_pubmed_pmid(reference)
            if not pmid:
                return None

            abstract = await self._fetch_abstract(pmid)
            if not abstract:
                return None

            logger.info(
                "pubmed_abstract_retrieved",
                pmid=pmid,
                chars=len(abstract),
            )
            return PaperDocument(
                paper_id=reference.doi or f"pmid:{pmid}",
                reference=reference,
                full_text=abstract,
                sections={"abstract": abstract},
                metadata={"source": "pubmed_abstract", "pmid": pmid},
            )

        except Exception as exc:
            logger.debug("pubmed_abstract_failed", error=str(exc)[:120])
            return None

    async def _find_pubmed_pmid(self, reference: ReferenceEntry) -> str | None:
        """Search PubMed to get PMID."""
        params: dict[str, str] = {
            "db": "pubmed",
            "rettype": "json",
            "retmode": "json",
            "retmax": "3",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        if reference.doi:
            params["term"] = f"{reference.doi}[doi]"
            pmid = await self._esearch(params)
            if pmid:
                return pmid

        if reference.title:
            params["term"] = f"{reference.title}[title]"
            pmid = await self._esearch(params)
            if pmid:
                return pmid

        return None

    async def _fetch_abstract(self, pmid: str) -> str:
        """Fetch abstract text from PubMed efetch."""
        url = f"{self._pubmed_base}/efetch.fcgi"
        params: dict[str, str] = {
            "db": "pubmed",
            "id": pmid,
            "rettype": "abstract",
            "retmode": "text",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.text.strip()

    # ==================================================================
    # Shared helpers
    # ==================================================================

    async def _esearch(self, params: dict[str, str]) -> str | None:
        """Execute an NCBI esearch and return the first ID, or None."""
        url = f"{self._pubmed_base}/esearch.fcgi"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        id_list = data.get("esearchresult", {}).get("idlist", [])
        return id_list[0] if id_list else None

    @staticmethod
    def _split_sections(text: str) -> dict[str, str]:
        """Split full text into named sections by ## headers."""
        sections: dict[str, str] = {}
        current_name = "introduction"
        current_lines: list[str] = []

        for line in text.split("\n"):
            if line.startswith("## "):
                if current_lines:
                    sections[current_name] = "\n".join(current_lines).strip()
                current_name = line[3:].strip().lower().replace(" ", "_")
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections[current_name] = "\n".join(current_lines).strip()

        return sections

    @staticmethod
    def _fallback_document(reference: ReferenceEntry) -> PaperDocument:
        """Create a metadata-only document when all retrieval tiers fail."""
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


# ======================================================================
# XML helper utilities
# ======================================================================


def _et_text(element: ET.Element | None) -> str:
    """Recursively extract all text from an ElementTree element."""
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def _table_to_text(table_el: ET.Element) -> str:
    """Convert an XML <table> element to tab-separated text.

    Handles <thead>/<tbody>/<tr>/<th>/<td> structure.
    """
    rows: list[str] = []
    for tr in table_el.iter("tr"):
        cells: list[str] = []
        for cell in tr:
            if cell.tag in ("th", "td"):
                cells.append(_et_text(cell))
        if cells:
            rows.append("\t".join(cells))
    return "\n".join(rows)

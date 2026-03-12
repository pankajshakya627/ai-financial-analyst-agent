"""
SEC EDGAR filing downloader and parser.
Supports 10-K (annual), 10-Q (quarterly), 8-K (current events), and DEF 14A (proxy).
Complies with SEC rate-limiting requirements (max 10 requests/second).
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import httpx
from bs4 import BeautifulSoup

from config.settings import settings

logger = logging.getLogger(__name__)

# SEC EDGAR API endpoints
EDGAR_COMPANY_SEARCH = "https://efts.sec.gov/LATEST/search-index?q={query}&dateRange=custom&startdt={start}&enddt={end}&forms={form_type}"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}"
EDGAR_FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index?q={query}&forms={forms}&dateRange=custom&startdt={start}&enddt={end}"


@dataclass
class SECFiling:
    """Represents a parsed SEC filing."""

    cik: str
    company_name: str
    form_type: str  # 10-K, 10-Q, 8-K, DEF14A
    filing_date: str
    accession_number: str
    primary_document: str
    raw_text: str = ""
    sections: dict = field(default_factory=dict)
    financial_data: dict = field(default_factory=dict)
    url: str = ""


class SECEdgarClient:
    """
    Async client for SEC EDGAR with rate limiting.
    Downloads and parses SEC filings for RAG ingestion.
    """

    SUPPORTED_FORMS = {"10-K", "10-Q", "8-K", "DEF 14A", "10-K/A", "10-Q/A"}
    RATE_LIMIT_DELAY = 0.12  # ~8 requests/sec to stay under SEC's 10/sec limit

    def __init__(self):
        self.user_agent = settings.sec_edgar_user_agent
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        self._last_request_time = 0.0
        self.filings_dir = Path(settings.sec_filings_dir)
        self.filings_dir.mkdir(parents=True, exist_ok=True)

    async def _rate_limited_get(self, url: str, client: httpx.AsyncClient) -> httpx.Response:
        """Make a rate-limited GET request to SEC EDGAR."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            await asyncio.sleep(self.RATE_LIMIT_DELAY - elapsed)

        response = await client.get(url, headers=self.headers, follow_redirects=True, timeout=30)
        self._last_request_time = time.time()
        response.raise_for_status()
        return response

    async def get_company_cik(self, ticker: str) -> Optional[str]:
        """Look up a company's CIK number from its ticker symbol."""
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker}&type=&dateb=&owner=include&count=10&search_text=&action=getcompany&output=atom"

        async with httpx.AsyncClient() as client:
            try:
                # Use the company tickers JSON endpoint
                tickers_url = "https://www.sec.gov/files/company_tickers.json"
                resp = await self._rate_limited_get(tickers_url, client)
                tickers_data = resp.json()

                for entry in tickers_data.values():
                    if entry.get("ticker", "").upper() == ticker.upper():
                        cik = str(entry["cik_str"]).zfill(10)
                        logger.info(f"Found CIK {cik} for ticker {ticker}")
                        return cik

                logger.warning(f"CIK not found for ticker: {ticker}")
                return None
            except Exception as e:
                logger.error(f"Error looking up CIK for {ticker}: {e}")
                return None

    async def get_company_filings(
        self,
        ticker: str,
        form_type: str = "10-K",
        count: int = 5,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[dict]:
        """
        Get a list of filings for a company.

        Args:
            ticker: Stock ticker symbol
            form_type: SEC form type (10-K, 10-Q, 8-K)
            count: Number of recent filings to retrieve
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        cik = await self.get_company_cik(ticker)
        if not cik:
            raise ValueError(f"Could not find CIK for ticker: {ticker}")

        async with httpx.AsyncClient() as client:
            url = EDGAR_SUBMISSIONS.format(cik=cik)
            resp = await self._rate_limited_get(url, client)
            submissions = resp.json()

        company_name = submissions.get("name", ticker)
        recent = submissions.get("filings", {}).get("recent", {})

        filings = []
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        for i in range(len(forms)):
            if forms[i] != form_type:
                continue

            filing_date = dates[i]
            if start_date and filing_date < start_date:
                continue
            if end_date and filing_date > end_date:
                continue

            accession_clean = accessions[i].replace("-", "")
            filings.append(
                {
                    "cik": cik,
                    "company_name": company_name,
                    "form_type": forms[i],
                    "filing_date": filing_date,
                    "accession_number": accessions[i],
                    "accession_clean": accession_clean,
                    "primary_document": primary_docs[i],
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{primary_docs[i]}",
                }
            )

            if len(filings) >= count:
                break

        logger.info(f"Found {len(filings)} {form_type} filings for {ticker}")
        return filings

    async def download_filing(self, filing_info: dict) -> SECFiling:
        """Download and parse a single SEC filing."""
        async with httpx.AsyncClient() as client:
            resp = await self._rate_limited_get(filing_info["url"], client)
            content = resp.text

        # Parse HTML filing
        raw_text = self._parse_filing_html(content)

        # Extract sections
        sections = self._extract_filing_sections(raw_text, filing_info["form_type"])

        # Try to extract financial tables
        financial_data = self._extract_financial_tables(content)

        filing = SECFiling(
            cik=filing_info["cik"],
            company_name=filing_info["company_name"],
            form_type=filing_info["form_type"],
            filing_date=filing_info["filing_date"],
            accession_number=filing_info["accession_number"],
            primary_document=filing_info["primary_document"],
            raw_text=raw_text,
            sections=sections,
            financial_data=financial_data,
            url=filing_info["url"],
        )

        # Save locally
        self._save_filing(filing)
        return filing

    def _parse_filing_html(self, html_content: str) -> str:
        """Extract clean text from SEC filing HTML."""
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "meta", "link"]):
            element.decompose()

        text = soup.get_text(separator="\n")

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)

        return text

    def _extract_filing_sections(self, text: str, form_type: str) -> dict:
        """Extract named sections from a filing based on form type."""
        sections = {}

        if form_type in ("10-K", "10-K/A"):
            section_map = {
                "business": r"(?i)ITEM\s+1\.?\s+BUSINESS",
                "risk_factors": r"(?i)ITEM\s+1A\.?\s+RISK\s+FACTORS",
                "properties": r"(?i)ITEM\s+2\.?\s+PROPERTIES",
                "legal_proceedings": r"(?i)ITEM\s+3\.?\s+LEGAL\s+PROCEEDINGS",
                "mda": r"(?i)ITEM\s+7\.?\s+MANAGEMENT.?S?\s+DISCUSSION",
                "quantitative_disclosures": r"(?i)ITEM\s+7A\.?\s+QUANTITATIVE",
                "financial_statements": r"(?i)ITEM\s+8\.?\s+FINANCIAL\s+STATEMENTS",
                "controls_procedures": r"(?i)ITEM\s+9A\.?\s+CONTROLS\s+AND\s+PROCEDURES",
            }
        elif form_type in ("10-Q", "10-Q/A"):
            section_map = {
                "financial_statements": r"(?i)ITEM\s+1\.?\s+FINANCIAL\s+STATEMENTS",
                "mda": r"(?i)ITEM\s+2\.?\s+MANAGEMENT.?S?\s+DISCUSSION",
                "quantitative_disclosures": r"(?i)ITEM\s+3\.?\s+QUANTITATIVE",
                "controls_procedures": r"(?i)ITEM\s+4\.?\s+CONTROLS\s+AND\s+PROCEDURES",
                "risk_factors": r"(?i)ITEM\s+1A\.?\s+RISK\s+FACTORS",
            }
        else:
            return {"full_text": text}

        # Find section boundaries
        found_sections = []
        for name, pattern in section_map.items():
            match = re.search(pattern, text)
            if match:
                found_sections.append((name, match.start()))

        found_sections.sort(key=lambda x: x[1])

        for i, (name, start) in enumerate(found_sections):
            end = found_sections[i + 1][1] if i + 1 < len(found_sections) else len(text)
            sections[name] = text[start:end]

        return sections

    def _extract_financial_tables(self, html_content: str) -> dict:
        """Extract financial data from HTML tables in SEC filings."""
        soup = BeautifulSoup(html_content, "html.parser")
        tables_data = {}

        for i, table in enumerate(soup.find_all("table")):
            rows = []
            for tr in table.find_all("tr"):
                cells = []
                for td in tr.find_all(["td", "th"]):
                    cell_text = td.get_text(strip=True)
                    cells.append(cell_text)
                if cells:
                    rows.append(cells)

            if rows:
                # Try to identify if this is a financial statement
                header_text = " ".join(str(c) for c in rows[0]) if rows else ""
                if any(
                    kw in header_text.lower()
                    for kw in [
                        "revenue",
                        "income",
                        "assets",
                        "liabilities",
                        "cash flow",
                        "balance",
                        "operations",
                    ]
                ):
                    tables_data[f"financial_table_{i}"] = rows

        return tables_data

    def _save_filing(self, filing: SECFiling) -> Path:
        """Save a filing to local storage."""
        safe_name = re.sub(r"[^\w\-]", "_", filing.company_name)
        filename = f"{safe_name}_{filing.form_type}_{filing.filing_date}.json"
        filepath = self.filings_dir / filename

        data = {
            "cik": filing.cik,
            "company_name": filing.company_name,
            "form_type": filing.form_type,
            "filing_date": filing.filing_date,
            "accession_number": filing.accession_number,
            "url": filing.url,
            "sections": {k: v[:5000] for k, v in filing.sections.items()},  # truncate for storage
            "raw_text_length": len(filing.raw_text),
            "downloaded_at": datetime.utcnow().isoformat(),
        }

        filepath.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved filing: {filepath}")
        return filepath

    async def search_filings(
        self,
        query: str,
        form_types: list[str] = None,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
    ) -> list[dict]:
        """Full-text search across SEC EDGAR filings."""
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        forms = ",".join(form_types or ["10-K", "10-Q"])
        url = (
            f"https://efts.sec.gov/LATEST/search-index?"
            f"q=%22{query}%22&forms={forms}"
            f"&dateRange=custom&startdt={start_date}&enddt={end_date}"
        )

        async with httpx.AsyncClient() as client:
            resp = await self._rate_limited_get(url, client)
            results = resp.json()

        hits = results.get("hits", {}).get("hits", [])
        return [
            {
                "company": hit.get("_source", {}).get("display_names", ["Unknown"])[0],
                "form_type": hit.get("_source", {}).get("form_type", ""),
                "filing_date": hit.get("_source", {}).get("file_date", ""),
                "url": f"https://www.sec.gov/Archives/edgar/data/{hit.get('_source', {}).get('file_num', '')}",
            }
            for hit in hits[:20]
        ]

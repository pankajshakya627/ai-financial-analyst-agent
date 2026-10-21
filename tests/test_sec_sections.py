"""SEC section extraction should skip table-of-contents entries."""

import sys
import types

import pytest

pytest.importorskip("bs4")
pytest.importorskip("httpx")


def test_section_skips_table_of_contents():
    from ingestion.sec_edgar import SECEdgarClient

    client = SECEdgarClient.__new__(SECEdgarClient)
    toc = "ITEM 1. BUSINESS\nITEM 1A. RISK FACTORS\nITEM 2. PROPERTIES\n"
    body = (
        "ITEM 1. BUSINESS\n" + "We make widgets. " * 50 + "\n"
        "ITEM 1A. RISK FACTORS\n" + "Supply chain risk. " * 50 + "\n"
        "ITEM 2. PROPERTIES\n" + "We own a factory. " * 50
    )
    sections = client._extract_filing_sections(toc + body, "10-K")
    assert "Supply chain risk" in sections["risk_factors"]
    assert len(sections["risk_factors"]) > 500

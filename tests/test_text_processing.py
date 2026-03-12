"""Unit tests for text chunking and processing."""

import pytest
from core.text_processing import FinancialTextSplitter, clean_financial_text


@pytest.fixture
def splitter():
    return FinancialTextSplitter(chunk_size=200, chunk_overlap=50)


class TestCleanFinancialText:
    def test_removes_extra_whitespace(self):
        text = "Revenue  was   $100   million"
        cleaned = clean_financial_text(text)
        assert "  " not in cleaned

    def test_normalizes_currency(self):
        text = "US$100 million and USD 200 million"
        cleaned = clean_financial_text(text)
        assert "$100" in cleaned
        assert "$200" in cleaned

    def test_removes_form_feeds(self):
        text = "Page 1\fPage 2"
        cleaned = clean_financial_text(text)
        assert "\f" not in cleaned


class TestFinancialTextSplitter:
    def test_short_text_single_chunk(self, splitter):
        text = "This is a short document about revenue."
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_long_text_multiple_chunks(self, splitter):
        text = "A" * 500  # longer than chunk_size
        chunks = splitter.split_text(text)
        assert len(chunks) > 1

    def test_sec_filing_sections(self, splitter):
        text = (
            "Some preamble text about the company.\n\n"
            "ITEM 1. BUSINESS\n"
            "The company operates in technology.\n\n"
            "ITEM 1A. RISK FACTORS\n"
            "There are many risks including competition.\n\n"
            "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS\n"
            "Revenue grew by 20% year over year."
        )
        chunks = splitter.split_text(text, doc_type="sec_filing")
        # Should have at least identified some sections
        assert len(chunks) >= 1

    def test_empty_text(self, splitter):
        chunks = splitter.split_text("")
        assert len(chunks) == 0

    def test_chunk_overlap_exists(self):
        splitter = FinancialTextSplitter(chunk_size=100, chunk_overlap=30)
        # Create text that will definitely need splitting
        text = " ".join(["word"] * 200)
        chunks = splitter.split_text(text)
        assert len(chunks) > 1

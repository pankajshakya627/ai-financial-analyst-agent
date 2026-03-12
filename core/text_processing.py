"""
Text chunking and preprocessing utilities for financial documents.
Handles intelligent splitting of SEC filings, earnings transcripts, and news.
"""

import re
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A processed text chunk with metadata."""

    text: str
    start_idx: int
    end_idx: int
    section: Optional[str] = None
    chunk_index: int = 0


class FinancialTextSplitter:
    """
    Intelligent text splitter tuned for financial documents.
    Respects section boundaries in SEC filings and speaker turns in earnings calls.
    """

    # SEC filing section headers (10-K / 10-Q)
    SEC_SECTION_PATTERNS = [
        r"(?i)ITEM\s+\d+[A-Z]?\.\s+",  # ITEM 1. / ITEM 1A.
        r"(?i)PART\s+[IVX]+",  # PART I, PART II, etc.
        r"(?i)MANAGEMENT'?S?\s+DISCUSSION\s+AND\s+ANALYSIS",
        r"(?i)RISK\s+FACTORS",
        r"(?i)FINANCIAL\s+STATEMENTS",
        r"(?i)NOTES\s+TO\s+(CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS",
        r"(?i)QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES",
        r"(?i)CONTROLS\s+AND\s+PROCEDURES",
    ]

    # Earnings call section patterns
    EARNINGS_SECTION_PATTERNS = [
        r"(?i)(operator|moderator)\s*:",
        r"(?i)prepared\s+remarks",
        r"(?i)question[s]?\s*(and|&)\s*answer[s]?",
        r"(?i)Q&A\s+session",
        r"(?i)opening\s+remarks",
        r"(?i)closing\s+remarks",
    ]

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, doc_type: str = "generic") -> list[TextChunk]:
        """Split text into chunks, respecting document structure."""
        if doc_type == "sec_filing":
            return self._split_sec_filing(text)
        elif doc_type == "earnings_call":
            return self._split_earnings_call(text)
        else:
            return self._split_generic(text)

    def _split_sec_filing(self, text: str) -> list[TextChunk]:
        """Split SEC filing respecting item/section boundaries."""
        sections = self._extract_sections(text, self.SEC_SECTION_PATTERNS)

        chunks = []
        for section_name, section_text, start_idx in sections:
            section_chunks = self._recursive_split(section_text, start_idx)
            for chunk in section_chunks:
                chunk.section = section_name
            chunks.extend(section_chunks)

        if not chunks:
            chunks = self._recursive_split(text, 0)

        return chunks

    def _split_earnings_call(self, text: str) -> list[TextChunk]:
        """Split earnings call by speaker turns and Q&A sections."""
        # Split by speaker turns (Name: dialogue)
        speaker_pattern = r"\n([A-Z][a-zA-Z\s\.]+)\s*(?:[-–—]|:)\s*"
        segments = re.split(speaker_pattern, text)

        chunks = []
        current_pos = 0

        if len(segments) > 1:
            # Reassemble speaker + dialogue pairs
            for i in range(0, len(segments) - 1, 2):
                segment = segments[i]
                if i + 1 < len(segments):
                    speaker = segments[i + 1] if i > 0 else ""
                    dialogue = segments[i] if i == 0 else segments[i + 1]

                    combined = f"{speaker}: {dialogue}" if speaker else dialogue
                    sub_chunks = self._recursive_split(combined.strip(), current_pos)
                    for chunk in sub_chunks:
                        chunk.section = speaker.strip() if speaker else "Introduction"
                    chunks.extend(sub_chunks)
                    current_pos += len(combined)
        else:
            chunks = self._recursive_split(text, 0)

        return chunks

    def _extract_sections(
        self, text: str, patterns: list[str]
    ) -> list[tuple[str, str, int]]:
        """Extract named sections from text using regex patterns."""
        # Strip leading flags like (?i) from each pattern before combining
        # Flags should be applied once at the start of the combined pattern
        stripped_patterns = []
        for p in patterns:
            # Remove leading (?i) or similar inline flags
            clean_p = re.sub(r"^\(\?[a-z]+\)", "", p)
            stripped_patterns.append(clean_p)

        # Apply case-insensitive flag once to the combined pattern
        combined_pattern = "|".join(f"({p})" for p in stripped_patterns)
        matches = list(re.finditer(combined_pattern, text, re.IGNORECASE))

        if not matches:
            return [("Full Document", text, 0)]

        sections = []
        for i, match in enumerate(matches):
            section_name = match.group().strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end]
            sections.append((section_name, section_text, start))

        # Include any text before the first section
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()]
            if preamble.strip():
                sections.insert(0, ("Preamble", preamble, 0))

        return sections

    def _recursive_split(self, text: str, base_offset: int = 0) -> list[TextChunk]:
        """Recursively split text into chunks, trying paragraph → sentence → character breaks."""
        if len(text) <= self.chunk_size:
            if text.strip():
                return [
                    TextChunk(
                        text=text.strip(),
                        start_idx=base_offset,
                        end_idx=base_offset + len(text),
                        chunk_index=0,
                    )
                ]
            return []

        # Try splitting by paragraphs first
        separators = ["\n\n", "\n", ". ", " "]
        for sep in separators:
            parts = text.split(sep)
            if len(parts) > 1:
                return self._merge_splits(parts, sep, base_offset)

        # Fallback: hard split at chunk_size
        chunks = []
        idx = 0
        chunk_num = 0
        while idx < len(text):
            end = min(idx + self.chunk_size, len(text))
            chunk_text = text[idx:end].strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        start_idx=base_offset + idx,
                        end_idx=base_offset + end,
                        chunk_index=chunk_num,
                    )
                )
                chunk_num += 1
            idx += self.chunk_size - self.chunk_overlap

        return chunks

    def _merge_splits(
        self, parts: list[str], separator: str, base_offset: int
    ) -> list[TextChunk]:
        """Merge small splits into chunks respecting chunk_size."""
        chunks = []
        current_chunk = ""
        current_start = base_offset
        chunk_num = 0

        for part in parts:
            candidate = current_chunk + separator + part if current_chunk else part

            if len(candidate) > self.chunk_size and current_chunk:
                chunks.append(
                    TextChunk(
                        text=current_chunk.strip(),
                        start_idx=current_start,
                        end_idx=current_start + len(current_chunk),
                        chunk_index=chunk_num,
                    )
                )
                chunk_num += 1

                # Overlap: keep the tail of the current chunk
                overlap_text = current_chunk[-self.chunk_overlap :] if self.chunk_overlap else ""
                current_chunk = overlap_text + separator + part if overlap_text else part
                current_start = current_start + len(current_chunk) - len(overlap_text) - len(part)
            else:
                current_chunk = candidate

        if current_chunk.strip():
            chunks.append(
                TextChunk(
                    text=current_chunk.strip(),
                    start_idx=current_start,
                    end_idx=current_start + len(current_chunk),
                    chunk_index=chunk_num,
                )
            )

        return chunks


def clean_financial_text(text: str) -> str:
    """Clean and normalize financial document text."""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Normalize dashes
    text = re.sub(r"[–—]", "-", text)
    # Remove page numbers / headers that repeat
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    # Normalize currency symbols
    text = text.replace("US$", "$").replace("USD ", "$")
    # Remove form feed characters
    text = text.replace("\f", "\n")
    return text.strip()

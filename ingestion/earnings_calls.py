"""
Earnings call transcript ingestion and parsing.
Supports multiple transcript sources and structured extraction of
prepared remarks, Q&A sessions, and speaker-attributed dialogue.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SpeakerTurn:
    """A single speaker turn in an earnings call."""

    speaker: str
    role: str  # CEO, CFO, Analyst, Operator, etc.
    affiliation: str  # Company name or analyst firm
    text: str
    section: str  # prepared_remarks, q_and_a
    turn_index: int = 0


@dataclass
class EarningsCallTranscript:
    """Structured earnings call transcript."""

    company: str
    ticker: str
    quarter: str  # e.g., "Q4 2024"
    call_date: str
    participants: list[dict] = field(default_factory=list)
    prepared_remarks: list[SpeakerTurn] = field(default_factory=list)
    q_and_a: list[SpeakerTurn] = field(default_factory=list)
    raw_text: str = ""
    metadata: dict = field(default_factory=dict)


class EarningsCallProcessor:
    """
    Parse and structure earnings call transcripts from raw text or JSON sources.
    Extracts speaker turns, sections, and key financial mentions.
    """

    # Common executive titles
    EXECUTIVE_PATTERNS = {
        r"(?i)chief\s+executive\s+officer|CEO": "CEO",
        r"(?i)chief\s+financial\s+officer|CFO": "CFO",
        r"(?i)chief\s+operating\s+officer|COO": "COO",
        r"(?i)chief\s+technology\s+officer|CTO": "CTO",
        r"(?i)president": "President",
        r"(?i)vice\s+president|VP": "VP",
        r"(?i)director\s+of\s+investor\s+relations|IR": "IR",
        r"(?i)analyst": "Analyst",
        r"(?i)operator|moderator": "Operator",
    }

    # Financial metric patterns for extraction
    FINANCIAL_PATTERNS = {
        "revenue": r"(?i)revenue[s]?\s+(?:of|was|were|totaled|reached)\s+\$?([\d,\.]+)\s*(million|billion|M|B)?",
        "eps": r"(?i)(?:earnings|EPS|earnings\s+per\s+share)\s+(?:of|was|were)\s+\$?([\d,\.]+)",
        "guidance": r"(?i)(?:guidance|outlook|expect|forecast)[^.]*\$?([\d,\.]+)\s*(million|billion|M|B)?",
        "margin": r"(?i)(?:gross|operating|net|EBITDA)\s+margin[s]?\s+(?:of|was|were|at)\s+([\d,\.]+)\s*%",
        "growth": r"(?i)(?:grew|growth|increased|up)\s+(?:by\s+)?([\d,\.]+)\s*%",
    }

    def __init__(self):
        self.transcripts_dir = Path(settings.earnings_calls_dir)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

    def parse_raw_transcript(
        self,
        text: str,
        company: str,
        ticker: str,
        quarter: str,
        call_date: str,
    ) -> EarningsCallTranscript:
        """Parse a raw text transcript into structured format."""

        # Detect section boundaries
        q_and_a_start = self._find_qa_section(text)

        if q_and_a_start:
            prepared_text = text[:q_and_a_start]
            qa_text = text[q_and_a_start:]
        else:
            # If no clear Q&A boundary, treat everything as prepared remarks
            prepared_text = text
            qa_text = ""

        # Extract speaker turns
        prepared_turns = self._extract_speaker_turns(prepared_text, "prepared_remarks")
        qa_turns = self._extract_speaker_turns(qa_text, "q_and_a")

        # Extract participants
        all_turns = prepared_turns + qa_turns
        participants = self._identify_participants(all_turns)

        transcript = EarningsCallTranscript(
            company=company,
            ticker=ticker,
            quarter=quarter,
            call_date=call_date,
            participants=participants,
            prepared_remarks=prepared_turns,
            q_and_a=qa_turns,
            raw_text=text,
            metadata={
                "total_turns": len(all_turns),
                "prepared_remarks_count": len(prepared_turns),
                "qa_count": len(qa_turns),
                "word_count": len(text.split()),
                "financial_mentions": self._extract_financial_mentions(text),
                "parsed_at": datetime.utcnow().isoformat(),
            },
        )

        self._save_transcript(transcript)
        return transcript

    def parse_json_transcript(self, data: dict) -> EarningsCallTranscript:
        """Parse a transcript from JSON format (e.g., from a transcript API)."""
        company = data.get("company", data.get("companyName", "Unknown"))
        ticker = data.get("ticker", data.get("symbol", ""))
        quarter = data.get("quarter", data.get("period", ""))
        call_date = data.get("date", data.get("callDate", ""))

        # Handle pre-structured transcripts
        if "content" in data and isinstance(data["content"], list):
            prepared_turns = []
            qa_turns = []

            for i, item in enumerate(data["content"]):
                speaker = item.get("speaker", item.get("name", "Unknown"))
                role = self._classify_speaker_role(speaker)
                section = item.get("section", "prepared_remarks")
                text = item.get("text", item.get("speech", ""))

                turn = SpeakerTurn(
                    speaker=speaker,
                    role=role,
                    affiliation=item.get("affiliation", company),
                    text=text,
                    section=section,
                    turn_index=i,
                )

                if section == "q_and_a":
                    qa_turns.append(turn)
                else:
                    prepared_turns.append(turn)

            participants = self._identify_participants(prepared_turns + qa_turns)
            raw_text = "\n\n".join(
                f"{t.speaker}: {t.text}" for t in prepared_turns + qa_turns
            )
        else:
            # Fallback: treat the raw text field
            raw_text = data.get("text", data.get("transcript", ""))
            return self.parse_raw_transcript(raw_text, company, ticker, quarter, call_date)

        transcript = EarningsCallTranscript(
            company=company,
            ticker=ticker,
            quarter=quarter,
            call_date=call_date,
            participants=participants,
            prepared_remarks=prepared_turns,
            q_and_a=qa_turns,
            raw_text=raw_text,
            metadata={
                "source": data.get("source", "api"),
                "parsed_at": datetime.utcnow().isoformat(),
                "financial_mentions": self._extract_financial_mentions(raw_text),
            },
        )

        self._save_transcript(transcript)
        return transcript

    def _find_qa_section(self, text: str) -> Optional[int]:
        """Find where the Q&A section begins."""
        patterns = [
            r"(?i)question[s]?\s*(?:and|&)\s*answer[s]?\s*(?:session)?",
            r"(?i)Q\s*&\s*A\s+(?:session|segment|portion)",
            r"(?i)we\s+(?:will|would)\s+(?:now\s+)?(?:open|begin)\s+(?:the\s+)?(?:floor|call|line)\s+(?:for|to)\s+questions",
            r"(?i)operator\s*:\s*(?:thank you|our first question)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.start()
        return None

    def _extract_speaker_turns(self, text: str, section: str) -> list[SpeakerTurn]:
        """Extract speaker turns from text."""
        # Pattern: "Speaker Name -- Title:" or "Speaker Name:" at start of line
        speaker_pattern = r"\n([A-Z][a-zA-Z\.\s]+?)(?:\s*[-–—]+\s*[A-Za-z\s,\.]+)?\s*:\s*"
        parts = re.split(speaker_pattern, text)

        turns = []
        turn_idx = 0

        for i in range(1, len(parts) - 1, 2):
            speaker = parts[i].strip()
            dialogue = parts[i + 1].strip() if i + 1 < len(parts) else ""

            if not dialogue or len(dialogue) < 10:
                continue

            role = self._classify_speaker_role(speaker)
            turns.append(
                SpeakerTurn(
                    speaker=speaker,
                    role=role,
                    affiliation="",  # Would need context to determine
                    text=dialogue,
                    section=section,
                    turn_index=turn_idx,
                )
            )
            turn_idx += 1

        # If no speaker turns found, return the whole text as one turn
        if not turns and text.strip():
            turns.append(
                SpeakerTurn(
                    speaker="Unknown",
                    role="Unknown",
                    affiliation="",
                    text=text.strip(),
                    section=section,
                    turn_index=0,
                )
            )

        return turns

    def _classify_speaker_role(self, speaker_text: str) -> str:
        """Classify a speaker's role based on their name/title text."""
        for pattern, role in self.EXECUTIVE_PATTERNS.items():
            if re.search(pattern, speaker_text):
                return role
        return "Executive"

    def _identify_participants(self, turns: list[SpeakerTurn]) -> list[dict]:
        """Identify unique participants from speaker turns."""
        seen = {}
        for turn in turns:
            if turn.speaker not in seen:
                seen[turn.speaker] = {
                    "name": turn.speaker,
                    "role": turn.role,
                    "affiliation": turn.affiliation,
                    "turn_count": 0,
                    "sections": set(),
                }
            seen[turn.speaker]["turn_count"] += 1
            seen[turn.speaker]["sections"].add(turn.section)

        participants = []
        for p in seen.values():
            p["sections"] = list(p["sections"])
            participants.append(p)

        return participants

    def _extract_financial_mentions(self, text: str) -> dict:
        """Extract financial metrics mentioned in the transcript."""
        mentions = {}
        for metric, pattern in self.FINANCIAL_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                mentions[metric] = [
                    {"value": m[0] if isinstance(m, tuple) else m}
                    for m in matches[:5]  # Cap at 5 mentions per metric
                ]
        return mentions

    def _save_transcript(self, transcript: EarningsCallTranscript) -> Path:
        """Save a transcript to local storage."""
        safe_ticker = re.sub(r"[^\w\-]", "_", transcript.ticker or "unknown")
        safe_quarter = re.sub(r"[^\w\-]", "_", transcript.quarter or "unknown")
        filename = f"{safe_ticker}_{safe_quarter}_{transcript.call_date}.json"
        filepath = self.transcripts_dir / filename

        data = {
            "company": transcript.company,
            "ticker": transcript.ticker,
            "quarter": transcript.quarter,
            "call_date": transcript.call_date,
            "participants": transcript.participants,
            "metadata": transcript.metadata,
            "prepared_remarks": [
                {"speaker": t.speaker, "role": t.role, "text": t.text[:2000]}
                for t in transcript.prepared_remarks
            ],
            "q_and_a": [
                {"speaker": t.speaker, "role": t.role, "text": t.text[:2000]}
                for t in transcript.q_and_a
            ],
        }

        filepath.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved transcript: {filepath}")
        return filepath

    def get_transcript_summary(self, transcript: EarningsCallTranscript) -> str:
        """Generate a concise summary prompt-ready string from a transcript."""
        parts = [
            f"# {transcript.company} ({transcript.ticker}) - {transcript.quarter} Earnings Call",
            f"Date: {transcript.call_date}",
            f"Participants: {len(transcript.participants)} speakers",
            "",
            "## Key Highlights (Prepared Remarks)",
        ]

        # Include first 3 executive turns from prepared remarks
        exec_turns = [
            t for t in transcript.prepared_remarks if t.role in ("CEO", "CFO", "President")
        ]
        for turn in exec_turns[:3]:
            parts.append(f"\n**{turn.speaker} ({turn.role}):** {turn.text[:500]}...")

        if transcript.metadata.get("financial_mentions"):
            parts.append("\n## Financial Metrics Mentioned")
            for metric, values in transcript.metadata["financial_mentions"].items():
                val_str = ", ".join(v["value"] for v in values[:3])
                parts.append(f"- {metric}: {val_str}")

        return "\n".join(parts)

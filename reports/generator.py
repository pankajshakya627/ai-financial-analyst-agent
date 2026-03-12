"""
Report generation module.
Produces structured financial analysis reports in multiple formats.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import settings
from analysis.financial_analyzer import AnalysisResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate formatted reports from analysis results."""

    def __init__(self):
        self.reports_dir = Path(settings.reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(
        self,
        analysis: AnalysisResult,
        include_raw_data: bool = False,
    ) -> str:
        """Generate a full Markdown report from an analysis result."""
        sections = [
            self._header(analysis),
            self._executive_summary(analysis),
            self._key_metrics_section(analysis),
            self._ratio_analysis_section(analysis),
            self._detailed_analysis_section(analysis),
            self._risks_section(analysis),
            self._catalysts_section(analysis),
            self._sources_section(analysis),
            self._disclaimer(),
        ]

        report = "\n\n".join(s for s in sections if s)

        # Save to disk
        filename = f"{analysis.ticker}_{analysis.analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = self.reports_dir / filename
        filepath.write_text(report)
        logger.info(f"Report saved: {filepath}")

        return report

    def generate_json_report(self, analysis: AnalysisResult) -> dict:
        """Generate a structured JSON report."""
        report = {
            "metadata": {
                "ticker": analysis.ticker,
                "analysis_type": analysis.analysis_type,
                "generated_at": datetime.utcnow().isoformat(),
                "disclaimer": "This report is for informational purposes only and does not constitute investment advice.",
            },
            "executive_summary": analysis.summary,
            "key_metrics": analysis.key_metrics,
            "financial_ratios": analysis.ratios,
            "detailed_analysis": analysis.detailed_analysis,
            "risks": analysis.risks,
            "catalysts": analysis.catalysts,
            "sources": analysis.sources,
        }

        filename = f"{analysis.ticker}_{analysis.analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.reports_dir / filename
        filepath.write_text(json.dumps(report, indent=2, default=str))
        logger.info(f"JSON report saved: {filepath}")

        return report

    def _header(self, analysis: AnalysisResult) -> str:
        return f"""# Financial Analysis Report: {analysis.ticker}

**Analysis Type:** {analysis.analysis_type.replace('_', ' ').title()}
**Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M UTC')}

---"""

    def _executive_summary(self, analysis: AnalysisResult) -> str:
        return f"""## Executive Summary

{analysis.summary}"""

    def _key_metrics_section(self, analysis: AnalysisResult) -> str:
        if not analysis.key_metrics:
            return ""

        lines = ["## Key Financial Metrics\n"]
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key, value in analysis.key_metrics.items():
            display_key = key.replace("_", " ").title()
            if isinstance(value, (int, float)):
                if abs(value) >= 1_000_000_000:
                    display_val = f"${value / 1_000_000_000:,.1f}B"
                elif abs(value) >= 1_000_000:
                    display_val = f"${value / 1_000_000:,.1f}M"
                else:
                    display_val = f"${value:,.2f}"
            else:
                display_val = str(value)
            lines.append(f"| {display_key} | {display_val} |")

        return "\n".join(lines)

    def _ratio_analysis_section(self, analysis: AnalysisResult) -> str:
        if not analysis.ratios:
            return ""

        lines = ["## Financial Ratio Analysis\n"]

        # Group by category
        categories = {}
        for ratio in analysis.ratios:
            cat = ratio.get("category", "other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(ratio)

        for category, ratios in categories.items():
            lines.append(f"\n### {category.replace('_', ' ').title()}\n")
            lines.append("| Ratio | Value | Benchmark |")
            lines.append("|-------|-------|-----------|")
            for r in ratios:
                val = r.get("value")
                val_str = f"{val:.4f}" if val is not None else "N/A"
                benchmark = r.get("benchmark", "")
                lines.append(f"| {r['name']} | {val_str} | {benchmark} |")

        return "\n".join(lines)

    def _detailed_analysis_section(self, analysis: AnalysisResult) -> str:
        return f"""## Detailed Analysis

{analysis.detailed_analysis}"""

    def _risks_section(self, analysis: AnalysisResult) -> str:
        if not analysis.risks:
            return ""

        lines = ["## Key Risks\n"]
        for risk in analysis.risks:
            lines.append(f"- {risk}")
        return "\n".join(lines)

    def _catalysts_section(self, analysis: AnalysisResult) -> str:
        if not analysis.catalysts:
            return ""

        lines = ["## Growth Catalysts\n"]
        for catalyst in analysis.catalysts:
            lines.append(f"- {catalyst}")
        return "\n".join(lines)

    def _sources_section(self, analysis: AnalysisResult) -> str:
        if not analysis.sources:
            return ""

        lines = ["## Sources\n"]
        for source in analysis.sources:
            source_type = source.get("type", "unknown").replace("_", " ").title()
            detail = source.get("detail", "")
            date = source.get("date", "")
            url = source.get("url", "")
            if url:
                lines.append(f"- [{source_type}: {detail} ({date})]({url})")
            else:
                lines.append(f"- {source_type}: {detail} ({date})")
        return "\n".join(lines)

    def _disclaimer(self) -> str:
        return """---

**Disclaimer:** This report is generated by an AI Financial Analyst Agent for
informational purposes only. It does not constitute investment advice, financial
advice, trading advice, or any other sort of advice. You should not treat any of
the report's content as such. The information provided is based on publicly
available data and AI analysis, which may contain errors or omissions.
Always consult with a qualified financial professional before making investment decisions."""

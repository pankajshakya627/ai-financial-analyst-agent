"""
High-level financial analysis engine.
Combines financial data, ratios, and RAG context to produce
structured analysis using the LLM.
"""

import json
import logging
from typing import Optional
from dataclasses import dataclass, field

from analysis.financial_data import FinancialDataClient, FinancialStatement
from analysis.ratios import FinancialRatioCalculator, RatioResult
from rag.retriever import FinancialRAGRetriever
from core.llm_provider import LLMClient

logger = logging.getLogger(__name__)


FINANCIAL_ANALYST_SYSTEM_PROMPT = """You are a senior financial analyst at an enterprise investment firm.
Your analysis should be:
- Data-driven with specific numbers and calculations
- Structured using GAAP-standard presentation
- Balanced, presenting both bull and bear cases
- Actionable with clear conclusions

When analyzing financial data, always:
1. Start with key metrics and trends
2. Compare to industry benchmarks when available
3. Identify risks and catalysts
4. Provide forward-looking commentary based on available guidance
5. Cite specific data points from SEC filings and earnings calls

Important: This is financial analysis for informational purposes only, not investment advice.
Always disclaim that professional financial advice should be sought for investment decisions."""


@dataclass
class AnalysisResult:
    """Structured output from the analysis engine."""

    ticker: str
    analysis_type: str
    summary: str
    detailed_analysis: str
    key_metrics: dict
    ratios: list[dict]
    risks: list[str]
    catalysts: list[str]
    sources: list[dict]
    raw_data: dict = field(default_factory=dict)


class FinancialAnalyzer:
    """
    Orchestrates financial analysis by combining:
    1. Structured financial data from APIs
    2. RAG-retrieved context from SEC filings / earnings calls
    3. LLM-powered synthesis and narrative generation
    """

    def __init__(
        self,
        llm_client: LLMClient,
        data_client: Optional[FinancialDataClient] = None,
        rag_retriever: Optional[FinancialRAGRetriever] = None,
    ):
        self.llm = llm_client
        self.data_client = data_client or FinancialDataClient()
        self.rag = rag_retriever or FinancialRAGRetriever()
        self.ratio_calculator = FinancialRatioCalculator()

    async def comprehensive_analysis(self, ticker: str) -> AnalysisResult:
        """
        Full company analysis combining structured data + RAG + LLM.
        Produces an investment-research-grade report.
        """
        logger.info(f"Starting comprehensive analysis for {ticker}")

        # 1. Fetch structured financial data
        financials = await self.data_client.get_full_financials(ticker)

        # 2. Calculate ratios
        ratios = self._compute_ratios(financials)

        # 3. Retrieve RAG context
        rag_result = self.rag.retrieve_for_analysis(ticker, "comprehensive")

        # 4. Build the analysis prompt
        prompt = self._build_analysis_prompt(
            ticker=ticker,
            financials=financials,
            ratios=ratios,
            rag_context=rag_result.context_text,
            analysis_type="comprehensive",
        )

        # 5. Generate analysis with LLM
        analysis_text = await self.llm.generate(
            prompt=prompt,
            system_prompt=FINANCIAL_ANALYST_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=4096,
        )

        # 6. Extract structured components
        key_metrics = self._extract_key_metrics(financials)

        return AnalysisResult(
            ticker=ticker,
            analysis_type="comprehensive",
            summary=self._extract_summary(analysis_text),
            detailed_analysis=analysis_text,
            key_metrics=key_metrics,
            ratios=[
                {
                    "name": r.name,
                    "value": round(r.value, 4) if r.value else None,
                    "category": r.category,
                    "interpretation": r.interpretation,
                    "benchmark": r.benchmark,
                }
                for r in ratios
            ],
            risks=self._extract_section(analysis_text, "risk"),
            catalysts=self._extract_section(analysis_text, "catalyst"),
            sources=rag_result.sources,
            raw_data={"financials_available": list(k for k, v in financials.items() if v is not None)},
        )

    async def earnings_analysis(self, ticker: str) -> AnalysisResult:
        """Analyze recent earnings performance and guidance."""
        financials = await self.data_client.get_full_financials(ticker)
        rag_result = self.rag.retrieve_for_analysis(ticker, "earnings")
        ratios = self._compute_ratios(financials)

        prompt = self._build_analysis_prompt(
            ticker=ticker,
            financials=financials,
            ratios=ratios,
            rag_context=rag_result.context_text,
            analysis_type="earnings",
        )

        analysis_text = await self.llm.generate(
            prompt=prompt,
            system_prompt=FINANCIAL_ANALYST_SYSTEM_PROMPT,
            temperature=0.1,
        )

        return AnalysisResult(
            ticker=ticker,
            analysis_type="earnings",
            summary=self._extract_summary(analysis_text),
            detailed_analysis=analysis_text,
            key_metrics=self._extract_key_metrics(financials),
            ratios=[{"name": r.name, "value": r.value, "category": r.category} for r in ratios],
            risks=self._extract_section(analysis_text, "risk"),
            catalysts=self._extract_section(analysis_text, "catalyst"),
            sources=rag_result.sources,
        )

    async def risk_analysis(self, ticker: str) -> AnalysisResult:
        """Deep dive into company risk factors."""
        rag_result = self.rag.retrieve_for_analysis(ticker, "risk")

        prompt = f"""Perform a comprehensive risk analysis for {ticker} based on the following
SEC filing context and earnings call commentary.

{rag_result.context_text}

Structure your analysis as:
1. **Operational Risks**: Business execution, supply chain, talent, technology
2. **Financial Risks**: Leverage, liquidity, currency, interest rate exposure
3. **Regulatory/Legal Risks**: Compliance, litigation, regulatory changes
4. **Market/Competitive Risks**: Competition, market shifts, disruption threats
5. **Macroeconomic Risks**: Economic cycle, geopolitical, inflation
6. **ESG Risks**: Environmental, social, governance concerns

For each risk, assess:
- Likelihood (high/medium/low)
- Potential impact (high/medium/low)
- Mitigating factors
- Monitoring indicators

Conclude with a risk matrix summary and overall risk assessment.

Important disclaimer: This risk analysis is for informational purposes only."""

        analysis_text = await self.llm.generate(
            prompt=prompt,
            system_prompt=FINANCIAL_ANALYST_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=4096,
        )

        return AnalysisResult(
            ticker=ticker,
            analysis_type="risk",
            summary=self._extract_summary(analysis_text),
            detailed_analysis=analysis_text,
            key_metrics={},
            ratios=[],
            risks=self._extract_section(analysis_text, "risk"),
            catalysts=[],
            sources=rag_result.sources,
        )

    async def valuation_analysis(self, ticker: str) -> AnalysisResult:
        """DCF-oriented valuation analysis."""
        financials = await self.data_client.get_full_financials(ticker)
        rag_result = self.rag.retrieve_for_analysis(ticker, "valuation")
        ratios = self._compute_ratios(financials)

        # Extract valuation-specific ratios
        val_ratios = [r for r in ratios if r.category == "valuation"]

        prompt = f"""Perform a valuation analysis for {ticker}.

## Available Financial Data
{self._format_financials_for_prompt(financials)}

## Calculated Valuation Ratios
{self._format_ratios_for_prompt(val_ratios)}

## SEC Filing & Earnings Context
{rag_result.context_text}

Please provide:
1. **Comparable Company Analysis**: Identify key valuation multiples and how they compare
   to sector peers (use your knowledge of typical sector ranges)
2. **DCF Framework**: Outline key assumptions for a DCF:
   - Revenue growth trajectory
   - Margin expansion/contraction path
   - Capital intensity
   - Appropriate discount rate range (WACC)
   - Terminal growth rate
3. **Sum-of-Parts** (if applicable): Segment-level valuation
4. **Implied Valuation Range**: Based on multiple approaches
5. **Key Sensitivities**: What assumptions drive the most variation

Important disclaimer: This valuation analysis is for informational purposes only
and should not be used as investment advice. Consult a financial professional."""

        analysis_text = await self.llm.generate(
            prompt=prompt,
            system_prompt=FINANCIAL_ANALYST_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=4096,
        )

        return AnalysisResult(
            ticker=ticker,
            analysis_type="valuation",
            summary=self._extract_summary(analysis_text),
            detailed_analysis=analysis_text,
            key_metrics=self._extract_key_metrics(financials),
            ratios=[{"name": r.name, "value": r.value, "category": r.category} for r in ratios],
            risks=[],
            catalysts=[],
            sources=rag_result.sources,
        )

    async def answer_question(self, question: str, ticker: Optional[str] = None) -> dict:
        """
        Answer a free-form financial question using RAG + data APIs.
        """
        # Retrieve relevant context
        rag_result = self.rag.retrieve(query=question, ticker=ticker)

        # Optionally fetch real-time data if a ticker is specified
        financial_context = ""
        if ticker:
            try:
                profile = await self.data_client.get_company_profile(ticker)
                financial_context = (
                    f"\n## Company Profile: {profile.name} ({ticker})\n"
                    f"Sector: {profile.sector} | Industry: {profile.industry}\n"
                    f"Market Cap: ${profile.market_cap:,.0f}\n"
                )
            except Exception:
                pass

        prompt = f"""Answer the following financial question using the provided context.

## Question
{question}

{financial_context}

## Retrieved Context
{rag_result.context_text}

Instructions:
- Be specific and cite data points when available
- If the context doesn't contain enough information, say so clearly
- Distinguish between facts (from filings) and interpretations
- Include relevant caveats about data recency

Important: This response is for informational purposes only and does not constitute financial advice."""

        answer = await self.llm.generate(
            prompt=prompt,
            system_prompt=FINANCIAL_ANALYST_SYSTEM_PROMPT,
            temperature=0.2,
        )

        return {
            "question": question,
            "answer": answer,
            "sources": rag_result.sources,
            "collections_searched": rag_result.collection_stats,
        }

    def _compute_ratios(self, financials: dict) -> list[RatioResult]:
        """Compute ratios from the fetched financial data."""
        income_data = {}
        balance_data = {}
        cash_flow_data = {}
        market_cap = None
        stock_price = None

        if financials.get("income_statement"):
            statements = financials["income_statement"]
            if statements:
                income_data = statements[0].data if isinstance(statements[0], FinancialStatement) else statements[0]

        if financials.get("balance_sheet"):
            statements = financials["balance_sheet"]
            if statements:
                balance_data = statements[0].data if isinstance(statements[0], FinancialStatement) else statements[0]

        if financials.get("cash_flow"):
            statements = financials["cash_flow"]
            if statements:
                cash_flow_data = statements[0].data if isinstance(statements[0], FinancialStatement) else statements[0]

        if financials.get("profile"):
            profile = financials["profile"]
            market_cap = getattr(profile, "market_cap", None)

        if financials.get("stock_price"):
            prices = financials["stock_price"]
            if prices:
                stock_price = prices[0].close if hasattr(prices[0], "close") else None

        return self.ratio_calculator.calculate_all_ratios(
            income=income_data,
            balance=balance_data,
            cash_flow=cash_flow_data,
            market_cap=market_cap,
            stock_price=stock_price,
        )

    def _build_analysis_prompt(
        self,
        ticker: str,
        financials: dict,
        ratios: list[RatioResult],
        rag_context: str,
        analysis_type: str,
    ) -> str:
        """Build a comprehensive analysis prompt for the LLM."""
        formatted_financials = self._format_financials_for_prompt(financials)
        formatted_ratios = self._format_ratios_for_prompt(ratios)

        type_instructions = {
            "comprehensive": """Provide a complete investment analysis covering:
1. Business Overview & Competitive Position
2. Financial Performance (revenue trends, profitability, margins)
3. Balance Sheet Strength (leverage, liquidity)
4. Cash Flow Analysis
5. Valuation Assessment
6. Key Risks
7. Growth Catalysts
8. Investment Thesis (bull case and bear case)""",
            "earnings": """Focus on recent earnings performance:
1. Revenue & EPS vs. estimates
2. Margin trends
3. Segment performance
4. Management guidance & commentary
5. Key takeaways from Q&A
6. Implications for forward estimates""",
        }

        return f"""Analyze {ticker} based on the following data and context.

## Structured Financial Data
{formatted_financials}

## Calculated Financial Ratios
{formatted_ratios}

## SEC Filings & Earnings Call Context (from RAG)
{rag_context}

## Analysis Instructions
{type_instructions.get(analysis_type, type_instructions["comprehensive"])}

Important disclaimer: This analysis is for informational purposes only
and does not constitute investment advice."""

    def _format_financials_for_prompt(self, financials: dict) -> str:
        """Format financial data into a readable string for the LLM prompt."""
        parts = []

        if financials.get("profile"):
            p = financials["profile"]
            if hasattr(p, "name"):
                parts.append(f"Company: {p.name} | Sector: {p.sector} | Market Cap: ${p.market_cap:,.0f}")

        if financials.get("income_statement"):
            parts.append("\n### Income Statement (Most Recent)")
            stmt = financials["income_statement"][0]
            data = stmt.data if isinstance(stmt, FinancialStatement) else stmt
            for key in ["revenue", "totalRevenue", "grossProfit", "operatingIncome", "netIncome", "ebitda", "eps"]:
                if key in data and data[key]:
                    parts.append(f"  {key}: {data[key]}")

        if financials.get("balance_sheet"):
            parts.append("\n### Balance Sheet (Most Recent)")
            stmt = financials["balance_sheet"][0]
            data = stmt.data if isinstance(stmt, FinancialStatement) else stmt
            for key in ["totalAssets", "totalCurrentAssets", "cashAndCashEquivalents",
                        "totalLiabilities", "totalCurrentLiabilities", "totalDebt",
                        "totalStockholdersEquity"]:
                if key in data and data[key]:
                    parts.append(f"  {key}: {data[key]}")

        if financials.get("cash_flow"):
            parts.append("\n### Cash Flow (Most Recent)")
            stmt = financials["cash_flow"][0]
            data = stmt.data if isinstance(stmt, FinancialStatement) else stmt
            for key in ["operatingCashFlow", "capitalExpenditure", "freeCashFlow",
                        "dividendsPaid", "commonStockRepurchased"]:
                if key in data and data[key]:
                    parts.append(f"  {key}: {data[key]}")

        return "\n".join(parts) if parts else "No structured financial data available."

    def _format_ratios_for_prompt(self, ratios: list[RatioResult]) -> str:
        """Format ratios into a readable string."""
        if not ratios:
            return "No ratios calculated."

        parts = []
        current_category = ""
        for r in ratios:
            if r.category != current_category:
                current_category = r.category
                parts.append(f"\n**{current_category.replace('_', ' ').title()}**")
            val_str = f"{r.value:.4f}" if r.value is not None else "N/A"
            parts.append(f"  {r.name}: {val_str} ({r.benchmark or ''})")
        return "\n".join(parts)

    def _extract_key_metrics(self, financials: dict) -> dict:
        """Extract headline metrics from financial data."""
        metrics = {}
        if financials.get("income_statement"):
            stmt = financials["income_statement"][0]
            data = stmt.data if isinstance(stmt, FinancialStatement) else stmt
            metrics["revenue"] = data.get("revenue") or data.get("totalRevenue")
            metrics["net_income"] = data.get("netIncome")
            metrics["eps"] = data.get("eps") or data.get("epsdiluted")

        if financials.get("profile"):
            p = financials["profile"]
            metrics["market_cap"] = getattr(p, "market_cap", None)
            metrics["sector"] = getattr(p, "sector", None)

        return {k: v for k, v in metrics.items() if v is not None}

    @staticmethod
    def _extract_summary(text: str) -> str:
        """Extract a summary from the first paragraph of the analysis."""
        paragraphs = text.strip().split("\n\n")
        for p in paragraphs:
            clean = p.strip().lstrip("#").strip()
            if len(clean) > 50:
                return clean[:500]
        return text[:500]

    @staticmethod
    def _extract_section(text: str, keyword: str) -> list[str]:
        """Extract bullet points from a section containing the keyword."""
        items = []
        lines = text.split("\n")
        in_section = False
        for line in lines:
            if keyword.lower() in line.lower() and ("#" in line or "**" in line):
                in_section = True
                continue
            if in_section:
                if line.startswith("#") or (line.startswith("**") and line.endswith("**")):
                    break
                clean = line.strip().lstrip("-•*").strip()
                if clean and len(clean) > 10:
                    items.append(clean)
        return items[:10]

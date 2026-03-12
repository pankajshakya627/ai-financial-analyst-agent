"""
Specialized Agent Handlers.
Each agent handles a specific category of financial tasks.
"""

import logging
from typing import Optional
from dataclasses import dataclass, field

from core.llm_provider import LLMClient
from core.vector_store import VectorStoreManager
from rag.retriever import FinancialRAGRetriever
from analysis.financial_data import FinancialDataClient
from analysis.financial_analyzer import FinancialAnalyzer
from analysis.ratios import FinancialRatioCalculator
from ingestion.pipeline import IngestionPipeline
from agent.intent_detector import IntentCategory, DetectedIntent

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from a specialized agent."""
    
    success: bool
    content: str
    data: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)
    error: Optional[str] = None


class ResearchAgent:
    """
    Handles information retrieval queries.
    Specializes in SEC filings, earnings calls, and news search.
    """
    
    def __init__(
        self,
        llm: LLMClient,
        vector_store: VectorStoreManager,
        rag: FinancialRAGRetriever,
    ):
        self.llm = llm
        self.vector_store = vector_store
        self.rag = rag
    
    async def handle(self, intent: DetectedIntent) -> AgentResult:
        """Handle research queries based on intent."""
        try:
            if intent.category == IntentCategory.SEC_FILING_QUERY:
                return await self._search_sec_filings(intent)
            elif intent.category == IntentCategory.EARNINGS_QUERY:
                return await self._search_earnings_calls(intent)
            elif intent.category == IntentCategory.NEWS_QUERY:
                return await self._search_news(intent)
            else:
                return AgentResult(
                    success=False,
                    content="Query not in research agent scope",
                    error=f"Unsupported category: {intent.category}"
                )
        except Exception as e:
            logger.error(f"Research agent error: {e}")
            return AgentResult(success=False, content="", error=str(e))
    
    async def _search_sec_filings(self, intent: DetectedIntent) -> AgentResult:
        """Search SEC filings."""
        ticker = intent.ticker or "unknown"
        query = intent.parameters.get("query", intent.entities.get("query", ""))
        
        result = self.rag.retrieve(
            query=query,
            ticker=ticker,
            collections=[self.rag.vector_store.client.get_or_create_collection(
                name=self.rag.vector_store.client.list_collections()[0].name
            )],
        )
        
        documents = [
            {
                "content": d.content,
                "metadata": d.metadata,
                "score": d.score,
            }
            for d in result.documents[:10]
        ]
        
        return AgentResult(
            success=True,
            content=f"Found {len(documents)} relevant SEC filing sections for {ticker}",
            data={"documents": documents, "context": result.context_text},
            sources=result.sources,
        )
    
    async def _search_earnings_calls(self, intent: DetectedIntent) -> AgentResult:
        """Search earnings call transcripts."""
        ticker = intent.ticker or "unknown"
        query = intent.parameters.get("query", "")
        
        # Similar implementation to SEC filings
        # (would need to specify earnings collection)
        return AgentResult(
            success=True,
            content=f"Earnings call search for {ticker}: {query}",
            data={"query": query, "ticker": ticker},
        )
    
    async def _search_news(self, intent: DetectedIntent) -> AgentResult:
        """Search financial news."""
        ticker = intent.ticker or "unknown"
        query = intent.parameters.get("query", "")
        
        return AgentResult(
            success=True,
            content=f"News search for {ticker}: {query}",
            data={"query": query, "ticker": ticker},
        )


class AnalysisAgent:
    """
    Handles analytical queries.
    Specializes in comprehensive analysis, risk assessment, valuation, and ratios.
    """
    
    def __init__(
        self,
        llm: LLMClient,
        data_client: FinancialDataClient,
        analyzer: FinancialAnalyzer,
        ratio_calc: FinancialRatioCalculator,
    ):
        self.llm = llm
        self.data_client = data_client
        self.analyzer = analyzer
        self.ratio_calc = ratio_calc
    
    async def handle(self, intent: DetectedIntent) -> AgentResult:
        """Handle analysis queries based on intent."""
        try:
            ticker = intent.ticker
            if not ticker:
                return AgentResult(
                    success=False,
                    content="Ticker symbol required for analysis",
                    error="No ticker provided"
                )
            
            if intent.category == IntentCategory.COMPREHENSIVE_ANALYSIS:
                return await self._comprehensive_analysis(ticker, intent)
            elif intent.category == IntentCategory.RISK_ANALYSIS:
                return await self._risk_analysis(ticker, intent)
            elif intent.category == IntentCategory.VALUATION_ANALYSIS:
                return await self._valuation_analysis(ticker, intent)
            elif intent.category == IntentCategory.RATIO_ANALYSIS:
                return await self._ratio_analysis(ticker, intent)
            elif intent.category == IntentCategory.EARNINGS_ANALYSIS:
                return await self._earnings_analysis(ticker, intent)
            else:
                return AgentResult(
                    success=False,
                    content="Query not in analysis agent scope",
                    error=f"Unsupported category: {intent.category}"
                )
        except Exception as e:
            logger.error(f"Analysis agent error: {e}")
            return AgentResult(success=False, content="", error=str(e))
    
    async def _comprehensive_analysis(self, ticker: str, intent: DetectedIntent) -> AgentResult:
        """Run comprehensive analysis."""
        result = await self.analyzer.comprehensive_analysis(ticker)
        
        return AgentResult(
            success=True,
            content=result.summary,
            data={
                "analysis": result.detailed_analysis,
                "key_metrics": result.key_metrics,
                "ratios": result.ratios,
            },
            sources=result.sources,
        )
    
    async def _risk_analysis(self, ticker: str, intent: DetectedIntent) -> AgentResult:
        """Run risk analysis."""
        result = await self.analyzer.risk_analysis(ticker)
        
        return AgentResult(
            success=True,
            content=f"Risk analysis for {ticker}",
            data={
                "risks": result.risks,
                "risk_score": result.risk_score if hasattr(result, 'risk_score') else "N/A",
            },
            sources=result.sources,
        )
    
    async def _valuation_analysis(self, ticker: str, intent: DetectedIntent) -> AgentResult:
        """Run valuation analysis."""
        result = await self.analyzer.valuation_analysis(ticker)
        
        return AgentResult(
            success=True,
            content=f"Valuation analysis for {ticker}",
            data={
                "valuation_metrics": result.key_metrics,
                "analysis": result.detailed_analysis,
            },
            sources=result.sources,
        )
    
    async def _ratio_analysis(self, ticker: str, intent: DetectedIntent) -> AgentResult:
        """Calculate financial ratios."""
        financials = await self.data_client.get_full_financials(ticker)
        ratios = self.analyzer._compute_ratios(financials)
        
        ratio_data = [
            {
                "name": r.name,
                "value": round(r.value, 4) if r.value else None,
                "category": r.category,
                "interpretation": r.interpretation,
            }
            for r in ratios
        ]
        
        return AgentResult(
            success=True,
            content=f"Calculated {len(ratio_data)} financial ratios for {ticker}",
            data={"ratios": ratio_data},
        )
    
    async def _earnings_analysis(self, ticker: str, intent: DetectedIntent) -> AgentResult:
        """Run earnings analysis."""
        result = await self.analyzer.earnings_analysis(ticker)
        
        return AgentResult(
            success=True,
            content=f"Earnings analysis for {ticker}",
            data={
                "analysis": result.detailed_analysis,
                "key_metrics": result.key_metrics,
            },
            sources=result.sources,
        )


class DataAgent:
    """
    Handles data retrieval and management.
    Specializes in financial statements, company profiles, stock prices, and data ingestion.
    """
    
    def __init__(
        self,
        llm: LLMClient,
        data_client: FinancialDataClient,
        ingestion: IngestionPipeline,
    ):
        self.llm = llm
        self.data_client = data_client
        self.ingestion = ingestion
    
    async def handle(self, intent: DetectedIntent) -> AgentResult:
        """Handle data queries based on intent."""
        try:
            if intent.category == IntentCategory.FINANCIAL_DATA_QUERY:
                return await self._get_financial_statements(intent)
            elif intent.category == IntentCategory.COMPANY_PROFILE:
                return await self._get_company_profile(intent)
            elif intent.category == IntentCategory.STOCK_PRICE:
                return await self._get_stock_price(intent)
            elif intent.category == IntentCategory.ANALYST_ESTIMATES:
                return await self._get_analyst_estimates(intent)
            elif intent.category == IntentCategory.INGEST_DATA:
                return await self._ingest_data(intent)
            elif intent.category == IntentCategory.UPDATE_DATA:
                return await self._update_data(intent)
            else:
                return AgentResult(
                    success=False,
                    content="Query not in data agent scope",
                    error=f"Unsupported category: {intent.category}"
                )
        except Exception as e:
            logger.error(f"Data agent error: {e}")
            return AgentResult(success=False, content="", error=str(e))
    
    async def _get_financial_statements(self, intent: DetectedIntent) -> AgentResult:
        """Get financial statements."""
        ticker = intent.ticker
        if not ticker:
            return AgentResult(success=False, content="Ticker required", error="No ticker")
        
        stmt_type = intent.parameters.get("statement_type", "all")
        period = intent.parameters.get("period", "annual")
        limit = intent.parameters.get("limit", 5)
        
        results = {}
        if stmt_type in ("income_statement", "all"):
            stmts = await self.data_client.get_income_statement(ticker, period, limit)
            results["income_statement"] = stmts
        
        if stmt_type in ("balance_sheet", "all"):
            stmts = await self.data_client.get_balance_sheet(ticker, period, limit)
            results["balance_sheet"] = stmts
        
        if stmt_type in ("cash_flow", "all"):
            stmts = await self.data_client.get_cash_flow(ticker, period, limit)
            results["cash_flow"] = stmts
        
        return AgentResult(
            success=True,
            content=f"Retrieved financial statements for {ticker}",
            data={"statements": results, "ticker": ticker},
        )
    
    async def _get_company_profile(self, intent: DetectedIntent) -> AgentResult:
        """Get company profile."""
        ticker = intent.ticker
        if not ticker:
            return AgentResult(success=False, content="Ticker required", error="No ticker")
        
        profile = await self.data_client.get_company_profile(ticker)
        
        return AgentResult(
            success=True,
            content=f"Company profile for {profile.name}",
            data={
                "ticker": profile.ticker,
                "name": profile.name,
                "sector": profile.sector,
                "industry": profile.industry,
                "market_cap": profile.market_cap,
                "description": profile.description,
                "ceo": profile.ceo,
                "employees": profile.employees,
            },
        )
    
    async def _get_stock_price(self, intent: DetectedIntent) -> AgentResult:
        """Get stock price data."""
        ticker = intent.ticker
        if not ticker:
            return AgentResult(success=False, content="Ticker required", error="No ticker")
        
        period = intent.parameters.get("period", "1Y")
        prices = await self.data_client.get_stock_price(ticker, period)
        
        if prices:
            closes = [p.close for p in prices]
            return AgentResult(
                success=True,
                content=f"Stock price data for {ticker}",
                data={
                    "ticker": ticker,
                    "current_price": closes[0],
                    "period_high": max(closes),
                    "period_low": min(closes),
                    "period_return": (closes[0] - closes[-1]) / closes[-1] if closes[-1] else 0,
                },
            )
        
        return AgentResult(success=False, content="No price data available", error="No data")
    
    async def _get_analyst_estimates(self, intent: DetectedIntent) -> AgentResult:
        """Get analyst estimates."""
        ticker = intent.ticker
        if not ticker:
            return AgentResult(success=False, content="Ticker required", error="No ticker")
        
        estimates = await self.data_client.get_analyst_estimates(ticker)
        
        return AgentResult(
            success=True,
            content=f"Analyst estimates for {ticker}",
            data={
                "estimates": [
                    {
                        "period": e.period,
                        "revenue_estimate": e.revenue_estimate,
                        "eps_estimate": e.eps_estimate,
                        "num_analysts": e.num_analysts,
                        "recommendation": e.recommendation,
                    }
                    for e in estimates
                ]
            },
        )
    
    async def _ingest_data(self, intent: DetectedIntent) -> AgentResult:
        """Ingest company data."""
        ticker = intent.ticker
        if not ticker:
            return AgentResult(success=False, content="Ticker required", error="No ticker")
        
        form_types = intent.parameters.get("form_types", ["10-K", "10-Q"])
        count = intent.parameters.get("filing_count", 3)
        
        result = await self.ingestion.ingest_full_company(
            ticker=ticker,
            sec_form_types=form_types,
            sec_count=count,
        )
        
        return AgentResult(
            success=True,
            content=f"Successfully ingested data for {ticker}",
            data=result,
        )
    
    async def _update_data(self, intent: DetectedIntent) -> AgentResult:
        """Update existing data."""
        # Similar to ingest, but could check for existing data first
        return await self._ingest_data(intent)

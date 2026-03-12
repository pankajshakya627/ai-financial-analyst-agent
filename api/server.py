"""
FastAPI server for the AI Financial Analyst Agent.
Provides REST API endpoints for all agent capabilities.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import settings
from core.llm_provider import get_llm_client
from core.vector_store import VectorStoreManager
from agent.orchestrator import FinancialAgentOrchestrator
from analysis.financial_data import FinancialDataClient
from analysis.financial_analyzer import FinancialAnalyzer
from rag.retriever import FinancialRAGRetriever
from ingestion.pipeline import IngestionPipeline
from reports.generator import ReportGenerator

logger = logging.getLogger(__name__)

# --- Global state ---
orchestrator: Optional[FinancialAgentOrchestrator] = None
ingestion_pipeline: Optional[IngestionPipeline] = None
report_generator: Optional[ReportGenerator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global orchestrator, ingestion_pipeline, report_generator

    logger.info("Starting AI Financial Analyst Agent...")

    vector_store = VectorStoreManager()
    llm_client = get_llm_client()

    orchestrator = FinancialAgentOrchestrator(llm_client, vector_store)
    ingestion_pipeline = IngestionPipeline(vector_store)
    report_generator = ReportGenerator()

    logger.info(f"Agent ready | LLM: {settings.llm_provider} | Embedding: {settings.embedding_provider}")
    yield
    logger.info("Shutting down AI Financial Analyst Agent...")


app = FastAPI(
    title="AI Financial Analyst Agent",
    description=(
        "Enterprise RAG-powered financial analysis platform. "
        "Combines SEC filings, earnings calls, financial data APIs, "
        "and LLM analysis to produce investment-research-grade insights."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---


class QueryRequest(BaseModel):
    """User query request."""

    query: str = Field(..., description="Natural language financial question")
    ticker: Optional[str] = Field(None, description="Stock ticker to focus on")


class QueryResponse(BaseModel):
    """Agent response."""

    answer: str
    tools_used: list[str]
    sources: list[dict]


class IngestRequest(BaseModel):
    """Data ingestion request."""

    ticker: str = Field(..., description="Stock ticker symbol")
    form_types: list[str] = Field(default=["10-K", "10-Q"], description="SEC form types")
    filing_count: int = Field(default=3, description="Number of filings per type")
    include_news: bool = Field(default=True, description="Also ingest financial news")
    news_days_back: int = Field(default=30, description="Days of news to fetch")


class AnalysisRequest(BaseModel):
    """Analysis request."""

    ticker: str = Field(..., description="Stock ticker symbol")
    analysis_type: str = Field(
        default="comprehensive",
        description="Type: comprehensive, earnings, risk, valuation",
    )
    generate_report: bool = Field(default=True, description="Generate a full report file")


class EarningsCallRequest(BaseModel):
    """Earnings call ingestion request."""

    raw_text: str = Field(..., description="Raw earnings call transcript text")
    company: str = Field(..., description="Company name")
    ticker: str = Field(..., description="Stock ticker")
    quarter: str = Field(..., description="Quarter (e.g., Q4 2024)")
    call_date: str = Field(..., description="Call date (YYYY-MM-DD)")


# --- API Endpoints ---


@app.get("/health")
async def health_check():
    """Service health check."""
    return {
        "status": "healthy",
        "llm_provider": settings.llm_provider.value,
        "embedding_provider": settings.embedding_provider.value,
    }


@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Main query endpoint — ask any financial question.
    The agent will automatically route to the right tools.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    response = await orchestrator.process_query(request.query)
    return QueryResponse(
        answer=response.answer,
        tools_used=response.tools_used,
        sources=response.sources,
    )


@app.post("/api/v1/ingest")
async def ingest_company(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest SEC filings and news for a company into the knowledge base.
    Runs in the background for large ingestions.
    """
    if not ingestion_pipeline:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not initialized")

    # Run ingestion (could be backgrounded for large jobs)
    result = await ingestion_pipeline.ingest_full_company(
        ticker=request.ticker,
        sec_form_types=request.form_types,
        sec_count=request.filing_count,
        news_days_back=request.news_days_back if request.include_news else 0,
    )

    return {"status": "completed", "ticker": request.ticker, "results": result}


@app.post("/api/v1/ingest/earnings-call")
async def ingest_earnings_call(request: EarningsCallRequest):
    """Ingest an earnings call transcript."""
    if not ingestion_pipeline:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not initialized")

    result = await ingestion_pipeline.ingest_earnings_call(
        raw_text=request.raw_text,
        company=request.company,
        ticker=request.ticker,
        quarter=request.quarter,
        call_date=request.call_date,
    )

    return {"status": "completed", "ticker": request.ticker, "results": result}


@app.post("/api/v1/analyze")
async def run_analysis(request: AnalysisRequest):
    """
    Run a structured financial analysis on a company.
    Returns analysis with optional report generation.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    analyzer = orchestrator.analyzer

    if request.analysis_type == "comprehensive":
        analysis = await analyzer.comprehensive_analysis(request.ticker)
    elif request.analysis_type == "earnings":
        analysis = await analyzer.earnings_analysis(request.ticker)
    elif request.analysis_type == "risk":
        analysis = await analyzer.risk_analysis(request.ticker)
    elif request.analysis_type == "valuation":
        analysis = await analyzer.valuation_analysis(request.ticker)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown analysis type: {request.analysis_type}")

    response = {
        "ticker": analysis.ticker,
        "analysis_type": analysis.analysis_type,
        "summary": analysis.summary,
        "detailed_analysis": analysis.detailed_analysis,
        "key_metrics": analysis.key_metrics,
        "ratios": analysis.ratios,
        "risks": analysis.risks,
        "catalysts": analysis.catalysts,
        "sources": analysis.sources,
    }

    if request.generate_report and report_generator:
        report_md = report_generator.generate_markdown_report(analysis)
        report_json = report_generator.generate_json_report(analysis)
        response["report_generated"] = True

    return response


@app.get("/api/v1/financials/{ticker}")
async def get_financials(
    ticker: str,
    statement: str = "all",
    period: str = "annual",
    limit: int = 5,
):
    """Get structured financial statement data for a company."""
    data_client = FinancialDataClient()

    result = {}
    try:
        if statement in ("income_statement", "all"):
            stmts = await data_client.get_income_statement(ticker, period, limit)
            result["income_statement"] = [{"date": s.date, "data": s.data} for s in stmts]

        if statement in ("balance_sheet", "all"):
            stmts = await data_client.get_balance_sheet(ticker, period, limit)
            result["balance_sheet"] = [{"date": s.date, "data": s.data} for s in stmts]

        if statement in ("cash_flow", "all"):
            stmts = await data_client.get_cash_flow(ticker, period, limit)
            result["cash_flow"] = [{"date": s.date, "data": s.data} for s in stmts]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch financials: {e}")

    return {"ticker": ticker, "period": period, "statements": result}


@app.get("/api/v1/profile/{ticker}")
async def get_profile(ticker: str):
    """Get company profile."""
    data_client = FinancialDataClient()
    try:
        profile = await data_client.get_company_profile(ticker)
        return {
            "ticker": profile.ticker,
            "name": profile.name,
            "sector": profile.sector,
            "industry": profile.industry,
            "market_cap": profile.market_cap,
            "description": profile.description,
            "exchange": profile.exchange,
            "ceo": profile.ceo,
            "employees": profile.employees,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ratios/{ticker}")
async def get_ratios(ticker: str):
    """Calculate and return financial ratios for a company."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    data_client = FinancialDataClient()
    financials = await data_client.get_full_financials(ticker)
    ratios = orchestrator.analyzer._compute_ratios(financials)

    return {
        "ticker": ticker,
        "ratios": [
            {
                "name": r.name,
                "value": round(r.value, 4) if r.value else None,
                "category": r.category,
                "interpretation": r.interpretation,
                "formula": r.formula,
                "benchmark": r.benchmark,
            }
            for r in ratios
        ],
    }


@app.get("/api/v1/knowledge-base/stats")
async def knowledge_base_stats():
    """Get statistics about the vector store collections."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    vs = orchestrator.vector_store
    collections = vs.list_collections()

    stats = {}
    for name in collections:
        stats[name] = vs.get_collection_stats(name)

    return {"collections": stats, "total_collections": len(collections)}


@app.post("/api/v1/conversation/reset")
async def reset_conversation():
    """Reset the agent's conversation history."""
    if orchestrator:
        orchestrator.reset_conversation()
    return {"status": "conversation reset"}


@app.get("/api/v1/config")
async def get_config():
    """Get current (non-sensitive) configuration."""
    return {
        "llm_provider": settings.llm_provider.value,
        "llm_model": (
            settings.anthropic_model
            if settings.llm_provider.value == "anthropic"
            else settings.openai_model
        ),
        "embedding_provider": settings.embedding_provider.value,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k_results": settings.top_k_results,
        "financial_apis": {
            "fmp": bool(settings.fmp_api_key),
            "alpha_vantage": bool(settings.alpha_vantage_api_key),
            "finnhub": bool(settings.finnhub_api_key),
            "polygon": bool(settings.polygon_api_key),
        },
    }

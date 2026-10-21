"""
FastAPI server for the AI Financial Analyst Agent.
Provides REST API endpoints for all agent capabilities.
"""

import asyncio
import json
import logging
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from api.security import authenticate
from config.settings import settings
from core.security import install_log_redaction, redact_secrets
from core.llm_provider import get_llm_client
from core.vector_store import VectorStoreManager
from agent.orchestrator import FinancialAgentOrchestrator
from analysis.financial_data import FinancialDataClient
from analysis.trends import compute_trends, compute_ttm
from ingestion.pipeline import IngestionPipeline
from reports.generator import ReportGenerator

logger = logging.getLogger(__name__)

# --- Global state ---
orchestrator: Optional[FinancialAgentOrchestrator] = None
ingestion_pipeline: Optional[IngestionPipeline] = None
report_generator: Optional[ReportGenerator] = None

# In-memory record of background ingestion jobs (most recent 100)
MAX_JOBS = 100
jobs: "OrderedDict[str, dict]" = OrderedDict()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global orchestrator, ingestion_pipeline, report_generator

    logger.info("Starting AI Financial Analyst Agent...")
    install_log_redaction()
    if not settings.api_key:
        logger.warning(
            "API_KEY is not set: the API is unauthenticated. Set API_KEY before exposing it."
        )

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
    # Browsers reject credentials with a wildcard origin, so only enable them for explicit origins
    allow_credentials="*" not in settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


router = APIRouter(dependencies=[Depends(authenticate)])

TICKER_PATTERN = r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$"
SESSION_PATTERN = r"^[A-Za-z0-9_\-]{1,64}$"
ALLOWED_FORMS = {"10-K", "10-Q", "8-K", "10-K/A", "10-Q/A", "DEF 14A"}
TickerPath = Path(..., pattern=TICKER_PATTERN, description="Stock ticker symbol")
_ingest_slots: Optional[asyncio.Semaphore] = None


def _session_key(request: Request, session_id: str) -> str:
    """Namespace sessions by client so one client cannot read another's history."""
    return f"{getattr(request.state, 'client_id', 'anon')}:{session_id}"


# --- Request/Response Models ---


class QueryRequest(BaseModel):
    """User query request."""

    query: str = Field(..., min_length=1, max_length=4000, description="Natural language financial question")
    session_id: Optional[str] = Field(
        None,
        pattern=SESSION_PATTERN,
        description="Conversation id. Omit to start a new conversation; reuse the returned id to continue it",
    )
    ticker: Optional[str] = Field(None, pattern=r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$", description="Stock ticker to focus on")


class QueryResponse(BaseModel):
    """Agent response."""

    answer: str
    tools_used: list[str]
    sources: list[dict]
    session_id: Optional[str] = None


class IngestRequest(BaseModel):
    """Data ingestion request."""

    ticker: str = Field(..., pattern=r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$", description="Stock ticker symbol")
    form_types: list[str] = Field(default=["10-K", "10-Q"], description="SEC form types")
    filing_count: int = Field(default=3, ge=1, le=10, description="Number of filings per type")
    include_news: bool = Field(default=True, description="Also ingest financial news")
    news_days_back: int = Field(default=30, ge=0, le=365, description="Days of news to fetch")
    @field_validator("form_types")
    @classmethod
    def _check_forms(cls, forms: list[str]) -> list[str]:
        bad = [f for f in forms if f not in ALLOWED_FORMS]
        if bad or not forms:
            raise ValueError(f"form_types must be a non-empty subset of {sorted(ALLOWED_FORMS)}")
        return forms

    run_in_background: bool = Field(
        default=True,
        description="Return a job id immediately and poll /api/v1/jobs/{id}; false waits for completion",
    )


class AnalysisRequest(BaseModel):
    """Analysis request."""

    ticker: str = Field(..., pattern=r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$", description="Stock ticker symbol")
    analysis_type: Literal["comprehensive", "earnings", "risk", "valuation"] = Field(
        default="comprehensive",
        description="Type: comprehensive, earnings, risk, valuation",
    )
    generate_report: bool = Field(default=True, description="Generate a full report file")


class EarningsCallRequest(BaseModel):
    """Earnings call ingestion request."""

    raw_text: str = Field(..., min_length=1, max_length=2_000_000, description="Raw earnings call transcript text")
    company: str = Field(..., description="Company name")
    ticker: str = Field(..., pattern=r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$", description="Stock ticker")
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


@router.post("/api/v1/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, http_request: Request):
    """
    Main query endpoint — ask any financial question.
    The agent will automatically route to the right tools.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    session_id = request.session_id or uuid.uuid4().hex
    response = await orchestrator.process_query(
        request.query,
        ticker=request.ticker,
        session_id=_session_key(http_request, session_id),
    )
    return QueryResponse(
        answer=response.answer,
        tools_used=response.tools_used,
        sources=response.sources,
        session_id=session_id,
    )


@router.post("/api/v1/query/stream")
async def stream_query(request: QueryRequest):
    """
    Stream an answer as server-sent events. Events are JSON objects:
    {"type": "sources"|"token"|"done"|"error", ...}. Sources arrive first so
    clients can resolve the [S#] citations as text streams in.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    async def event_stream():
        async for event in orchestrator.analyzer.stream_answer(request.query, request.ticker):
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _new_job(kind: str, ticker: str) -> str:
    job_id = uuid.uuid4().hex
    jobs[job_id] = {
        "job_id": job_id,
        "kind": kind,
        "ticker": ticker,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result": None,
        "error": None,
    }
    while len(jobs) > MAX_JOBS:
        jobs.popitem(last=False)
    return job_id


async def _run_ingest(request: IngestRequest) -> dict:
    return await ingestion_pipeline.ingest_full_company(
        ticker=request.ticker.upper(),
        sec_form_types=request.form_types,
        sec_count=request.filing_count,
        news_days_back=request.news_days_back if request.include_news else 0,
    )


async def _ingest_job(job_id: str, request: IngestRequest) -> None:
    job = jobs.get(job_id)
    if job is None:
        return
    global _ingest_slots
    if _ingest_slots is None:
        _ingest_slots = asyncio.Semaphore(max(1, settings.max_ingest_jobs))
    async with _ingest_slots:  # limit concurrent heavy downloads
        job["status"] = "running"
        try:
            job["result"] = await _run_ingest(request)
            job["status"] = "completed"
        except Exception as e:
            logger.error(f"Ingest job {job_id} failed: {redact_secrets(e)}")
            job["error"] = redact_secrets(e)
            job["status"] = "failed"


@router.post("/api/v1/ingest")
async def ingest_company(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest SEC filings and news for a company into the knowledge base.
    By default runs in the background and returns a job id (HTTP 202);
    poll GET /api/v1/jobs/{job_id}. Set run_in_background=false to wait.
    """
    if not ingestion_pipeline:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not initialized")

    if not request.run_in_background:
        result = await _run_ingest(request)
        return {"status": "completed", "ticker": request.ticker, "results": result}

    job_id = _new_job("ingest", request.ticker.upper())
    background_tasks.add_task(_ingest_job, job_id, request)
    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "job_id": job_id,
            "status_url": f"/api/v1/jobs/{job_id}",
        },
    )


@router.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str = Path(..., pattern=r"^[0-9a-f]{32}$")):
    """Get the status and result of a background job."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/api/v1/ingest/earnings-call")
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


@router.post("/api/v1/analyze")
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
        "trends": analysis.trends,
        "verification": analysis.verification,
    }

    if request.generate_report and report_generator:
        report_generator.generate_markdown_report(analysis)
        report_generator.generate_json_report(analysis)
        response["report_generated"] = True
        response["report_files"] = {
            "markdown": str(report_generator.last_markdown_path),
            "json": str(report_generator.last_json_path),
        }

    return response


@router.get("/api/v1/financials/{ticker}")
async def get_financials(
    ticker: str = TickerPath,
    statement: Literal["all", "income_statement", "balance_sheet", "cash_flow"] = "all",
    period: Literal["annual", "quarterly"] = "annual",
    limit: int = Query(5, ge=1, le=20),
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
        raise HTTPException(status_code=500, detail=f"Failed to fetch financials: {redact_secrets(e)}")

    return {"ticker": ticker, "period": period, "statements": result}


@router.get("/api/v1/profile/{ticker}")
async def get_profile(ticker: str = TickerPath):
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
        raise HTTPException(status_code=502, detail=redact_secrets(e))


@router.get("/api/v1/ratios/{ticker}")
async def get_ratios(ticker: str = TickerPath):
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


@router.get("/api/v1/trends/{ticker}")
async def get_trends(
    ticker: str = TickerPath,
    years: int = Query(5, ge=2, le=10),
    include_ttm: bool = False,
):
    """Multi-year growth, CAGR and margin trends from annual statements."""
    data_client = FinancialDataClient()
    try:
        income, balance, cash_flow = await asyncio.gather(
            data_client.get_income_statement(ticker, "annual", years),
            data_client.get_balance_sheet(ticker, "annual", years),
            data_client.get_cash_flow(ticker, "annual", years),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch financials: {redact_secrets(e)}")
    trends = compute_trends(
        {"income_statement": income, "balance_sheet": balance, "cash_flow": cash_flow}
    )
    response = {"ticker": ticker.upper(), "trends": trends}
    if include_ttm:
        try:
            q_income, q_cash = await asyncio.gather(
                data_client.get_income_statement(ticker, "quarterly", 8),
                data_client.get_cash_flow(ticker, "quarterly", 8),
            )
            response["ttm"] = compute_ttm({"income_statement": q_income, "cash_flow": q_cash})
        except Exception as e:
            response["ttm_error"] = redact_secrets(e)
    return response


@router.get("/api/v1/knowledge-base/stats")
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


@router.post("/api/v1/conversation/reset")
async def reset_conversation(
    http_request: Request,
    session_id: str = Query(..., pattern=SESSION_PATTERN),
):
    """Reset one conversation's history."""
    if orchestrator:
        orchestrator.reset_conversation(_session_key(http_request, session_id))
    return {"status": "conversation reset", "session_id": session_id}


@router.get("/api/v1/config")
async def get_config():
    """Get current (non-sensitive) configuration."""
    return {
        "llm_provider": settings.llm_provider.value,
        "llm_model": settings.active_llm_model,
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


app.include_router(router)

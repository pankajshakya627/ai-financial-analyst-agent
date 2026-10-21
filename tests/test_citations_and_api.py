"""Tests for numbered citations, orchestrator state, and API behaviour."""

import asyncio

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("fastapi")

from core.vector_store import RetrievedDocument
from rag.retriever import FinancialRAGRetriever


def _doc(source, date, **extra):
    meta = {"source": source, "ticker": "AAPL", "company": "Apple", "filing_date": date,
            "form_type": "10-K", **extra}
    return RetrievedDocument(content="text", metadata=meta, score=0.9, doc_id=f"{source}{date}")


def _retriever():
    return FinancialRAGRetriever.__new__(FinancialRAGRetriever)


def test_sources_are_numbered_and_deduplicated():
    docs = [_doc("sec_edgar", "2024-11-01"), _doc("sec_edgar", "2024-11-01"), _doc("sec_edgar", "2023-11-01")]
    sources, index = _retriever()._build_sources(docs, id_start=3)
    assert [s["id"] for s in sources] == [3, 4]
    context = _retriever()._assemble_context(docs, index)
    assert "[S3]" in context and "[S4]" in context


def test_orchestrator_keeps_citation_numbers_unique_and_uses_ticker():
    from agent.orchestrator import FinancialAgentOrchestrator

    calls = []

    class FakeRag:
        def retrieve(self, query, ticker, collections, source_id_start):
            calls.append((ticker, source_id_start))
            sources, _ = _retriever()._build_sources(
                [_doc("sec_edgar", f"2024-0{source_id_start}-01")], id_start=source_id_start)
            from rag.retriever import RetrievalResult
            return RetrievalResult(query, [], "ctx", sources)

    orch = FinancialAgentOrchestrator.__new__(FinancialAgentOrchestrator)
    orch.rag = FakeRag()
    state = {"next_source_id": 1}

    async def run():
        await orch._execute_tool("search_sec_filings", {"query": "risk"}, "AAPL", state)
        await orch._execute_tool("search_sec_filings", {"query": "debt", "ticker": "msft"}, "AAPL", state)

    asyncio.run(run())
    assert calls == [("AAPL", 1), ("MSFT", 2)]
    assert state["next_source_id"] == 3


def test_analysis_tool_shares_citation_counter():
    from agent.orchestrator import FinancialAgentOrchestrator
    from analysis.financial_analyzer import AnalysisResult

    starts = []

    class FakeAnalyzer:
        async def comprehensive_analysis(self, ticker, source_id_start=1):
            starts.append(source_id_start)
            sources, _ = _retriever()._build_sources(
                [_doc("sec_edgar", f"2024-0{source_id_start}-0{n}") for n in (1, 2)],
                id_start=source_id_start,
            )
            return AnalysisResult(
                ticker=ticker, analysis_type="comprehensive", summary="", detailed_analysis="",
                key_metrics={}, ratios=[], risks=[], catalysts=[], sources=sources,
            )

    class FakeRag:
        def retrieve(self, query, ticker, collections, source_id_start):
            sources, _ = _retriever()._build_sources(
                [_doc("sec_edgar", "2023-01-01")], id_start=source_id_start)
            from rag.retriever import RetrievalResult
            return RetrievalResult(query, [], "ctx", sources)

    orch = FinancialAgentOrchestrator.__new__(FinancialAgentOrchestrator)
    orch.analyzer = FakeAnalyzer()
    orch.rag = FakeRag()
    state = {"next_source_id": 1}

    async def run():
        search = await orch._execute_tool("search_sec_filings", {"query": "risk"}, "AAPL", state)
        analysis = await orch._execute_tool(
            "run_comprehensive_analysis", {"ticker": "AAPL"}, "AAPL", state)
        return search, analysis

    search, analysis = asyncio.run(run())
    assert starts == [2]  # analysis picks up where the search left off
    assert [s["id"] for s in search["sources"]] == [1]
    assert [s["id"] for s in analysis["sources"]] == [2, 3]
    assert state["next_source_id"] == 4


def test_orchestrator_returns_text_only_when_no_tool_calls():
    from agent.orchestrator import FinancialAgentOrchestrator

    class FakeLLM:
        def __init__(self):
            self.n = 0

        async def generate_with_tools(self, prompt, tools, system_prompt=None, temperature=None):
            self.n += 1
            if self.n == 1:
                return {"text": "Let me check", "tool_calls": [{"name": "get_company_profile", "arguments": {"ticker": "A"}, "id": "1"}]}
            if self.n == 2:
                return {"text": "Checking more", "tool_calls": [{"name": "get_company_profile", "arguments": {"ticker": "A"}, "id": "2"}]}
            return {"text": "FINAL", "tool_calls": []}

    orch = FinancialAgentOrchestrator.__new__(FinancialAgentOrchestrator)
    orch.llm = FakeLLM()
    from collections import OrderedDict
    orch.conversations = OrderedDict()

    async def fake_exec(name, args, ticker=None, state=None):
        return {"ok": True, "sources": []}

    orch._execute_tool = fake_exec
    response = asyncio.run(orch.process_query("q"))
    assert response.answer == "FINAL"
    assert len(response.raw_tool_results) == 2


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient
    import api.server as server

    class FakeAnalyzer:
        async def stream_answer(self, question, ticker=None):
            yield {"type": "sources", "sources": []}
            yield {"type": "token", "text": f"hi {ticker}"}
            yield {"type": "done"}

    class FakeOrch:
        analyzer = FakeAnalyzer()
        seen = {}

        async def process_query(self, query, ticker=None, session_id="default"):
            FakeOrch.seen["ticker"] = ticker
            FakeOrch.seen["session"] = session_id
            from agent.orchestrator import AgentResponse
            return AgentResponse(answer="ok", tools_used=[], sources=[])

    class FakePipeline:
        async def ingest_full_company(self, **kwargs):
            return {"ticker": kwargs["ticker"], "news_days_back": kwargs["news_days_back"]}

    monkeypatch.setattr(server, "orchestrator", FakeOrch())
    monkeypatch.setattr(server, "ingestion_pipeline", FakePipeline())
    server.jobs.clear()
    from api.security import reset_rate_limits
    reset_rate_limits()
    monkeypatch.setattr(server.settings, "api_key", None)
    monkeypatch.setattr(server.settings, "rate_limit_per_minute", 60)
    # skip lifespan (it builds real clients)
    return TestClient(server.app), FakeOrch


def test_query_passes_ticker(client):
    c, orch = client
    assert c.post("/api/v1/query", json={"query": "x", "ticker": "AAPL"}).status_code == 200
    assert orch.seen["ticker"] == "AAPL"


def test_invalid_ticker_rejected(client):
    c, _ = client
    assert c.post("/api/v1/query", json={"query": "x", "ticker": "../etc"}).status_code == 422


def test_stream_endpoint_emits_sse_events(client):
    c, _ = client
    r = c.post("/api/v1/query/stream", json={"query": "x", "ticker": "AAPL"})
    assert r.headers["content-type"].startswith("text/event-stream")
    assert 'data: {"type": "token", "text": "hi AAPL"}' in r.text
    assert '"type": "done"' in r.text


def test_ingest_runs_in_background_and_job_is_pollable(client):
    c, _ = client
    r = c.post("/api/v1/ingest", json={"ticker": "aapl", "include_news": False})
    assert r.status_code == 202
    job = c.get(r.json()["status_url"]).json()
    # TestClient runs background tasks before returning
    assert job["status"] == "completed"
    assert job["result"]["news_days_back"] == 0
    assert c.get("/api/v1/jobs/" + "0" * 32).status_code == 404
    assert c.get("/api/v1/jobs/nope").status_code == 422


def test_ingest_can_wait_for_result(client):
    c, _ = client
    r = c.post("/api/v1/ingest", json={"ticker": "AAPL", "run_in_background": False})
    assert r.status_code == 200 and r.json()["status"] == "completed"


def test_config_reports_active_model(client, monkeypatch):
    import api.server as server
    from config.settings import LLMProvider
    c, _ = client
    monkeypatch.setattr(server.settings, "llm_provider", LLMProvider.OLLAMA)
    assert c.get("/api/v1/config").json()["llm_model"] == server.settings.ollama_model

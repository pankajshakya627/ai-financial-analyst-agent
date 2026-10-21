"""Security behaviour: auth, rate limits, sessions, validation, redaction."""

import asyncio
from collections import OrderedDict

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("fastapi")

from core.security import redact_secrets


def test_redacts_query_string_keys():
    text = "Client error for url 'https://x.io/q?symbol=A&apikey=SECRET123&limit=5'"
    out = redact_secrets(text)
    assert "SECRET123" not in out and "limit=5" in out


def test_redacts_configured_key_values(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "fmp_api_key", "abcdef123456")
    assert "abcdef123456" not in redact_secrets("failed using abcdef123456 today")


@pytest.fixture
def api(monkeypatch):
    from fastapi.testclient import TestClient
    import api.server as server
    from api.security import reset_rate_limits

    class FakeOrch:
        seen = []

        async def process_query(self, query, ticker=None, session_id="default"):
            FakeOrch.seen.append(session_id)
            from agent.orchestrator import AgentResponse
            return AgentResponse(answer="ok", tools_used=[], sources=[])

        def reset_conversation(self, session_id="default"):
            FakeOrch.seen.append(("reset", session_id))

    reset_rate_limits()
    FakeOrch.seen = []
    monkeypatch.setattr(server, "orchestrator", FakeOrch())
    monkeypatch.setattr(server.settings, "api_key", None)
    monkeypatch.setattr(server.settings, "rate_limit_per_minute", 60)
    return TestClient(server.app), server, FakeOrch


def test_api_key_required_when_configured(api, monkeypatch):
    c, server, _ = api
    monkeypatch.setattr(server.settings, "api_key", "s3cret-key")
    assert c.post("/api/v1/query", json={"query": "x"}).status_code == 401
    assert c.post("/api/v1/query", json={"query": "x"}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.post("/api/v1/query", json={"query": "x"}, headers={"X-API-Key": "s3cret-key"}).status_code == 200
    assert c.get("/health").status_code == 200  # health stays public


def test_rate_limit(api, monkeypatch):
    c, server, _ = api
    monkeypatch.setattr(server.settings, "rate_limit_per_minute", 2)
    codes = [c.post("/api/v1/query", json={"query": "x"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_sessions_are_separate_and_returned(api):
    c, _, orch = api
    a = c.post("/api/v1/query", json={"query": "x"}).json()["session_id"]
    b = c.post("/api/v1/query", json={"query": "x"}).json()["session_id"]
    assert a != b
    c.post("/api/v1/query", json={"query": "x", "session_id": "mine"})
    assert orch.seen[-1].endswith(":mine")


def test_sessions_are_namespaced_by_api_key(api, monkeypatch):
    c, server, orch = api
    monkeypatch.setattr(server.settings, "api_key", "k1")
    c.post("/api/v1/query", json={"query": "x", "session_id": "shared"}, headers={"X-API-Key": "k1"})
    first = orch.seen[-1]
    monkeypatch.setattr(server.settings, "api_key", "k2")
    c.post("/api/v1/query", json={"query": "x", "session_id": "shared"}, headers={"X-API-Key": "k2"})
    assert first != orch.seen[-1]


def test_bad_inputs_rejected(api):
    c, _, _ = api
    assert c.get("/api/v1/profile/..%2Fetc").status_code in (404, 422)
    assert c.get("/api/v1/financials/AAPL?limit=999").status_code == 422
    assert c.post("/api/v1/ingest", json={"ticker": "AAPL", "form_types": ["EVIL"]}).status_code == 422
    assert c.post("/api/v1/ingest", json={"ticker": "AAPL", "filing_count": 500}).status_code == 422
    assert c.post("/api/v1/analyze", json={"ticker": "AAPL", "analysis_type": "nope"}).status_code == 422
    assert c.post("/api/v1/query", json={"query": "x", "session_id": "bad id!"}).status_code == 422


def test_reset_is_per_session(api):
    c, _, orch = api
    r = c.post("/api/v1/conversation/reset?session_id=abc")
    assert r.status_code == 200
    assert orch.seen[-1][0] == "reset" and orch.seen[-1][1].endswith(":abc")
    assert c.post("/api/v1/conversation/reset").status_code == 422


def test_orchestrator_sessions_isolated_and_bounded(monkeypatch):
    from agent.orchestrator import FinancialAgentOrchestrator
    from config.settings import settings

    monkeypatch.setattr(settings, "max_sessions", 2)
    orch = FinancialAgentOrchestrator.__new__(FinancialAgentOrchestrator)
    orch.conversations = OrderedDict()
    orch._history("a").append("x")
    orch._history("b")
    orch._history("c")
    assert "a" not in orch.conversations and len(orch.conversations) == 2
    orch.reset_conversation("b")
    assert "b" not in orch.conversations


def test_tool_args_validated():
    from agent.orchestrator import FinancialAgentOrchestrator

    orch = FinancialAgentOrchestrator.__new__(FinancialAgentOrchestrator)
    with pytest.raises(ValueError):
        orch._validate_args("get_company_profile", {"ticker": "../../x"})
    with pytest.raises(ValueError):
        orch._validate_args("ingest_company_data", {"ticker": "AAPL", "form_types": ["BAD"]})
    args = orch._validate_args("ingest_company_data", {"ticker": "AAPL", "filing_count": 9999})
    assert args["filing_count"] == 10
    assert asyncio.run(asyncio.sleep(0)) is None

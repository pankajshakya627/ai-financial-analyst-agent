"""
Agent orchestrator — the brain of the Financial Analyst Agent.
Routes user queries to the appropriate tools, executes them,
and synthesizes results into coherent responses.
"""

import json
import logging
from typing import Optional
from dataclasses import dataclass, field

from config.settings import settings
from core.llm_provider import LLMClient, get_llm_client
from core.vector_store import VectorStoreManager
from rag.retriever import FinancialRAGRetriever
from analysis.financial_data import FinancialDataClient
from analysis.financial_analyzer import FinancialAnalyzer
from analysis.ratios import FinancialRatioCalculator
from ingestion.pipeline import IngestionPipeline
from agent.tools import AGENT_TOOLS

logger = logging.getLogger(__name__)


ORCHESTRATOR_SYSTEM_PROMPT = """You are an AI Financial Analyst Agent with access to:
1. SEC EDGAR filings (10-K, 10-Q, 8-K) stored in a vector database
2. Earnings call transcripts with speaker attribution
3. Financial news with sentiment analysis
4. Real-time financial data APIs (income statements, balance sheets, cash flows)
5. Financial ratio calculation engine
6. Comprehensive analysis generation

Your job is to answer financial questions accurately by using the appropriate tools.

IMPORTANT GUIDELINES:
- Always cite your sources (SEC filing date, earnings call quarter, news source)
- Distinguish facts (from filings) from analysis (your interpretation)
- If data is not available in the knowledge base, say so and suggest ingesting it
- Use specific numbers and percentages, not vague language
- Present both bull and bear perspectives when doing analysis
- This is financial analysis for informational purposes only — not investment advice

TOOL USAGE:
- For questions about specific disclosures, risk factors, or management discussion:
  use search_sec_filings
- For management commentary, guidance, and analyst Q&A: use search_earnings_calls
- For recent developments and sentiment: use search_financial_news
- For structured financial data and ratios: use get_financial_statements or
  calculate_financial_ratios
- For comprehensive analysis: use run_comprehensive_analysis
- If a company hasn't been ingested yet: use ingest_company_data first"""


@dataclass
class ConversationMessage:
    """A message in the conversation history."""

    role: str  # user, assistant, tool_result
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentResponse:
    """The agent's response to a user query."""

    answer: str
    tools_used: list[str]
    sources: list[dict]
    raw_tool_results: dict = field(default_factory=dict)


class FinancialAgentOrchestrator:
    """
    Main agent orchestrator. Handles:
    1. User query interpretation
    2. Tool selection via LLM function calling
    3. Tool execution
    4. Response synthesis with citations
    """

    MAX_TOOL_ITERATIONS = 5  # prevent infinite tool loops

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        vector_store: Optional[VectorStoreManager] = None,
    ):
        self.llm = llm_client or get_llm_client()
        self.vector_store = vector_store or VectorStoreManager()
        self.rag = FinancialRAGRetriever(self.vector_store)
        self.data_client = FinancialDataClient()
        self.analyzer = FinancialAnalyzer(self.llm, self.data_client, self.rag)
        self.ingestion = IngestionPipeline(self.vector_store)
        self.ratio_calc = FinancialRatioCalculator()
        self.conversation_history: list[ConversationMessage] = []

    async def process_query(self, user_query: str) -> AgentResponse:
        """
        Process a user query end-to-end:
        1. Add to conversation history
        2. Call LLM with tools to determine action plan
        3. Execute tools
        4. Synthesize final response
        """
        self.conversation_history.append(
            ConversationMessage(role="user", content=user_query)
        )

        tools_used = []
        all_sources = []
        tool_results = {}

        # Iterative tool calling loop
        for iteration in range(self.MAX_TOOL_ITERATIONS):
            # Build the conversation context
            context = self._build_context(tool_results)

            # Call LLM with tool definitions
            llm_response = await self.llm.generate_with_tools(
                prompt=f"{context}\n\nUser Question: {user_query}",
                tools=AGENT_TOOLS,
                system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            )

            # If LLM returned text without tool calls, we're done
            if not llm_response.get("tool_calls") and llm_response.get("text"):
                final_answer = llm_response["text"]
                break

            # Execute each tool call
            for tool_call in llm_response.get("tool_calls", []):
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                tools_used.append(tool_name)

                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                try:
                    result = await self._execute_tool(tool_name, tool_args)
                    tool_results[f"{tool_name}_{iteration}"] = result

                    # Collect sources
                    if isinstance(result, dict) and "sources" in result:
                        all_sources.extend(result["sources"])

                except Exception as e:
                    logger.error(f"Tool execution error: {tool_name}: {e}")
                    tool_results[f"{tool_name}_{iteration}"] = {"error": str(e)}

            # If we have text alongside tool calls, check if it's a final answer
            if llm_response.get("text") and iteration > 0:
                final_answer = llm_response["text"]
                break
        else:
            # Exceeded max iterations — synthesize from what we have
            final_answer = await self._synthesize_response(user_query, tool_results)

        # Add to conversation history
        self.conversation_history.append(
            ConversationMessage(
                role="assistant",
                content=final_answer,
                metadata={"tools_used": tools_used},
            )
        )

        return AgentResponse(
            answer=final_answer,
            tools_used=list(set(tools_used)),
            sources=self._deduplicate_sources(all_sources),
            raw_tool_results=tool_results,
        )

    async def _execute_tool(self, tool_name: str, args: dict) -> dict:
        """Execute a tool by name and return structured results."""

        if tool_name == "search_sec_filings":
            result = self.rag.retrieve(
                query=args["query"],
                ticker=args.get("ticker"),
                collections=[settings.chroma_collection_sec],
            )
            return {
                "documents": [
                    {"content": d.content[:500], "metadata": d.metadata, "score": d.score}
                    for d in result.documents
                ],
                "sources": result.sources,
                "context": result.context_text[:3000],
            }

        elif tool_name == "search_earnings_calls":
            result = self.rag.retrieve(
                query=args["query"],
                ticker=args.get("ticker"),
                collections=[settings.chroma_collection_earnings],
            )
            return {
                "documents": [
                    {"content": d.content[:500], "metadata": d.metadata, "score": d.score}
                    for d in result.documents
                ],
                "sources": result.sources,
                "context": result.context_text[:3000],
            }

        elif tool_name == "search_financial_news":
            result = self.rag.retrieve(
                query=args["query"],
                ticker=args.get("ticker"),
                collections=[settings.chroma_collection_news],
            )
            return {
                "documents": [
                    {"content": d.content[:500], "metadata": d.metadata, "score": d.score}
                    for d in result.documents
                ],
                "sources": result.sources,
                "context": result.context_text[:2000],
            }

        elif tool_name == "get_financial_statements":
            ticker = args["ticker"]
            stmt_type = args.get("statement_type", "all")
            period = args.get("period", "annual")
            limit = args.get("limit", 5)

            results = {}
            if stmt_type in ("income_statement", "all"):
                stmts = await self.data_client.get_income_statement(ticker, period, limit)
                results["income_statement"] = [
                    {"date": s.date, "data": {k: v for k, v in s.data.items() if v and k != "link"}}
                    for s in stmts
                ]

            if stmt_type in ("balance_sheet", "all"):
                stmts = await self.data_client.get_balance_sheet(ticker, period, limit)
                results["balance_sheet"] = [
                    {"date": s.date, "data": {k: v for k, v in s.data.items() if v and k != "link"}}
                    for s in stmts
                ]

            if stmt_type in ("cash_flow", "all"):
                stmts = await self.data_client.get_cash_flow(ticker, period, limit)
                results["cash_flow"] = [
                    {"date": s.date, "data": {k: v for k, v in s.data.items() if v and k != "link"}}
                    for s in stmts
                ]

            return {"ticker": ticker, "period": period, "statements": results, "sources": []}

        elif tool_name == "get_company_profile":
            profile = await self.data_client.get_company_profile(args["ticker"])
            return {
                "ticker": profile.ticker,
                "name": profile.name,
                "sector": profile.sector,
                "industry": profile.industry,
                "market_cap": profile.market_cap,
                "description": profile.description[:500],
                "exchange": profile.exchange,
                "ceo": profile.ceo,
                "employees": profile.employees,
                "sources": [],
            }

        elif tool_name == "calculate_financial_ratios":
            ticker = args["ticker"]
            financials = await self.data_client.get_full_financials(ticker)
            ratios = self.analyzer._compute_ratios(financials)
            return {
                "ticker": ticker,
                "ratios": [
                    {
                        "name": r.name,
                        "value": round(r.value, 4) if r.value else None,
                        "category": r.category,
                        "interpretation": r.interpretation,
                        "benchmark": r.benchmark,
                    }
                    for r in ratios
                ],
                "sources": [],
            }

        elif tool_name == "get_stock_price":
            prices = await self.data_client.get_stock_price(
                args["ticker"], args.get("period", "1Y")
            )
            # Return summary stats instead of all data points
            if prices:
                closes = [p.close for p in prices]
                return {
                    "ticker": args["ticker"],
                    "period": args.get("period", "1Y"),
                    "current_price": closes[0],
                    "period_high": max(closes),
                    "period_low": min(closes),
                    "period_return": (closes[0] - closes[-1]) / closes[-1] if closes[-1] else 0,
                    "data_points": len(prices),
                    "sources": [],
                }
            return {"error": "No price data available", "sources": []}

        elif tool_name == "run_comprehensive_analysis":
            ticker = args["ticker"]
            analysis_type = args.get("analysis_type", "comprehensive")

            if analysis_type == "comprehensive":
                result = await self.analyzer.comprehensive_analysis(ticker)
            elif analysis_type == "earnings":
                result = await self.analyzer.earnings_analysis(ticker)
            elif analysis_type == "risk":
                result = await self.analyzer.risk_analysis(ticker)
            elif analysis_type == "valuation":
                result = await self.analyzer.valuation_analysis(ticker)
            else:
                result = await self.analyzer.comprehensive_analysis(ticker)

            return {
                "analysis": result.detailed_analysis,
                "summary": result.summary,
                "key_metrics": result.key_metrics,
                "ratios": result.ratios[:10],  # top 10
                "risks": result.risks[:5],
                "catalysts": result.catalysts[:5],
                "sources": result.sources,
            }

        elif tool_name == "ingest_company_data":
            ticker = args["ticker"]
            form_types = args.get("form_types", ["10-K", "10-Q"])
            count = args.get("filing_count", 3)

            result = await self.ingestion.ingest_full_company(
                ticker=ticker,
                sec_form_types=form_types,
                sec_count=count,
            )
            return {
                "ticker": ticker,
                "sec_filings": result.get("sec_filings", {}),
                "news": result.get("news", {}),
                "message": f"Successfully ingested data for {ticker}",
                "sources": [],
            }

        elif tool_name == "get_analyst_estimates":
            estimates = await self.data_client.get_analyst_estimates(args["ticker"])
            return {
                "ticker": args["ticker"],
                "estimates": [
                    {
                        "period": e.period,
                        "revenue_estimate": e.revenue_estimate,
                        "eps_estimate": e.eps_estimate,
                        "num_analysts": e.num_analysts,
                        "recommendation": e.recommendation,
                    }
                    for e in estimates
                ],
                "sources": [],
            }

        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _build_context(self, tool_results: dict) -> str:
        """Build conversation context including previous tool results."""
        parts = []

        # Include recent conversation history
        for msg in self.conversation_history[-6:]:  # last 3 turns
            parts.append(f"[{msg.role}]: {msg.content[:500]}")

        # Include tool results from this iteration
        if tool_results:
            parts.append("\n## Previous Tool Results")
            for tool_key, result in tool_results.items():
                # Truncate large results
                result_str = json.dumps(result, default=str)[:2000]
                parts.append(f"\n### {tool_key}\n{result_str}")

        return "\n".join(parts)

    async def _synthesize_response(self, query: str, tool_results: dict) -> str:
        """Synthesize a final response from accumulated tool results."""
        results_text = json.dumps(tool_results, default=str)[:6000]

        prompt = f"""Based on the following tool results, provide a comprehensive answer
to the user's question.

## User Question
{query}

## Tool Results
{results_text}

Provide a well-structured, data-driven response with specific numbers and citations.
If any data was unavailable, acknowledge the gap.

Important: This response is for informational purposes only and does not constitute
financial or investment advice."""

        return await self.llm.generate(
            prompt=prompt,
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            temperature=0.2,
        )

    @staticmethod
    def _deduplicate_sources(sources: list[dict]) -> list[dict]:
        """Remove duplicate sources."""
        seen = set()
        unique = []
        for source in sources:
            key = f"{source.get('type')}_{source.get('ticker')}_{source.get('date')}"
            if key not in seen:
                seen.add(key)
                unique.append(source)
        return unique

    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history.clear()

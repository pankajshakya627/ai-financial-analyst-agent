"""
Enhanced Agent Orchestrator with Intent-Based Routing.
Detects user intent and routes to specialized agents for optimal task execution.
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
from agent.intent_detector import IntentDetector, get_intent_detector, IntentCategory, DetectedIntent
from agent.agents import ResearchAgent, AnalysisAgent, DataAgent, AgentResult

logger = logging.getLogger(__name__)


ORCHESTRATOR_SYSTEM_PROMPT = """You are an AI Financial Analyst Agent with intent-based routing capabilities.

Your workflow:
1. DETECT the user's intent (what they're really asking for)
2. ROUTE to the appropriate specialized agent:
   - Research Agent: SEC filings, earnings calls, news search
   - Analysis Agent: Comprehensive analysis, risk assessment, valuation, ratios
   - Data Agent: Financial statements, company profiles, stock prices, data ingestion
3. EXECUTE the task using the specialized agent
4. SYNTHESIZE a clear, actionable response with citations

IMPORTANT GUIDELINES:
- Always cite your sources (SEC filing date, earnings call quarter, news source)
- Distinguish facts (from filings) from analysis (your interpretation)
- If data is not available, say so and offer to ingest it
- Use specific numbers and percentages, not vague language
- Present both bull and bear perspectives when doing analysis
- This is for informational purposes only — not investment advice

INTENT CATEGORIES:
- sec_filing_query: Questions about 10-K, 10-Q, 8-K filings
- earnings_query: Earnings call transcripts and management commentary
- news_query: Recent financial news and developments
- financial_data_query: Financial statements and numerical data
- comprehensive_analysis: Full investment research report
- risk_analysis: Risk assessment and red flags
- valuation_analysis: Valuation metrics and fair value
- ratio_analysis: Financial ratio calculations
- company_profile: Company overview and business model
- stock_price: Stock price and performance data
- analyst_estimates: Analyst recommendations and estimates
- ingest_data: Download and process new company data"""


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
    intent: DetectedIntent
    agent_used: str  # research, analysis, data
    sources: list[dict]
    raw_result: Optional[AgentResult] = None


class FinancialAgentOrchestrator:
    """
    Intent-based agent orchestrator. Handles:
    1. Intent detection (what does the user want?)
    2. Agent routing (which specialist should handle this?)
    3. Task execution (run the appropriate agent)
    4. Response synthesis (combine results into clear answer)
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        vector_store: Optional[VectorStoreManager] = None,
    ):
        self.llm = llm_client or get_llm_client()
        self.vector_store = vector_store or VectorStoreManager()
        
        # Initialize intent detector
        self.intent_detector = get_intent_detector()
        
        # Initialize shared components
        self.rag = FinancialRAGRetriever(self.vector_store)
        self.data_client = FinancialDataClient()
        self.analyzer = FinancialAnalyzer(self.llm, self.data_client, self.rag)
        self.ingestion = IngestionPipeline(self.vector_store)
        self.ratio_calc = FinancialRatioCalculator()
        
        # Initialize specialized agents
        self.research_agent = ResearchAgent(self.llm, self.vector_store, self.rag)
        self.analysis_agent = AnalysisAgent(
            self.llm, self.data_client, self.analyzer, self.ratio_calc
        )
        self.data_agent = DataAgent(self.llm, self.data_client, self.ingestion)
        
        self.conversation_history: list[ConversationMessage] = []

    async def process_query(self, user_query: str) -> AgentResponse:
        """
        Process a user query using intent-based routing:
        1. Detect intent
        2. Route to appropriate agent
        3. Execute task
        4. Synthesize response
        """
        logger.info(f"Processing query: {user_query[:100]}...")
        
        # Add to conversation history
        self.conversation_history.append(
            ConversationMessage(role="user", content=user_query)
        )
        
        # Step 1: Detect intent
        intent = await self.intent_detector.detect_intent(user_query)
        logger.info(f"Detected intent: {intent.category} (confidence: {intent.confidence:.2f})")
        
        # Step 2: Route to appropriate agent
        agent_result, agent_name = await self._route_to_agent(intent)
        
        # Step 3: Synthesize response
        answer = await self._synthesize_response(user_query, intent, agent_result)
        
        # Add to conversation history
        self.conversation_history.append(
            ConversationMessage(
                role="assistant",
                content=answer,
                metadata={
                    "intent": intent.category.value,
                    "agent": agent_name,
                    "confidence": intent.confidence,
                },
            )
        )
        
        return AgentResponse(
            answer=answer,
            intent=intent,
            agent_used=agent_name,
            sources=agent_result.sources if agent_result else [],
            raw_result=agent_result,
        )

    async def _route_to_agent(self, intent: DetectedIntent) -> tuple[Optional[AgentResult], str]:
        """
        Route intent to appropriate specialized agent.
        
        Routing logic:
        - Research queries → Research Agent
        - Analysis queries → Analysis Agent
        - Data queries → Data Agent
        """
        # Research Agent: Information retrieval
        if intent.category in [
            IntentCategory.SEC_FILING_QUERY,
            IntentCategory.EARNINGS_QUERY,
            IntentCategory.NEWS_QUERY,
        ]:
            logger.info("Routing to Research Agent")
            result = await self.research_agent.handle(intent)
            return result, "research"
        
        # Analysis Agent: Analytical tasks
        elif intent.category in [
            IntentCategory.COMPREHENSIVE_ANALYSIS,
            IntentCategory.RISK_ANALYSIS,
            IntentCategory.VALUATION_ANALYSIS,
            IntentCategory.RATIO_ANALYSIS,
            IntentCategory.EARNINGS_ANALYSIS,
        ]:
            logger.info("Routing to Analysis Agent")
            result = await self.analysis_agent.handle(intent)
            return result, "analysis"
        
        # Data Agent: Data retrieval and management
        elif intent.category in [
            IntentCategory.FINANCIAL_DATA_QUERY,
            IntentCategory.COMPANY_PROFILE,
            IntentCategory.STOCK_PRICE,
            IntentCategory.ANALYST_ESTIMATES,
            IntentCategory.INGEST_DATA,
            IntentCategory.UPDATE_DATA,
        ]:
            logger.info("Routing to Data Agent")
            result = await self.data_agent.handle(intent)
            return result, "data"
        
        # Unknown intent - try general LLM response
        else:
            logger.warning(f"Unknown intent category: {intent.category}, using LLM fallback")
            return None, "llm_fallback"

    async def _synthesize_response(
        self,
        query: str,
        intent: DetectedIntent,
        agent_result: Optional[AgentResult],
    ) -> str:
        """Synthesize a clear response from agent results."""
        
        # Handle LLM fallback
        if agent_result is None:
            return await self._llm_fallback(query)
        
        # Handle agent errors
        if not agent_result.success:
            return f"I encountered an error: {agent_result.error}. Please try rephrasing your question or provide a ticker symbol."
        
        # Build context from agent result
        context_parts = []
        
        if agent_result.content:
            context_parts.append(f"Analysis: {agent_result.content}")
        
        if agent_result.data:
            context_parts.append(f"Data: {json.dumps(agent_result.data, default=str)[:3000]}")
        
        if agent_result.sources:
            sources_text = "\n".join([
                f"- {s.get('type', 'Source')}: {s.get('ticker', 'N/A')} ({s.get('date', 'N/A')})"
                for s in agent_result.sources[:5]
            ])
            context_parts.append(f"Sources:\n{sources_text}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Based on the analysis results, provide a clear, comprehensive answer.

## User Question
{query}

## Detected Intent
{intent.category.value} (confidence: {intent.confidence:.2f})
Ticker: {intent.ticker or 'Not specified'}

## Analysis Results
{context}

Provide a well-structured response with:
1. Direct answer to the question
2. Key findings with specific numbers
3. Sources and citations
4. Any limitations or caveats

Important: This is for informational purposes only and does not constitute financial advice."""

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            temperature=0.2,
        )
        
        return response

    async def _llm_fallback(self, query: str) -> str:
        """Handle queries that don't match any specific intent."""
        prompt = f"""Answer the following financial question to the best of your ability.

Question: {query}

If you need more information (like a ticker symbol), ask for it.
If the question is outside your expertise, say so.

Important: This is for informational purposes only and does not constitute financial advice."""

        return await self.llm.generate(
            prompt=prompt,
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            temperature=0.3,
        )

    def get_intent_stats(self) -> dict:
        """Get statistics about detected intents from conversation history."""
        intent_counts = {}
        agent_counts = {}
        
        for msg in self.conversation_history:
            if msg.role == "assistant" and msg.metadata:
                intent = msg.metadata.get("intent", "unknown")
                agent = msg.metadata.get("agent", "unknown")
                
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        return {
            "total_queries": len([m for m in self.conversation_history if m.role == "user"]),
            "intent_distribution": intent_counts,
            "agent_distribution": agent_counts,
        }

    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")

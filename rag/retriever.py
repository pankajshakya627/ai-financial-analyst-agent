"""
Financial RAG retriever with query routing, re-ranking, and
context assembly for the LLM agent.
"""

import logging
from typing import Optional
from dataclasses import dataclass, field

from config.settings import settings
from core.vector_store import VectorStoreManager, RetrievedDocument

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Structured retrieval result ready for LLM consumption."""

    query: str
    documents: list[RetrievedDocument]
    context_text: str  # assembled context string for the LLM prompt
    sources: list[dict]  # source attribution metadata
    collection_stats: dict = field(default_factory=dict)


class FinancialRAGRetriever:
    """
    Multi-collection retriever that routes queries to the most relevant
    vector store collections and assembles context for the LLM.
    """

    # Query routing keywords
    SEC_KEYWORDS = [
        "10-k", "10-q", "8-k", "annual report", "quarterly report",
        "sec filing", "risk factor", "mda", "management discussion",
        "financial statement", "balance sheet", "income statement",
        "cash flow", "notes to financial", "disclosure", "proxy",
        "audit", "controls", "procedures", "segment", "goodwill",
        "intangible", "impairment", "restructuring", "related party",
    ]

    EARNINGS_KEYWORDS = [
        "earnings call", "transcript", "guidance", "outlook",
        "ceo said", "cfo said", "management commentary",
        "q&a", "analyst question", "forward looking",
        "next quarter", "full year", "backlog", "pipeline",
        "beat expectations", "missed estimates", "raised guidance",
        "lowered guidance", "reaffirmed",
    ]

    NEWS_KEYWORDS = [
        "news", "recent", "latest", "today", "this week",
        "announcement", "press release", "market reaction",
        "upgrade", "downgrade", "analyst rating", "price target",
        "merger", "acquisition", "ipo", "offering", "buyback",
        "dividend", "split", "lawsuit", "investigation",
    ]

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.vector_store = vector_store or VectorStoreManager()

    def route_query(self, query: str) -> list[str]:
        """
        Determine which collections to search based on query content.
        Returns collection names ranked by relevance.
        """
        query_lower = query.lower()
        scores = {
            settings.chroma_collection_sec: 0,
            settings.chroma_collection_earnings: 0,
            settings.chroma_collection_news: 0,
        }

        for kw in self.SEC_KEYWORDS:
            if kw in query_lower:
                scores[settings.chroma_collection_sec] += 2

        for kw in self.EARNINGS_KEYWORDS:
            if kw in query_lower:
                scores[settings.chroma_collection_earnings] += 2

        for kw in self.NEWS_KEYWORDS:
            if kw in query_lower:
                scores[settings.chroma_collection_news] += 2

        # If no strong signal, search all collections
        if max(scores.values()) == 0:
            return list(scores.keys())

        # Return collections sorted by relevance score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [name for name, score in ranked if score > 0]

    def retrieve(
        self,
        query: str,
        ticker: Optional[str] = None,
        top_k: Optional[int] = None,
        collections: Optional[list[str]] = None,
        source_id_start: int = 1,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Natural language query
            ticker: Optional ticker to filter results
            top_k: Number of results to return
            collections: Override automatic collection routing
            source_id_start: First citation number ([S<n>]) to assign; lets callers
                keep numbers unique across several retrievals in one answer
        """
        top_k = top_k or settings.top_k_results
        target_collections = collections or self.route_query(query)

        # Build optional metadata filter
        where_filter = None
        if ticker:
            where_filter = {"ticker": ticker.upper()}

        # Query across collections
        all_docs = []
        collection_stats = {}

        for collection_name in target_collections:
            docs = self.vector_store.query(
                collection_name=collection_name,
                query_text=query,
                top_k=top_k,
                where_filter=where_filter,
            )
            all_docs.extend(docs)
            collection_stats[collection_name] = len(docs)

        # Re-rank across all results by score
        all_docs.sort(key=lambda d: d.score, reverse=True)
        top_docs = all_docs[:top_k]

        # Assemble context text
        # Extract numbered source citations, then label context with the same numbers
        sources, source_index = self._build_sources(top_docs, id_start=source_id_start)
        context_text = self._assemble_context(top_docs, source_index)

        result = RetrievalResult(
            query=query,
            documents=top_docs,
            context_text=context_text,
            sources=sources,
            collection_stats=collection_stats,
        )

        logger.info(
            f"Retrieved {len(top_docs)} documents for query: '{query[:80]}...' "
            f"Collections: {collection_stats}"
        )
        return result

    def retrieve_for_analysis(
        self,
        ticker: str,
        analysis_type: str = "comprehensive",
        source_id_start: int = 1,
    ) -> RetrievalResult:
        """
        Targeted retrieval for specific analysis types.
        Fetches the most relevant context for financial analysis.

        Args:
            ticker: Company ticker to filter results
            analysis_type: Which query template to use
            source_id_start: First citation number ([S<n>]) to assign; lets callers
                keep numbers unique across several retrievals in one answer
        """
        query_templates = {
            "comprehensive": (
                f"Complete financial overview of {ticker} including revenue, "
                f"earnings, growth, margins, risks, and outlook"
            ),
            "risk": (
                f"{ticker} risk factors, regulatory risks, competitive threats, "
                f"and material uncertainties"
            ),
            "earnings": (
                f"{ticker} recent earnings results, revenue growth, EPS, "
                f"guidance, and management commentary"
            ),
            "valuation": (
                f"{ticker} financial metrics for valuation: revenue, EBITDA, "
                f"free cash flow, growth rates, and comparable companies"
            ),
            "competitive": (
                f"{ticker} competitive position, market share, industry trends, "
                f"and competitive advantages or moat"
            ),
        }

        query = query_templates.get(analysis_type, query_templates["comprehensive"])
        return self.retrieve(
            query=query, ticker=ticker, top_k=15, source_id_start=source_id_start
        )

    def _assemble_context(
        self,
        docs: list[RetrievedDocument],
        source_index: Optional[dict[str, int]] = None,
    ) -> str:
        """
        Assemble retrieved documents into a structured context string
        for the LLM prompt.
        """
        if not docs:
            return "No relevant documents found in the knowledge base."

        source_index = source_index or {}
        parts = [
            "## Retrieved Financial Context\n",
            "Cite supporting passages inline using their [S#] labels.\n",
        ]

        def label(doc: RetrievedDocument) -> str:
            n = source_index.get(self._source_key(doc))
            return f"[S{n}] " if n is not None else ""

        # Group by source type
        sec_docs = [d for d in docs if d.metadata.get("source") == "sec_edgar"]
        earnings_docs = [d for d in docs if d.metadata.get("source") == "earnings_call"]
        news_docs = [d for d in docs if d.metadata.get("source") == "financial_news"]

        if sec_docs:
            parts.append("### SEC Filings")
            for doc in sec_docs:
                meta = doc.metadata
                parts.append(
                    f"\n**{label(doc)}{meta.get('form_type', 'Filing')} - "
                    f"{meta.get('company', '')} ({meta.get('filing_date', '')})**\n"
                    f"Section: {meta.get('section', 'N/A')} | "
                    f"Relevance: {doc.score:.2f}\n"
                    f"{doc.content}\n"
                )

        if earnings_docs:
            parts.append("\n### Earnings Call Transcripts")
            for doc in earnings_docs:
                meta = doc.metadata
                parts.append(
                    f"\n**{label(doc)}{meta.get('company', '')} {meta.get('quarter', '')} "
                    f"Earnings Call ({meta.get('call_date', '')})**\n"
                    f"Speaker: {meta.get('speaker', 'N/A')} "
                    f"({meta.get('speaker_role', '')}) | "
                    f"Section: {meta.get('section', '')} | "
                    f"Relevance: {doc.score:.2f}\n"
                    f"{doc.content}\n"
                )

        if news_docs:
            parts.append("\n### Financial News")
            for doc in news_docs:
                meta = doc.metadata
                parts.append(
                    f"\n**{label(doc)}{meta.get('title', 'News Article')}** "
                    f"({meta.get('published_at', '')[:10]})\n"
                    f"Source: {meta.get('news_source', '')} | "
                    f"Sentiment: {meta.get('sentiment', 'N/A')} | "
                    f"Relevance: {doc.score:.2f}\n"
                    f"{doc.content}\n"
                )

        return "\n".join(parts)

    @staticmethod
    def _source_key(doc: RetrievedDocument) -> str:
        meta = doc.metadata
        when = meta.get("filing_date", meta.get("call_date", meta.get("published_at", "")))
        detail = meta.get("form_type", meta.get("quarter", meta.get("title", "")))
        return f"{meta.get('source')}_{meta.get('ticker')}_{when}_{detail}"

    def _build_sources(
        self, docs: list[RetrievedDocument], id_start: int = 1
    ) -> tuple[list[dict], dict[str, int]]:
        """Build numbered, de-duplicated source citations.

        Returns the source list and a map from source key to its citation number.
        """
        index: dict[str, int] = {}
        sources: list[dict] = []
        for doc in docs:
            key = self._source_key(doc)
            if key in index:
                continue
            meta = doc.metadata
            number = id_start + len(sources)
            index[key] = number
            sources.append(
                {
                    "id": number,
                    "type": meta.get("source", "unknown"),
                    "company": meta.get("company", ""),
                    "ticker": meta.get("ticker", ""),
                    "date": meta.get(
                        "filing_date", meta.get("call_date", meta.get("published_at", ""))
                    ),
                    "detail": meta.get("form_type", meta.get("quarter", meta.get("title", ""))),
                    "url": meta.get("url", ""),
                    "relevance": doc.score,
                }
            )
        return sources, index

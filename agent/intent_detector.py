"""
Intent Detection and Agent Routing System.
Classifies user queries into intent categories and routes to specialized agents.
"""

import logging
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field

from config.settings import settings
from core.llm_provider import LLMClient, get_llm_client

logger = logging.getLogger(__name__)


class IntentCategory(str, Enum):
    """Supported intent categories for financial queries."""
    
    # Information Retrieval
    SEC_FILING_QUERY = "sec_filing_query"  # Questions about SEC filings
    EARNINGS_QUERY = "earnings_query"  # Earnings call transcripts
    NEWS_QUERY = "news_query"  # Recent financial news
    FINANCIAL_DATA_QUERY = "financial_data_query"  # Financial statements, ratios
    
    # Analysis Tasks
    COMPREHENSIVE_ANALYSIS = "comprehensive_analysis"  # Full company analysis
    EARNINGS_ANALYSIS = "earnings_analysis"  # Earnings performance analysis
    RISK_ANALYSIS = "risk_analysis"  # Risk assessment
    VALUATION_ANALYSIS = "valuation_analysis"  # Valuation analysis
    RATIO_ANALYSIS = "ratio_analysis"  # Financial ratio calculation
    
    # Data Management
    INGEST_DATA = "ingest_data"  # Ingest new company data
    UPDATE_DATA = "update_data"  # Update existing data
    
    # General
    COMPANY_PROFILE = "company_profile"  # Company overview
    STOCK_PRICE = "stock_price"  # Stock price queries
    ANALYST_ESTIMATES = "analyst_estimates"  # Analyst recommendations
    GENERAL_QUESTION = "general_question"  # General financial questions
    UNKNOWN = "unknown"


@dataclass
class DetectedIntent:
    """Result of intent detection."""
    
    category: IntentCategory
    confidence: float  # 0.0 to 1.0
    ticker: Optional[str] = None
    entities: dict = field(default_factory=dict)  # Extracted entities
    parameters: dict = field(default_factory=dict)  # Tool parameters
    reasoning: str = ""  # Why this intent was detected


# Intent classification patterns and keywords
INTENT_PATTERNS = {
    IntentCategory.SEC_FILING_QUERY: {
        "keywords": [
            "10-K", "10-Q", "8-K", "SEC filing", "annual report", "quarterly report",
            "risk factors", "MD&A", "management discussion", "financial statements",
            "disclosure", "item 1A", "item 7", "item 8", "notes to financial"
        ],
        "patterns": [
            r"what does.*10[-KQ].*say",
            r"according to.*SEC filing",
            r"disclosed in.*filing",
            r"risk factors.*mention",
        ]
    },
    IntentCategory.EARNINGS_QUERY: {
        "keywords": [
            "earnings call", "transcript", "management commentary", "guidance",
            "analyst Q&A", "quarterly results", "CEO said", "CFO said",
            "forward looking", "outlook", "revenue guidance", "EPS guidance"
        ],
        "patterns": [
            r"earnings call.*said",
            r"management.*comment",
            r"guidance.*quarter",
            r"analyst.*asked",
        ]
    },
    IntentCategory.NEWS_QUERY: {
        "keywords": [
            "news", "recent", "latest", "announcement", "press release",
            "upgrade", "downgrade", "analyst rating", "price target",
            "merger", "acquisition", "lawsuit", "investigation", "scandal"
        ],
        "patterns": [
            r"recent news.*",
            r"latest.*announcement",
            r"news article",
            r"stock.*up.*down",
        ]
    },
    IntentCategory.FINANCIAL_DATA_QUERY: {
        "keywords": [
            "revenue", "income", "profit", "margin", "EPS", "EBITDA",
            "assets", "liabilities", "equity", "cash flow", "debt",
            "balance sheet", "income statement", "cash flow statement"
        ],
        "patterns": [
            r"revenue.*quarter",
            r"profit margin",
            r"debt.*equity",
            r"cash flow.*year",
        ]
    },
    IntentCategory.COMPREHENSIVE_ANALYSIS: {
        "keywords": [
            "comprehensive analysis", "full analysis", "analyze", "investment thesis",
            "buy or sell", "recommendation", "should I invest", "worth buying",
            "investment opportunity", "deep dive", "complete picture"
        ],
        "patterns": [
            r"analyze.*company",
            r"comprehensive.*analysis",
            r"investment.*thesis",
            r"should.*invest",
        ]
    },
    IntentCategory.RISK_ANALYSIS: {
        "keywords": [
            "risk", "risky", "concerns", "red flag", "warning", "threat",
            "challenge", "headwind", "vulnerability", "exposure", "litigation"
        ],
        "patterns": [
            r"what.*risk",
            r"risky.*investment",
            r"concerns.*about",
            r"downside.*risk",
        ]
    },
    IntentCategory.VALUATION_ANALYSIS: {
        "keywords": [
            "valuation", "overvalued", "undervalued", "fair value", "intrinsic value",
            "P/E ratio", "price target", "DCF", "discounted cash flow",
            "comparable", "multiple", "EV/EBITDA"
        ],
        "patterns": [
            r"overvalued.*undervalued",
            r"fair value",
            r"valuation.*metric",
            r"worth.*price",
        ]
    },
    IntentCategory.RATIO_ANALYSIS: {
        "keywords": [
            "ratio", "metric", "ROE", "ROA", "current ratio", "debt-to-equity",
            "profit margin", "asset turnover", "P/E", "P/B", "EV/EBITDA",
            "financial health", "liquidity", "leverage", "efficiency"
        ],
        "patterns": [
            r"calculate.*ratio",
            r"financial.*metric",
            r"ROE.*ROA",
            r"debt.*equity ratio",
        ]
    },
    IntentCategory.INGEST_DATA: {
        "keywords": [
            "ingest", "download", "fetch", "get data", "add to knowledge",
            "update database", "refresh data", "pull filings"
        ],
        "patterns": [
            r"ingest.*data",
            r"download.*filings",
            r"add.*knowledge",
            r"refresh.*data",
        ]
    },
    IntentCategory.COMPANY_PROFILE: {
        "keywords": [
            "company profile", "about", "overview", "what does", "business model",
            "sector", "industry", "CEO", "headquarters", "employees", "founded"
        ],
        "patterns": [
            r"what does.*do",
            r"tell me about",
            r"company.*overview",
            r"who.*CEO",
        ]
    },
    IntentCategory.STOCK_PRICE: {
        "keywords": [
            "stock price", "share price", "trading", "stock performance",
            "YTD return", "52 week", "high", "low", "market cap"
        ],
        "patterns": [
            r"stock.*price",
            r"share.*price",
            r"trading.*at",
            r"performance.*year",
        ]
    },
    IntentCategory.ANALYST_ESTIMATES: {
        "keywords": [
            "analyst", "consensus", "estimate", "forecast", "projection",
            "buy rating", "sell rating", "hold rating", "price target",
            "Wall Street", "recommendation"
        ],
        "patterns": [
            r"analyst.*estimate",
            r"consensus.*rating",
            r"price target",
            r"Wall Street.*think",
        ]
    },
}

# Ticker extraction patterns
TICKER_PATTERNS = [
    r"\b([A-Z]{1,5})\b",  # Uppercase tickers (AAPL, MSFT)
    r"\$([A-Z]{1,5})\b",  # With dollar sign ($AAPL)
    r"\(([A-Z]{1,5})\)",  # In parentheses (AAPL)
]

# Company name to ticker mapping
COMPANY_TO_TICKER = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "nvidia": "NVDA",
    "jp morgan": "JPM",
    "jpmorgan": "JPM",
    "bank of america": "BAC",
    "wells fargo": "WFC",
    "netflix": "NFLX",
    "disney": "DIS",
    "coinbase": "COIN",
    "paypal": "PYPL",
    "block": "SQ",
    "square": "SQ",
    "uber": "UBER",
    "lyft": "LYFT",
    "intel": "INTC",
    "amd": "AMD",
}


class IntentDetector:
    """
    Detects user intent and routes to appropriate agent/tools.
    Uses a combination of:
    1. Keyword matching
    2. Pattern matching
    3. LLM-based classification for ambiguous cases
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or get_llm_client()
        self.confidence_threshold = 0.7  # Below this, use LLM fallback

    async def detect_intent(self, query: str) -> DetectedIntent:
        """
        Detect the intent of a user query.
        
        Process:
        1. Extract entities (ticker, dates, etc.)
        2. Rule-based classification (keywords/patterns)
        3. LLM classification if confidence is low
        4. Build parameters for tool invocation
        """
        logger.info(f"Detecting intent for query: {query[:100]}...")
        
        # Step 1: Extract entities
        entities = self._extract_entities(query)
        
        # Step 2: Rule-based classification
        category, confidence, reasoning = self._rule_based_classify(query)
        
        # Step 3: LLM classification if needed
        if confidence < self.confidence_threshold:
            llm_result = await self._llm_classify(query)
            if llm_result["confidence"] > confidence:
                category = llm_result["category"]
                confidence = llm_result["confidence"]
                reasoning = llm_result["reasoning"]
        
        # Step 4: Build tool parameters
        parameters = self._build_parameters(category, entities, query)
        
        detected = DetectedIntent(
            category=category,
            confidence=confidence,
            ticker=entities.get("ticker"),
            entities=entities,
            parameters=parameters,
            reasoning=reasoning,
        )
        
        logger.info(f"Detected intent: {category} (confidence: {confidence:.2f})")
        return detected

    def _extract_entities(self, query: str) -> dict:
        """Extract entities from query (ticker, dates, numbers, etc.)."""
        entities = {}
        query_upper = query.upper()
        query_lower = query.lower()
        
        # Extract ticker symbol
        ticker = self._extract_ticker(query)
        if ticker:
            entities["ticker"] = ticker
        
        # Extract company name and map to ticker
        for company, ticker_symbol in COMPANY_TO_TICKER.items():
            if company in query_lower:
                entities["company_name"] = company
                if not ticker:
                    entities["ticker"] = ticker_symbol
                break
        
        # Extract time periods
        time_patterns = {
            "quarterly": ["quarterly", "Q1", "Q2", "Q3", "Q4", "last quarter"],
            "annual": ["annual", "yearly", "fiscal year", "10-K"],
            "recent": ["recent", "latest", "last", "current"],
            "1Y": ["1 year", "1Y", "past year"],
            "5Y": ["5 year", "5Y", "past 5 years"],
        }
        for period, keywords in time_patterns.items():
            if any(kw in query_lower for kw in keywords):
                entities["period"] = period
                break
        
        # Extract form types
        form_types = ["10-K", "10-Q", "8-K", "DEF 14A"]
        for form_type in form_types:
            if form_type in query_upper:
                entities["form_type"] = form_type
                break
        
        # Extract numbers (for limits, thresholds)
        import re
        numbers = re.findall(r'\b(\d+)\b', query)
        if numbers:
            entities["numbers"] = [int(n) for n in numbers]
        
        return entities

    def _extract_ticker(self, query: str) -> Optional[str]:
        """Extract ticker symbol from query."""
        import re
        
        # Try pattern matching
        for pattern in TICKER_PATTERNS:
            match = re.search(pattern, query.upper())
            if match:
                ticker = match.group(1)
                # Filter out common false positives
                if ticker not in ["THE", "AND", "FOR", "WITH", "ABOUT"]:
                    return ticker
        
        return None

    def _rule_based_classify(self, query: str) -> tuple[IntentCategory, float, str]:
        """Classify intent using keyword and pattern matching."""
        query_lower = query.lower()
        scores = {}
        
        for category, patterns in INTENT_PATTERNS.items():
            score = 0.0
            matched_keywords = []
            matched_patterns = []
            
            # Keyword matching
            for keyword in patterns["keywords"]:
                if keyword.lower() in query_lower:
                    score += 0.15
                    matched_keywords.append(keyword)
            
            # Pattern matching
            import re
            for pattern in patterns["patterns"]:
                if re.search(pattern, query_lower):
                    score += 0.3
                    matched_patterns.append(pattern)
            
            # Cap score at 1.0
            scores[category] = min(score, 1.0)
        
        # Get best match
        if not scores:
            return IntentCategory.UNKNOWN, 0.0, "No matching patterns found"
        
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]
        
        # Build reasoning
        reasoning = f"Matched category {best_category.value} with score {best_score:.2f}"
        
        return best_category, best_score, reasoning

    async def _llm_classify(self, query: str) -> dict:
        """Fallback LLM-based classification for ambiguous queries."""
        categories_list = "\n".join([f"- {c.value}" for c in IntentCategory])
        
        prompt = f"""Classify the following financial query into one of these intent categories:

{categories_list}

Query: "{query}"

Respond in JSON format:
{{
    "category": "<category from list above>",
    "confidence": <0.0 to 1.0>,
    "reasoning": "<brief explanation>"
}}
"""
        
        try:
            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=200,
            )
            
            # Parse JSON response
            import json
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response[start:end])
                return {
                    "category": IntentCategory(result.get("category", "unknown")),
                    "confidence": float(result.get("confidence", 0.5)),
                    "reasoning": result.get("reasoning", "LLM classification"),
                }
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
        
        # Default fallback
        return {
            "category": IntentCategory.GENERAL_QUESTION,
            "confidence": 0.5,
            "reasoning": "LLM classification failed, using fallback",
        }

    def _build_parameters(self, category: IntentCategory, entities: dict, query: str) -> dict:
        """Build tool parameters based on detected intent."""
        params = {}
        
        # Always include ticker if available
        if "ticker" in entities:
            params["ticker"] = entities["ticker"]
        
        # Category-specific parameters
        if category == IntentCategory.SEC_FILING_QUERY:
            params["query"] = query
            params["form_type"] = entities.get("form_type", "all")
            
        elif category == IntentCategory.EARNINGS_QUERY:
            params["query"] = query
            
        elif category == IntentCategory.NEWS_QUERY:
            params["query"] = query
            
        elif category == IntentCategory.FINANCIAL_DATA_QUERY:
            params["statement_type"] = "all"
            params["period"] = entities.get("period", "annual")
            params["limit"] = entities.get("numbers", [5])[0] if entities.get("numbers") else 5
            
        elif category in [
            IntentCategory.COMPREHENSIVE_ANALYSIS,
            IntentCategory.RISK_ANALYSIS,
            IntentCategory.VALUATION_ANALYSIS,
            IntentCategory.EARNINGS_ANALYSIS,
        ]:
            params["analysis_type"] = category.value.replace("_analysis", "")
            
        elif category == IntentCategory.INGEST_DATA:
            params["form_types"] = ["10-K", "10-Q"]
            params["filing_count"] = entities.get("numbers", [3])[0] if entities.get("numbers") else 3
            
        elif category == IntentCategory.STOCK_PRICE:
            params["period"] = entities.get("period", "1Y")
        
        return params


# Singleton instance
_intent_detector: Optional[IntentDetector] = None


def get_intent_detector() -> IntentDetector:
    """Get or create the intent detector singleton."""
    global _intent_detector
    if _intent_detector is None:
        _intent_detector = IntentDetector()
    return _intent_detector

"""
Tool definitions for the Financial Analyst Agent.
Each tool maps to a capability the agent can invoke via function calling.
"""

AGENT_TOOLS = [
    {
        "name": "search_sec_filings",
        "description": (
            "Search and retrieve SEC filings (10-K annual reports, 10-Q quarterly reports, "
            "8-K current reports) from the knowledge base. Use this to find information from "
            "a company's official SEC filings including financial statements, risk factors, "
            "MD&A (management discussion and analysis), and business descriptions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant SEC filing content",
                },
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., AAPL, MSFT) to filter results",
                },
                "form_type": {
                    "type": "string",
                    "enum": ["10-K", "10-Q", "8-K", "all"],
                    "description": "Type of SEC filing to search",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_earnings_calls",
        "description": (
            "Search earnings call transcripts for management commentary, guidance, "
            "analyst Q&A, and forward-looking statements. Use this to understand "
            "what company leadership said about performance, strategy, and outlook."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for earnings call content",
                },
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker to filter results",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_financial_news",
        "description": (
            "Search recent financial news articles for a company or topic. "
            "Returns news with sentiment analysis. Use for recent developments, "
            "analyst upgrades/downgrades, M&A activity, and market-moving events."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for financial news",
                },
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker to filter news",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_financial_statements",
        "description": (
            "Retrieve structured financial statements (income statement, balance sheet, "
            "cash flow statement) from financial data APIs. Returns numerical data that "
            "can be used for ratio analysis and financial modeling."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "statement_type": {
                    "type": "string",
                    "enum": ["income_statement", "balance_sheet", "cash_flow", "all"],
                    "description": "Which financial statement to retrieve",
                },
                "period": {
                    "type": "string",
                    "enum": ["annual", "quarterly"],
                    "description": "Annual or quarterly statements",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of periods to retrieve",
                    "default": 5,
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_company_profile",
        "description": (
            "Get a company overview including sector, industry, market cap, CEO, "
            "employee count, and business description. Good starting point for "
            "understanding a company."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate_financial_ratios",
        "description": (
            "Calculate comprehensive financial ratios including profitability "
            "(margins, ROE, ROA), liquidity (current ratio, quick ratio), "
            "leverage (debt-to-equity, interest coverage), efficiency (asset turnover, "
            "DSO), valuation (P/E, EV/EBITDA), and cash flow quality metrics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_stock_price",
        "description": (
            "Get historical stock price data including open, high, low, close, "
            "and volume. Use for price trend analysis and performance tracking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "period": {
                    "type": "string",
                    "enum": ["1M", "3M", "6M", "1Y", "5Y"],
                    "description": "Historical period",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "run_comprehensive_analysis",
        "description": (
            "Run a full comprehensive analysis on a company combining SEC filings, "
            "earnings calls, financial data, ratio analysis, and LLM-powered synthesis. "
            "This produces an investment-research-grade report covering business overview, "
            "financial performance, valuation, risks, and catalysts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol to analyze",
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["comprehensive", "earnings", "risk", "valuation"],
                    "description": "Type of analysis to perform",
                    "default": "comprehensive",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "ingest_company_data",
        "description": (
            "Ingest SEC filings, earnings calls, and news for a company into the "
            "knowledge base. This must be done before searching for a company's data. "
            "Fetches documents from SEC EDGAR and news APIs, chunks them, and stores "
            "embeddings in the vector database."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "form_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "SEC form types to ingest (default: 10-K, 10-Q)",
                },
                "filing_count": {
                    "type": "integer",
                    "description": "Number of recent filings per form type",
                    "default": 3,
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_analyst_estimates",
        "description": (
            "Get analyst consensus estimates including revenue and EPS forecasts, "
            "number of covering analysts, and buy/hold/sell recommendations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
            },
            "required": ["ticker"],
        },
    },
]

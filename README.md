# AI Financial Analyst Agent

Enterprise RAG-powered financial analysis platform that combines SEC filings, earnings call transcripts, financial data APIs, and LLM-driven analysis to produce investment-research-grade insights.

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AI Financial Analyst System                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                     │
│  │   CLI        │     │   FastAPI    │     │   Chat       │                     │
│  │   Interface  │     │   Server     │     │   Interface  │                     │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘                     │
│         │                    │                    │                             │
│         └────────────────────┼────────────────────┘                             │
│                              │                                                  │
│                     ┌────────▼────────┐                                         │
│                     │   Agent         │                                         │
│                     │   Orchestrator  │                                         │
│                     │   (Tool Router) │                                         │
│                     └────────┬────────┘                                         │
│                              │                                                  │
│         ┌────────────────────┼────────────────────┐                             │
│         │                    │                    │                             │
│         ▼                    ▼                    ▼                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                       │
│  │   RAG        │    │   Financial  │    │   Analysis   │                       │
│  │   Retriever  │    │   Data APIs  │    │   Engine     │                       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                       │
│         │                   │                   │                               │
└─────────┼───────────────────┼───────────────────┼───────────────────────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                              Data Layer                                        │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │  Ingestion Pipeline                                                       │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │ │
│  │  │ SEC EDGAR   │  │ Earnings    │  │ Financial   │  │ RSS Feeds   │       │ │
│  │  │ Downloader  │  │ Calls       │  │ News APIs   │  │ (Fallback)  │       │ │
│  │  │             │  │ Processor   │  │ (Finnhub,   │  │             │       │ │
│  │  │ • 10-K      │  │             │  │  Alpha      │  │ • MarketWatch│      │ │
│  │  │ • 10-Q      │  │ • Transcripts│ │  Vantage)   │  │ • CNBC      │       │ │
│  │  │ • 8-K       │  │ • Speakers  │  │             │  │ • Business  │       │ │
│  │  │             │  │ • Q&A       │  │ • Headlines │  │   Insider   │       │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │ │
│  │         │                │                │                │              │ │
│  │         └────────────────┴────────────────┴────────────────┘              │ │
│  │                              │                                            │ │
│  │                     ┌────────▼────────┐                                   │ │
│  │                     │ Text Processing │                                   │ │
│  │                     │ • Cleaning      │                                   │ │
│  │                     │ • Chunking      │                                   │ │
│  │                     │ • Section Split │                                   │ │
│  │                     └────────┬────────┘                                   │ │
│  └──────────────────────────────┼────────────────────────────────────────────┘ │
│                                 │                                              │
│                                 ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │  Embedding & Vector Store                                                 │ │
│  │  ┌─────────────────┐    ┌─────────────────────────────────────────────┐   │ │
│  │  │ Ollama /        │    │ ChromaDB Collections                        │   │ │
│  │  │ Sentence-       │───▶│ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │   │ │
│  │  │ Transformers    │    │ │ SEC Filings │ │ Earnings    │ │    News │ │   │ │
│  │  │ (768-dim)       │    │ │ (10-K,10-Q) │ │ Calls       │ │         │ │   │ │
│  │  └─────────────────┘    │ └─────────────┘ └─────────────┘ └─────────┘ │   │ │
│  │                         └─────────────────────────────────────────────┘   │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │  Local Storage (./data/)                                                  │ │
│  │  ├── sec_filings/     (JSON metadata for downloaded filings)              │ │
│  │  ├── news/            (Fetched news articles)                             │ │
│  │  ├── earnings_calls/  (Transcripts)                                       │ │
│  │  └── vector_store/    (ChromaDB persistent embeddings)                    │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              LLM Layer                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │  Supported Providers (switch via LM_PROVIDER in .env)                     │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │  │
│  │  │ Anthropic    │  │ OpenAI       │  │ Ollama       │  │ llama.cpp    │   │  │
│  │  │ Claude       │  │ GPT-4        │  │ Local        │  │ GGUF Models  │   │  │
│  │  │ (Recommended)│  │              │  │ (Qwen3)      │  │              │   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Output Layer                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ Structured      │  │ Citations &     │  │ Financial       │                  │
│  │ Reports         │  │ Sources         │  │ Ratios &        │                  │
│  │ (Markdown/JSON) │  │                 │  │ Metrics         │                  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Ingest    │────▶│   Process   │────▶│   Embed     │────▶│   Store     │
│   (Fetch)   │     │   (Clean)   │     │   (Vector)  │     │   (ChromaDB)│
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
  SEC Filings         Remove HTML       Generate          Persist to
  News Articles       Split sections    768-dim           disk
  Earnings Calls      Chunk text        embeddings        ./data/vector_store/
```

### Query Flow

```
User Question ──▶ Agent Orchestrator ──▶ Tool Selection
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
                    ▼                         ▼                         ▼
            ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
            │ RAG Retrieval │         │ Financial     │         │ Analysis      │
            │ (Vector DB)   │         │ Data Fetch    │         │ Calculations  │
            └───────┬───────┘         └───────┬───────┘         └───────┬───────┘
                    │                         │                         │
                    └─────────────────────────┼─────────────────────────┘
                                              │
                                              ▼
                                      ┌───────────────┐
                                      │ LLM Synthesis │
                                      │ + Citations   │
                                      └───────┬───────┘
                                              │
                                              ▼
                                      ┌───────────────┐
                                      │ Final Answer  │
                                      └───────────────┘
```

## LLM Providers

Supports 4 LLM backends — switch via `LLM_PROVIDER` in `.env`:

| Provider | Config | Best For |
|----------|--------|----------|
| **Anthropic** | `LLM_PROVIDER=anthropic` | Highest quality analysis |
| **OpenAI** | `LLM_PROVIDER=openai` | Strong alternative |
| **Ollama** | `LLM_PROVIDER=ollama` | Local/private, no API costs |
| **llama.cpp** | `LLM_PROVIDER=llamacpp` | Direct GGUF model loading |

### Ollama Setup (Recommended for Local)

```bash
# Install & start Ollama
ollama serve

# Pull models
ollama pull qwen3:latest            # LLM
ollama pull nomic-embed-text-v2-moe # Embeddings
```

### llama.cpp Setup

```bash
# Option A: llama-server (HTTP API)
./llama-server -m model.gguf --port 8080

# Option B: Python bindings (in-process)
pip install llama-cpp-python
# Set LLAMACPP_MODEL_PATH=/path/to/model.gguf in .env
```

## Quick Start

```bash
# 1. Clone and install
cd ai_financial_analyst
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your API keys and LLM provider choice

# 3. Ingest company data
python main.py ingest AAPL --forms 10-K 10-Q --count 3

# 4. Run analysis
python main.py analyze AAPL --type comprehensive

# 5. Interactive chat
python main.py chat

# 6. Start API server
python main.py serve
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/query` | Ask any financial question |
| POST | `/api/v1/ingest` | Ingest SEC filings + news |
| POST | `/api/v1/ingest/earnings-call` | Ingest earnings transcript |
| POST | `/api/v1/analyze` | Run structured analysis |
| GET | `/api/v1/financials/{ticker}` | Get financial statements |
| GET | `/api/v1/profile/{ticker}` | Get company profile |
| GET | `/api/v1/ratios/{ticker}` | Calculate financial ratios |
| GET | `/api/v1/knowledge-base/stats` | Vector store statistics |
| GET | `/api/v1/config` | Current configuration |
| GET | `/health` | Health check |

### Example API Call

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are Apple risk factors from their latest 10-K?", "ticker": "AAPL"}'
```

## Financial Data Sources

- **SEC EDGAR**: 10-K, 10-Q, 8-K filings (free, rate-limited)
- **Financial Modeling Prep**: Statements, ratios, estimates (free tier available)
- **Alpha Vantage**: Prices, fundamentals, news (free tier)
- **Finnhub**: News, estimates, recommendations (free tier)
- **Polygon.io**: Market data (free tier)
- **RSS Feeds**: Financial news from MarketWatch, CNBC, Business Insider (free, no API key required)

### News Ingestion Priority

The system fetches news using the following priority:

1. **Finnhub API** (if `FINNHUB_API_KEY` is set)
2. **Alpha Vantage News** (if `ALPHA_VANTAGE_API_KEY` is set)
3. **RSS Feeds** (automatic fallback — no configuration needed)

RSS feeds search for company names (e.g., "Apple" for AAPL) in article titles and content. If company-specific news is unavailable, general market news is fetched as a fallback.

### Supported RSS Feeds

| Source | Feed Type |
|--------|-----------|
| MarketWatch | Top Stories |
| CNBC | Top News & Technology |
| Investing.com | Technology News |
| Business Insider | General Business |

### Company Name Mapping

The system automatically maps ticker symbols to company names for better RSS matching:

- **AAPL**: Apple, Apple Inc
- **MSFT**: Microsoft, Microsoft Corp
- **GOOGL/GOOG**: Google, Alphabet
- **AMZN**: Amazon, Amazon.com
- **TSLA**: Tesla, Tesla Inc
- **META**: Meta, Facebook
- **NVDA**: Nvidia, NVIDIA
- **JPM**: JPMorgan, J.P. Morgan
- **BAC**: Bank of America, BofA
- **WFC**: Wells Fargo
- **AMD**: AMD, Advanced Micro Devices
- **INTC**: Intel
- **NFLX**: Netflix
- **DIS**: Disney, Walt Disney
- **COIN**: Coinbase
- **PYPL**: PayPal
- **SQ**: Block, Square
- **UBER**: Uber
- **LYFT**: Lyft

To add more companies, edit the `COMPANY_NAMES` dictionary in `ingestion/financial_news.py`.

## Analysis Capabilities

- **Comprehensive Analysis**: Full investment research report
- **Earnings Analysis**: Quarter-over-quarter performance
- **Risk Analysis**: Operational, financial, regulatory, market risks
- **Valuation Analysis**: DCF framework, comparable multiples
- **30+ Financial Ratios**: Profitability, liquidity, leverage, efficiency, valuation, cash flow
- **Variance Analysis**: Budget vs. actual with driver decomposition

## Project Structure

```
ai_financial_analyst/
├── main.py                  # CLI entry point
├── config/
│   └── settings.py          # Pydantic settings (env vars)
├── core/
│   ├── llm_provider.py      # Anthropic/OpenAI/Ollama/llama.cpp
│   ├── embeddings.py        # Sentence-transformers/OpenAI/Ollama
│   ├── vector_store.py      # ChromaDB manager
│   └── text_processing.py   # Financial text chunking
├── ingestion/
│   ├── sec_edgar.py         # SEC EDGAR downloader/parser
│   ├── earnings_calls.py    # Earnings call processor
│   ├── financial_news.py    # Multi-source news aggregator
│   └── pipeline.py          # Unified ingestion orchestrator
├── rag/
│   └── retriever.py         # Multi-collection RAG retriever
├── analysis/
│   ├── financial_data.py    # Financial data API client
│   ├── ratios.py            # 30+ financial ratio calculator
│   └── financial_analyzer.py # LLM-powered analysis engine
├── agent/
│   ├── orchestrator.py      # Agent brain (tool routing)
│   └── tools.py             # Tool definitions for function calling
├── reports/
│   └── generator.py         # Markdown/JSON report generation
├── api/
│   └── server.py            # FastAPI endpoints
├── tests/
│   ├── test_ratios.py
│   └── test_text_processing.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Testing

```bash
pytest tests/ -v --cov=.
```

## Data Storage

Ingested data is stored locally in the `./data/` directory:

```
data/
├── sec_filings/           # Downloaded SEC filing metadata
│   ├── Apple_Inc__10-K_2025-10-31.json
│   ├── Apple_Inc__10-Q_2026-01-30.json
│   └── ...
├── news/                  # Fetched news articles
│   ├── news_AAPL_20260312_234023.json
│   └── ...
├── earnings_calls/        # Earnings call transcripts (if available)
└── vector_store/          # ChromaDB vector embeddings
    ├── sec_filings/
    ├── earnings_calls/
    └── financial_news/
```

### Vector Store Collections

The system maintains separate ChromaDB collections for different data types:

| Collection | Description |
|------------|-------------|
| `sec_filings` | SEC 10-K, 10-Q, 8-K filings |
| `earnings_calls` | Earnings call transcripts |
| `financial_news` | Financial news articles |

## Recent Updates

### Bug Fixes (March 2026)

1. **Regex Pattern Fix**: Fixed "global flags not at the start of the expression" error in SEC filing text processing by properly handling `(?i)` flags in combined regex patterns.

2. **ChromaDB Compatibility**: Fixed "'str' object is not callable" error by converting `ChromaEmbeddingFunction.name` from attribute to method for ChromaDB API compatibility.

### New Features (March 2026)

1. **RSS Feed News Ingestion**: Added automatic fallback to RSS feeds for financial news when API keys are not available. Supports MarketWatch, CNBC, Investing.com, and Business Insider.

2. **Company Name Matching**: Enhanced RSS news matching with company name mappings for 20+ major tech and finance companies (e.g., "Apple" for AAPL, "Microsoft" for MSFT).

3. **Intent-Based Agent Routing**: Implemented intelligent intent detection and multi-agent orchestration system:
   - **Intent Detector**: Classifies queries into 15+ intent categories using keyword matching, pattern recognition, and LLM fallback
   - **Research Agent**: Specializes in SEC filings, earnings calls, and news search
   - **Analysis Agent**: Handles comprehensive analysis, risk assessment, valuation, and ratio calculations
   - **Data Agent**: Manages financial statements, company profiles, stock prices, and data ingestion
   - Automatic routing based on detected intent for optimal task execution

## Intent Categories

The system recognizes the following intent categories:

| Category | Description | Example Queries |
|----------|-------------|-----------------|
| `sec_filing_query` | SEC filing search | "What did Apple's 10-K say about risk factors?" |
| `earnings_query` | Earnings call transcripts | "What did the CEO say about guidance on the earnings call?" |
| `news_query` | Financial news search | "Recent news about Microsoft" |
| `financial_data_query` | Financial statements | "Show me Tesla's income statement for last quarter" |
| `comprehensive_analysis` | Full analysis | "Analyze whether I should invest in NVDA" |
| `risk_analysis` | Risk assessment | "What are the risks of investing in AMZN?" |
| `valuation_analysis` | Valuation metrics | "Is GOOGL overvalued or undervalued?" |
| `ratio_analysis` | Financial ratios | "Calculate ROE and debt-to-equity for JPM" |
| `company_profile` | Company overview | "Tell me about Apple's business model" |
| `stock_price` | Stock price data | "What's the stock price performance for TSLA this year?" |
| `analyst_estimates` | Analyst recommendations | "What are analysts saying about AAPL?" |
| `ingest_data` | Data ingestion | "Download SEC filings for MSFT" |

## Agent Architecture

```
                    User Query
                        │
                        ▼
                ┌───────────────┐
                │   Intent      │
                │   Detector    │
                └───────┬───────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Research    │ │  Analysis    │ │    Data      │
│   Agent      │ │   Agent      │ │    Agent     │
│              │ │              │ │              │
│ • SEC Search │ │ • Comprehensive│ │ • Financial  │
│ • Earnings   │ │ • Risk       │ │   Statements │
│ • News       │ │ • Valuation  │ │ • Profile    │
│              │ │ • Ratios     │ │ • Stock Price│
└──────────────┘ └──────────────┘ └──────────────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
                        ▼
                ┌───────────────┐
                │   Response    │
                │   Synthesis   │
                └───────┬───────┘
                        │
                        ▼
                  Final Answer
                  + Citations
```

## Docker

```bash
# With Ollama
docker-compose up -d

# Pull models inside the Ollama container
docker exec -it ollama ollama pull qwen3:latest
docker exec -it ollama ollama pull nomic-embed-text-v2-moe:latest
```

## Disclaimer

This tool is for **informational purposes only** and does not constitute investment advice. Always consult qualified financial professionals before making investment decisions.

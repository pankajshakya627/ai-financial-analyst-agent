# AI Financial Analyst Agent

Enterprise RAG-powered financial analysis platform that combines SEC filings, earnings call transcripts, financial data APIs, and LLM-driven analysis to produce investment-research-grade insights.

## Architecture

```
User Question
      |
      v
  Agent Orchestrator (tool routing via LLM function calling)
      |
      +---> RAG Retriever (ChromaDB vector search)
      |         |
      |         +---> SEC Filings (10-K, 10-Q, 8-K)
      |         +---> Earnings Call Transcripts
      |         +---> Financial News
      |
      +---> Financial Data APIs
      |         |
      |         +---> Income Statements
      |         +---> Balance Sheets
      |         +---> Cash Flow Statements
      |         +---> Stock Prices & Estimates
      |
      +---> Analysis Engine
      |         |
      |         +---> 30+ Financial Ratios
      |         +---> Variance Analysis
      |         +---> Risk Assessment
      |         +---> Valuation Framework
      |
      v
  LLM Synthesis (Anthropic / OpenAI / Ollama / llama.cpp)
      |
      v
  Structured Report + Citations
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

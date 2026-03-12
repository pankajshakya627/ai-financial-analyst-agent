"""
Centralized configuration for the AI Financial Analyst Agent.
Loads from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from enum import Enum


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- LLM Configuration ---
    llm_provider: LLMProvider = Field(
        default=LLMProvider.ANTHROPIC,
        description="Active LLM provider: 'anthropic' or 'openai'",
    )
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    anthropic_model: str = Field(default="claude-sonnet-4-20250514", description="Anthropic model ID")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model ID")
    llm_temperature: float = Field(default=0.1, description="LLM temperature for analysis tasks")
    llm_max_tokens: int = Field(default=4096, description="Maximum response tokens")

    # --- Ollama Configuration ---
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_model: str = Field(default="qwen3:latest", description="Ollama model name")
    ollama_embedding_model: str = Field(
        default="nomic-embed-text-v2-moe:latest",
        description="Ollama embedding model for RAG",
    )

    # --- llama.cpp Configuration ---
    llamacpp_base_url: str = Field(
        default="http://localhost:8080",
        description="llama.cpp server URL (llama-server --port 8080)",
    )
    llamacpp_model_path: Optional[str] = Field(
        default=None, description="Path to GGUF model file (for llama-cpp-python)"
    )
    llamacpp_n_ctx: int = Field(default=8192, description="llama.cpp context window size")
    llamacpp_n_gpu_layers: int = Field(default=-1, description="GPU layers (-1 = all)")

    # --- Embedding Configuration ---
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.SENTENCE_TRANSFORMERS,
        description="Embedding provider for RAG pipeline",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model name (sentence-transformers or OpenAI)",
    )
    embedding_dimension: int = Field(default=384, description="Embedding vector dimension")

    # --- ChromaDB Configuration ---
    chroma_persist_dir: str = Field(default="./data/vector_store", description="ChromaDB persistence directory")
    chroma_collection_sec: str = Field(default="sec_filings", description="Collection name for SEC filings")
    chroma_collection_earnings: str = Field(
        default="earnings_calls", description="Collection name for earnings calls"
    )
    chroma_collection_news: str = Field(default="financial_news", description="Collection name for financial news")

    # --- RAG Configuration ---
    chunk_size: int = Field(default=1000, description="Text chunk size in characters for RAG")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    top_k_results: int = Field(default=10, description="Number of top results to retrieve")
    similarity_threshold: float = Field(default=0.3, description="Minimum similarity score for retrieval")

    # --- Financial Data API Keys ---
    sec_edgar_user_agent: str = Field(
        default="FinancialAnalystAgent/1.0 (contact@example.com)",
        description="SEC EDGAR user agent (required by SEC)",
    )
    alpha_vantage_api_key: Optional[str] = Field(default=None, description="Alpha Vantage API key")
    finnhub_api_key: Optional[str] = Field(default=None, description="Finnhub API key")
    polygon_api_key: Optional[str] = Field(default=None, description="Polygon.io API key")
    fmp_api_key: Optional[str] = Field(default=None, description="Financial Modeling Prep API key")

    # --- Server Configuration ---
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    cors_origins: list[str] = Field(default=["*"], description="Allowed CORS origins")

    # --- Data Paths ---
    data_dir: str = Field(default="./data", description="Root data directory")
    sec_filings_dir: str = Field(default="./data/sec_filings", description="SEC filings storage")
    earnings_calls_dir: str = Field(default="./data/earnings_calls", description="Earnings call transcripts")
    reports_dir: str = Field(default="./data/reports", description="Generated reports output directory")

    # --- Logging ---
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        description="Log format string",
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "case_sensitive": False}


# Singleton
settings = Settings()

"""
AI Financial Analyst Agent — Entry Point

Usage:
    # Start the API server
    python main.py serve

    # Ingest data for a company
    python main.py ingest AAPL --forms 10-K 10-Q --count 3

    # Run analysis
    python main.py analyze AAPL --type comprehensive

    # Interactive chat mode
    python main.py chat

    # Check configuration
    python main.py config
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format=settings.log_format,
    )


def cmd_serve(args):
    """Start the FastAPI server."""
    import uvicorn

    setup_logging()
    uvicorn.run(
        "api.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


def cmd_ingest(args):
    """Ingest data for a company."""
    setup_logging()
    from ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    result = asyncio.run(
        pipeline.ingest_full_company(
            ticker=args.ticker,
            sec_form_types=args.forms,
            sec_count=args.count,
            news_days_back=args.news_days,
        )
    )

    print(f"\n{'='*60}")
    print(f"Ingestion Complete for {args.ticker}")
    print(f"{'='*60}")

    for form_type, stats in result.get("sec_filings", {}).items():
        print(f"\n{form_type}:")
        print(f"  Filings found: {stats.get('filings_found', 0)}")
        print(f"  Filings processed: {stats.get('filings_processed', 0)}")
        print(f"  Chunks created: {stats.get('chunks_created', 0)}")
        if stats.get("errors"):
            print(f"  Errors: {stats['errors']}")

    news_stats = result.get("news", {})
    print(f"\nNews:")
    print(f"  Articles fetched: {news_stats.get('articles_fetched', 0)}")
    print(f"  Chunks created: {news_stats.get('chunks_created', 0)}")


def cmd_analyze(args):
    """Run financial analysis."""
    setup_logging()
    from core.llm_provider import get_llm_client
    from analysis.financial_analyzer import FinancialAnalyzer
    from reports.generator import ReportGenerator

    llm = get_llm_client()
    analyzer = FinancialAnalyzer(llm)
    report_gen = ReportGenerator()

    analysis_map = {
        "comprehensive": analyzer.comprehensive_analysis,
        "earnings": analyzer.earnings_analysis,
        "risk": analyzer.risk_analysis,
        "valuation": analyzer.valuation_analysis,
    }

    analysis_fn = analysis_map.get(args.type, analyzer.comprehensive_analysis)
    result = asyncio.run(analysis_fn(args.ticker))

    # Generate report
    report = report_gen.generate_markdown_report(result)
    report_gen.generate_json_report(result)

    print(f"\n{'='*60}")
    print(f"{args.type.title()} Analysis: {args.ticker}")
    print(f"{'='*60}")
    print(report[:3000])
    print(f"\n... Report saved to {settings.reports_dir}/")


def cmd_chat(args):
    """Interactive chat mode."""
    setup_logging()
    from core.llm_provider import get_llm_client
    from agent.orchestrator import FinancialAgentOrchestrator

    llm = get_llm_client()
    agent = FinancialAgentOrchestrator(llm)

    print("\n" + "=" * 60)
    print("AI Financial Analyst Agent — Interactive Mode")
    print(f"LLM: {settings.llm_provider.value} | Model: {_get_model_name()}")
    print("Type 'quit' to exit, 'reset' to clear history")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("You: ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if query.lower() == "reset":
                agent.reset_conversation()
                print("Conversation reset.\n")
                continue

            response = asyncio.run(agent.process_query(query))

            print(f"\nAgent: {response.answer}")
            if response.tools_used:
                print(f"\n[Tools used: {', '.join(response.tools_used)}]")
            if response.sources:
                print(f"[Sources: {len(response.sources)} documents]")
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def cmd_config(args):
    """Show current configuration."""
    print("\n" + "=" * 60)
    print("AI Financial Analyst Agent — Configuration")
    print("=" * 60)
    print(f"\nLLM Provider:      {settings.llm_provider.value}")
    print(f"LLM Model:         {_get_model_name()}")
    print(f"Embedding:         {settings.embedding_provider.value} / {settings.embedding_model}")
    print(f"Temperature:       {settings.llm_temperature}")
    print(f"Max Tokens:        {settings.llm_max_tokens}")
    print(f"\nChunk Size:        {settings.chunk_size}")
    print(f"Chunk Overlap:     {settings.chunk_overlap}")
    print(f"Top-K Results:     {settings.top_k_results}")
    print(f"\nData APIs:")
    print(f"  FMP:             {'configured' if settings.fmp_api_key else 'not set'}")
    print(f"  Alpha Vantage:   {'configured' if settings.alpha_vantage_api_key else 'not set'}")
    print(f"  Finnhub:         {'configured' if settings.finnhub_api_key else 'not set'}")
    print(f"  Polygon:         {'configured' if settings.polygon_api_key else 'not set'}")
    print(f"\nServer:            {settings.host}:{settings.port}")

    if settings.llm_provider.value == "ollama":
        print(f"\nOllama URL:        {settings.ollama_base_url}")
        print(f"Ollama Model:      {settings.ollama_model}")
        print(f"Ollama Embed:      {settings.ollama_embedding_model}")

    if settings.llm_provider.value == "llamacpp":
        print(f"\nllama.cpp URL:     {settings.llamacpp_base_url}")
        if settings.llamacpp_model_path:
            print(f"GGUF Model:        {settings.llamacpp_model_path}")
        print(f"Context Window:    {settings.llamacpp_n_ctx}")


def _get_model_name() -> str:
    """Get the active model name based on provider."""
    provider = settings.llm_provider.value
    if provider == "anthropic":
        return settings.anthropic_model
    elif provider == "openai":
        return settings.openai_model
    elif provider == "ollama":
        return settings.ollama_model
    elif provider == "llamacpp":
        return settings.llamacpp_model_path or f"server@{settings.llamacpp_base_url}"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="AI Financial Analyst Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.set_defaults(func=cmd_serve)

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest company data")
    ingest_parser.add_argument("ticker", help="Stock ticker symbol")
    ingest_parser.add_argument("--forms", nargs="+", default=["10-K", "10-Q"], help="SEC form types")
    ingest_parser.add_argument("--count", type=int, default=3, help="Filings per form type")
    ingest_parser.add_argument("--news-days", type=int, default=30, help="Days of news to fetch")
    ingest_parser.set_defaults(func=cmd_ingest)

    # analyze
    analyze_parser = subparsers.add_parser("analyze", help="Run financial analysis")
    analyze_parser.add_argument("ticker", help="Stock ticker symbol")
    analyze_parser.add_argument(
        "--type",
        choices=["comprehensive", "earnings", "risk", "valuation"],
        default="comprehensive",
        help="Analysis type",
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    # chat
    chat_parser = subparsers.add_parser("chat", help="Interactive chat mode")
    chat_parser.set_defaults(func=cmd_chat)

    # config
    config_parser = subparsers.add_parser("config", help="Show configuration")
    config_parser.set_defaults(func=cmd_config)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()

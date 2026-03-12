"""
Unified ingestion pipeline that orchestrates document fetching,
parsing, chunking, and vector store insertion.
"""

import logging
from typing import Optional
from datetime import datetime

from config.settings import settings
from core.text_processing import FinancialTextSplitter, clean_financial_text
from core.vector_store import VectorStoreManager, DocumentChunk
from ingestion.sec_edgar import SECEdgarClient
from ingestion.earnings_calls import EarningsCallProcessor
from ingestion.financial_news import FinancialNewsIngester

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    End-to-end ingestion pipeline:
    1. Fetch documents (SEC filings, earnings calls, news)
    2. Parse and clean text
    3. Chunk into optimal segments
    4. Embed and store in ChromaDB
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
    ):
        self.vector_store = vector_store or VectorStoreManager()
        self.sec_client = SECEdgarClient()
        self.earnings_processor = EarningsCallProcessor()
        self.news_ingester = FinancialNewsIngester()
        self.text_splitter = FinancialTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    async def ingest_sec_filing(
        self,
        ticker: str,
        form_type: str = "10-K",
        count: int = 3,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict:
        """
        Ingest SEC filings for a company into the vector store.

        Returns:
            dict with ingestion stats (filings processed, chunks created, etc.)
        """
        stats = {"filings_found": 0, "filings_processed": 0, "chunks_created": 0, "errors": []}

        try:
            filings_list = await self.sec_client.get_company_filings(
                ticker=ticker,
                form_type=form_type,
                count=count,
                start_date=start_date,
                end_date=end_date,
            )
            stats["filings_found"] = len(filings_list)
        except Exception as e:
            stats["errors"].append(f"Failed to fetch filings list: {e}")
            return stats

        for filing_info in filings_list:
            try:
                filing = await self.sec_client.download_filing(filing_info)

                # Clean and chunk the text
                cleaned_text = clean_financial_text(filing.raw_text)
                text_chunks = self.text_splitter.split_text(cleaned_text, doc_type="sec_filing")

                # Build document chunks with rich metadata
                doc_chunks = []
                for i, chunk in enumerate(text_chunks):
                    doc_chunks.append(
                        DocumentChunk(
                            content=chunk.text,
                            metadata={
                                "source": "sec_edgar",
                                "ticker": ticker,
                                "company": filing.company_name,
                                "form_type": filing.form_type,
                                "filing_date": filing.filing_date,
                                "section": chunk.section or "unknown",
                                "chunk_index": i,
                                "accession": filing.accession_number,
                                "url": filing.url,
                            },
                            doc_id=f"sec_{ticker}_{filing.form_type}_{filing.filing_date}_{i}",
                        )
                    )

                # Also chunk individual sections for targeted retrieval
                for section_name, section_text in filing.sections.items():
                    section_cleaned = clean_financial_text(section_text)
                    section_chunks = self.text_splitter.split_text(
                        section_cleaned, doc_type="sec_filing"
                    )
                    for j, sc in enumerate(section_chunks):
                        doc_chunks.append(
                            DocumentChunk(
                                content=sc.text,
                                metadata={
                                    "source": "sec_edgar",
                                    "ticker": ticker,
                                    "company": filing.company_name,
                                    "form_type": filing.form_type,
                                    "filing_date": filing.filing_date,
                                    "section": section_name,
                                    "chunk_index": j,
                                    "is_section_chunk": True,
                                },
                                doc_id=f"sec_{ticker}_{filing.form_type}_{filing.filing_date}_{section_name}_{j}",
                            )
                        )

                # Insert into vector store
                added = self.vector_store.add_documents(
                    settings.chroma_collection_sec, doc_chunks
                )
                stats["chunks_created"] += added
                stats["filings_processed"] += 1

                logger.info(
                    f"Ingested {filing.form_type} for {ticker} "
                    f"({filing.filing_date}): {added} chunks"
                )

            except Exception as e:
                stats["errors"].append(
                    f"Failed to process {filing_info.get('filing_date', 'unknown')}: {e}"
                )
                logger.error(f"Error ingesting filing: {e}", exc_info=True)

        return stats

    async def ingest_earnings_call(
        self,
        raw_text: str,
        company: str,
        ticker: str,
        quarter: str,
        call_date: str,
    ) -> dict:
        """Ingest an earnings call transcript into the vector store."""
        stats = {"turns_processed": 0, "chunks_created": 0, "errors": []}

        try:
            transcript = self.earnings_processor.parse_raw_transcript(
                text=raw_text,
                company=company,
                ticker=ticker,
                quarter=quarter,
                call_date=call_date,
            )

            doc_chunks = []

            # Chunk prepared remarks
            for turn in transcript.prepared_remarks:
                cleaned = clean_financial_text(turn.text)
                chunks = self.text_splitter.split_text(cleaned, doc_type="earnings_call")
                for i, chunk in enumerate(chunks):
                    doc_chunks.append(
                        DocumentChunk(
                            content=chunk.text,
                            metadata={
                                "source": "earnings_call",
                                "ticker": ticker,
                                "company": company,
                                "quarter": quarter,
                                "call_date": call_date,
                                "section": "prepared_remarks",
                                "speaker": turn.speaker,
                                "speaker_role": turn.role,
                                "chunk_index": i,
                            },
                            doc_id=f"ec_{ticker}_{quarter}_{turn.speaker}_{i}",
                        )
                    )

            # Chunk Q&A
            for turn in transcript.q_and_a:
                cleaned = clean_financial_text(turn.text)
                chunks = self.text_splitter.split_text(cleaned, doc_type="earnings_call")
                for i, chunk in enumerate(chunks):
                    doc_chunks.append(
                        DocumentChunk(
                            content=chunk.text,
                            metadata={
                                "source": "earnings_call",
                                "ticker": ticker,
                                "company": company,
                                "quarter": quarter,
                                "call_date": call_date,
                                "section": "q_and_a",
                                "speaker": turn.speaker,
                                "speaker_role": turn.role,
                                "chunk_index": i,
                            },
                            doc_id=f"ec_{ticker}_{quarter}_qa_{turn.speaker}_{i}",
                        )
                    )

            added = self.vector_store.add_documents(
                settings.chroma_collection_earnings, doc_chunks
            )
            stats["turns_processed"] = len(transcript.prepared_remarks) + len(transcript.q_and_a)
            stats["chunks_created"] = added

        except Exception as e:
            stats["errors"].append(f"Failed to process earnings call: {e}")
            logger.error(f"Error ingesting earnings call: {e}")

        return stats

    async def ingest_news(
        self,
        ticker: str,
        days_back: int = 30,
        max_articles: int = 50,
    ) -> dict:
        """Ingest financial news for a company into the vector store."""
        stats = {"articles_fetched": 0, "chunks_created": 0, "errors": []}

        try:
            articles = await self.news_ingester.fetch_company_news(
                ticker=ticker, days_back=days_back, max_articles=max_articles
            )
            stats["articles_fetched"] = len(articles)

            doc_chunks = []
            for i, article in enumerate(articles):
                # Combine title + summary for embedding
                text = f"{article.title}\n\n{article.summary}"
                cleaned = clean_financial_text(text)

                doc_chunks.append(
                    DocumentChunk(
                        content=cleaned,
                        metadata={
                            "source": "financial_news",
                            "ticker": ticker,
                            "title": article.title,
                            "news_source": article.source,
                            "published_at": article.published_at,
                            "sentiment": article.sentiment or "unknown",
                            "url": article.url,
                            "category": article.category,
                        },
                        doc_id=f"news_{ticker}_{i}_{datetime.now().strftime('%Y%m%d')}",
                    )
                )

            added = self.vector_store.add_documents(
                settings.chroma_collection_news, doc_chunks
            )
            stats["chunks_created"] = added

            # Save articles to disk
            self.news_ingester.save_articles(articles, label=ticker)

        except Exception as e:
            stats["errors"].append(f"Failed to ingest news: {e}")
            logger.error(f"Error ingesting news: {e}")

        return stats

    async def ingest_full_company(
        self,
        ticker: str,
        sec_form_types: list[str] = None,
        sec_count: int = 3,
        news_days_back: int = 30,
    ) -> dict:
        """Full ingestion pipeline for a company — SEC filings + news."""
        if sec_form_types is None:
            sec_form_types = ["10-K", "10-Q"]

        results = {
            "ticker": ticker,
            "sec_filings": {},
            "news": {},
            "started_at": datetime.utcnow().isoformat(),
        }

        # Ingest SEC filings
        for form_type in sec_form_types:
            sec_stats = await self.ingest_sec_filing(
                ticker=ticker, form_type=form_type, count=sec_count
            )
            results["sec_filings"][form_type] = sec_stats

        # Ingest news
        results["news"] = await self.ingest_news(
            ticker=ticker, days_back=news_days_back
        )

        results["completed_at"] = datetime.utcnow().isoformat()
        logger.info(f"Full ingestion complete for {ticker}: {results}")
        return results

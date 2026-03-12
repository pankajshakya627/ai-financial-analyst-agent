"""
Financial news ingestion from multiple sources.
Supports Finnhub, Alpha Vantage News, and RSS feeds.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from urllib.parse import quote

import httpx
import feedparser

from config.settings import settings

logger = logging.getLogger(__name__)


# RSS feeds for financial news (free, no API key required)
RSS_FEEDS = {
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "cnbc_top": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147",
    "cnbc_tech": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000113",
    "investing_technology": "https://www.investing.com/rss/technology.rss",
    "business_insider": "https://www.businessinsider.com/rss",
}

# Common company name mappings for ticker symbols
COMPANY_NAMES = {
    "AAPL": ["Apple", "Apple Inc", "AAPL"],
    "MSFT": ["Microsoft", "Microsoft Corp", "MSFT"],
    "GOOGL": ["Google", "Alphabet", "GOOGL"],
    "GOOG": ["Google", "Alphabet", "GOOG"],
    "AMZN": ["Amazon", "Amazon.com", "AMZN"],
    "TSLA": ["Tesla", "Tesla Inc", "TSLA"],
    "META": ["Meta", "Facebook", "META"],
    "NVDA": ["Nvidia", "NVIDIA", "NVDA"],
    "JPM": ["JPMorgan", "J.P. Morgan", "JPM"],
    "BAC": ["Bank of America", "BofA", "BAC"],
    "WFC": ["Wells Fargo", "WFC"],
    "AMD": ["AMD", "Advanced Micro Devices"],
    "INTC": ["Intel", "INTC"],
    "NFLX": ["Netflix", "NFLX"],
    "DIS": ["Disney", "Walt Disney", "DIS"],
    "COIN": ["Coinbase", "COIN"],
    "PYPL": ["PayPal", "PYPL"],
    "SQ": ["Block", "Square", "SQ"],
    "UBER": ["Uber", "UBER"],
    "LYFT": ["Lyft", "LYFT"],
}


@dataclass
class NewsArticle:
    """A financial news article."""

    title: str
    summary: str
    source: str
    url: str
    published_at: str
    tickers: list[str] = field(default_factory=list)
    sentiment: Optional[str] = None  # positive, negative, neutral
    category: str = ""  # earnings, M&A, regulatory, etc.
    full_text: str = ""


class FinancialNewsIngester:
    """
    Multi-source financial news aggregator.
    Fetches from Finnhub, Alpha Vantage, and financial RSS feeds.
    """

    def __init__(self):
        self.finnhub_key = settings.finnhub_api_key
        self.alpha_vantage_key = settings.alpha_vantage_api_key
        self.news_dir = Path(settings.data_dir) / "news"
        self.news_dir.mkdir(parents=True, exist_ok=True)
        self.use_rss_fallback = True  # Enable RSS feeds as fallback

    async def fetch_company_news(
        self,
        ticker: str,
        days_back: int = 30,
        max_articles: int = 50,
    ) -> list[NewsArticle]:
        """Fetch news for a specific company from all available sources."""
        all_articles = []

        tasks = []
        if self.finnhub_key:
            tasks.append(self._fetch_finnhub_news(ticker, days_back))
        if self.alpha_vantage_key:
            tasks.append(self._fetch_alpha_vantage_news(ticker, max_articles))
        
        # Always add RSS feeds as fallback (or primary if no API keys)
        if self.use_rss_fallback or (not self.finnhub_key and not self.alpha_vantage_key):
            tasks.append(self._fetch_rss_news(ticker, days_back, max_articles))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"News fetch error: {result}")
            else:
                all_articles.extend(result)

        # Deduplicate by title similarity
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title_key = re.sub(r"[^\w]", "", article.title.lower())[:50]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)

        # Sort by publication date (newest first)
        unique_articles.sort(key=lambda a: a.published_at, reverse=True)
        return unique_articles[:max_articles]

    async def _fetch_finnhub_news(self, ticker: str, days_back: int) -> list[NewsArticle]:
        """Fetch news from Finnhub API."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        url = (
            f"https://finnhub.io/api/v1/company-news?"
            f"symbol={ticker}"
            f"&from={start_date.strftime('%Y-%m-%d')}"
            f"&to={end_date.strftime('%Y-%m-%d')}"
            f"&token={self.finnhub_key}"
        )

        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

        articles = []
        for item in data:
            articles.append(
                NewsArticle(
                    title=item.get("headline", ""),
                    summary=item.get("summary", ""),
                    source=item.get("source", "finnhub"),
                    url=item.get("url", ""),
                    published_at=datetime.fromtimestamp(
                        item.get("datetime", 0)
                    ).isoformat(),
                    tickers=[ticker],
                    category=item.get("category", ""),
                )
            )

        logger.info(f"Fetched {len(articles)} articles from Finnhub for {ticker}")
        return articles

    async def _fetch_alpha_vantage_news(
        self, ticker: str, limit: int = 50
    ) -> list[NewsArticle]:
        """Fetch news from Alpha Vantage News Sentiment API."""
        url = (
            f"https://www.alphavantage.co/query?"
            f"function=NEWS_SENTIMENT"
            f"&tickers={ticker}"
            f"&limit={limit}"
            f"&apikey={self.alpha_vantage_key}"
        )

        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

        articles = []
        for item in data.get("feed", []):
            # Determine sentiment from Alpha Vantage score
            sentiment_score = float(item.get("overall_sentiment_score", 0))
            if sentiment_score > 0.15:
                sentiment = "positive"
            elif sentiment_score < -0.15:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            # Extract related tickers
            tickers = [
                t["ticker"]
                for t in item.get("ticker_sentiment", [])
            ]

            articles.append(
                NewsArticle(
                    title=item.get("title", ""),
                    summary=item.get("summary", ""),
                    source=item.get("source", "alpha_vantage"),
                    url=item.get("url", ""),
                    published_at=item.get("time_published", ""),
                    tickers=tickers or [ticker],
                    sentiment=sentiment,
                    category=item.get("category_within_source", ""),
                )
            )

        logger.info(f"Fetched {len(articles)} articles from Alpha Vantage for {ticker}")
        return articles

    async def _fetch_rss_news(
        self,
        ticker: str,
        days_back: int = 30,
        max_articles: int = 50,
    ) -> list[NewsArticle]:
        """
        Fetch news from RSS feeds.
        This is a fallback method when API keys are not available.
        """
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Get company name variations for matching
        search_terms = COMPANY_NAMES.get(ticker.upper(), [ticker.upper()])
        
        # Fetch from general financial RSS feeds
        for feed_name, feed_url in RSS_FEEDS.items():
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(feed_url, timeout=15, follow_redirects=True)
                    resp.raise_for_status()
                    
                feed = feedparser.parse(resp.text)
                
                for entry in feed.entries[:max_articles // len(RSS_FEEDS) + 1]:
                    # Parse publication date
                    published_at = self._parse_rss_date(entry)
                    
                    # Skip old articles
                    if published_at and published_at < cutoff_date:
                        continue
                    
                    # Check if article mentions the company
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    content = entry.get("content", [{}])[0].get("value", "") if entry.get("content") else ""
                    
                    # Look for any of the search terms in title or content
                    text_to_search = f"{title} {summary} {content}"
                    if not any(term in text_to_search for term in search_terms):
                        continue
                    
                    # Extract source
                    source = feed.feed.get("title", feed_name)
                    
                    articles.append(
                        NewsArticle(
                            title=title,
                            summary=summary[:500] if summary else "",
                            source=source,
                            url=entry.get("link", ""),
                            published_at=published_at.isoformat() if published_at else datetime.now().isoformat(),
                            tickers=[ticker],
                            category="general",
                        )
                    )
                    
                    if len(articles) >= max_articles:
                        break
                        
            except Exception as e:
                logger.warning(f"RSS feed {feed_name} failed: {e}")
            
            if len(articles) >= max_articles:
                break
        
        # If no ticker-specific articles found, fetch general market news
        if len(articles) < 5:
            logger.info(f"No ticker-specific news found for {ticker}, fetching general market news")
            for feed_name, feed_url in RSS_FEEDS.items():
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(feed_url, timeout=15, follow_redirects=True)
                        resp.raise_for_status()
                        
                    feed = feedparser.parse(resp.text)
                    
                    for entry in feed.entries[:10]:
                        published_at = self._parse_rss_date(entry)
                        
                        articles.append(
                            NewsArticle(
                                title=entry.get("title", ""),
                                summary=entry.get("summary", "")[:500] if entry.get("summary") else "",
                                source=feed.feed.get("title", feed_name),
                                url=entry.get("link", ""),
                                published_at=published_at.isoformat() if published_at else datetime.now().isoformat(),
                                tickers=[ticker],
                                category="market_news",
                            )
                        )
                        
                        if len(articles) >= max_articles:
                            break
                            
                except Exception as e:
                    logger.warning(f"RSS feed {feed_name} failed: {e}")
                
                if len(articles) >= max_articles:
                    break
        
        logger.info(f"Fetched {len(articles)} articles from RSS feeds for {ticker}")
        return articles

    def _parse_rss_date(self, entry: dict) -> Optional[datetime]:
        """Parse publication date from RSS entry."""
        date_fields = [
            "published_parsed",
            "updated_parsed",
            "created_parsed",
        ]
        
        for field in date_fields:
            parsed = entry.get(field)
            if parsed:
                try:
                    return datetime(*parsed[:6])
                except (TypeError, ValueError):
                    pass
        
        # Try string parsing
        for field in ["published", "updated", "created"]:
            date_str = entry.get(field)
            if date_str:
                try:
                    # Try common RSS date formats
                    for fmt in [
                        "%a, %d %b %Y %H:%M:%S %Z",
                        "%a, %d %b %Y %H:%M:%S %z",
                        "%Y-%m-%dT%H:%M:%S%z",
                        "%Y-%m-%d %H:%M:%S",
                    ]:
                        try:
                            return datetime.strptime(date_str, fmt)
                        except ValueError:
                            continue
                except Exception:
                    pass
        
        return None

    async def fetch_market_news(
        self, category: str = "general", max_articles: int = 30
    ) -> list[NewsArticle]:
        """Fetch general market news (not company-specific)."""
        articles = []

        if self.finnhub_key:
            url = (
                f"https://finnhub.io/api/v1/news?"
                f"category={category}"
                f"&token={self.finnhub_key}"
            )
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=15)
                resp.raise_for_status()
                data = resp.json()

            for item in data[:max_articles]:
                articles.append(
                    NewsArticle(
                        title=item.get("headline", ""),
                        summary=item.get("summary", ""),
                        source=item.get("source", "finnhub"),
                        url=item.get("url", ""),
                        published_at=datetime.fromtimestamp(
                            item.get("datetime", 0)
                        ).isoformat(),
                        category=item.get("category", category),
                    )
                )

        # Fallback to RSS feeds if no API key or no articles
        if not articles and self.use_rss_fallback:
            return await self._fetch_rss_news("MARKET", 7, max_articles)

        return articles

    def save_articles(self, articles: list[NewsArticle], label: str = "batch") -> Path:
        """Save a batch of articles to local storage."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"news_{label}_{timestamp}.json"
        filepath = self.news_dir / filename

        data = [
            {
                "title": a.title,
                "summary": a.summary,
                "source": a.source,
                "url": a.url,
                "published_at": a.published_at,
                "tickers": a.tickers,
                "sentiment": a.sentiment,
                "category": a.category,
            }
            for a in articles
        ]

        filepath.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved {len(articles)} articles to {filepath}")
        return filepath

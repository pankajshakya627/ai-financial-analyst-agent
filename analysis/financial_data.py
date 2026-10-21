"""
Financial data API client supporting multiple providers:
- Financial Modeling Prep (FMP)
- Alpha Vantage
- Polygon.io
- Finnhub

Provides a unified interface for:
- Income statements, balance sheets, cash flow statements
- Stock prices (historical + real-time)
- Company profiles and key metrics
- Analyst estimates and recommendations
"""

import asyncio
import logging
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

import httpx

from config.settings import settings
from ingestion.sec_edgar import SECEdgarClient
from ingestion.sec_xbrl import facts_to_statements

logger = logging.getLogger(__name__)


@dataclass
class FinancialStatement:
    """Parsed financial statement data."""

    ticker: str
    period: str  # "annual" or "quarterly"
    date: str  # filing date
    data: dict  # key-value pairs of line items
    statement_type: str  # income_statement, balance_sheet, cash_flow


@dataclass
class StockPrice:
    """Stock price data point."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None


@dataclass
class CompanyProfile:
    """Company overview data."""

    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: float
    description: str
    exchange: str
    ceo: str = ""
    employees: int = 0
    website: str = ""
    country: str = ""


@dataclass
class AnalystEstimate:
    """Analyst consensus estimates."""

    ticker: str
    period: str
    revenue_estimate: Optional[float] = None
    eps_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    eps_actual: Optional[float] = None
    num_analysts: int = 0
    recommendation: str = ""  # buy, hold, sell


class FinancialDataClient:
    """
    Unified client for financial data from multiple API providers.
    Automatically uses whichever API keys are configured.
    """

    def __init__(self):
        self.fmp_key = settings.fmp_api_key
        self.alpha_vantage_key = settings.alpha_vantage_api_key
        self.polygon_key = settings.polygon_api_key
        self.finnhub_key = settings.finnhub_api_key

        # Determine primary provider based on available keys
        if self.fmp_key:
            self.primary = "fmp"
        elif self.alpha_vantage_key:
            self.primary = "alpha_vantage"
        elif self.polygon_key:
            self.primary = "polygon"
        else:
            self.primary = "sec_xbrl"
            logger.info("No financial data API keys configured; using SEC XBRL filings data")
        self._sec_client: Optional[SECEdgarClient] = None
        self._facts_cache: dict[str, tuple[float, dict]] = {}

    async def _sec_facts(self, ticker: str) -> dict:
        """SEC company facts, cached for an hour per ticker."""
        import time

        key = ticker.upper()
        cached = self._facts_cache.get(key)
        if cached and time.time() - cached[0] < 3600:
            return cached[1]
        self._sec_client = self._sec_client or SECEdgarClient()
        facts = await self._sec_client.get_company_facts(ticker)
        self._facts_cache[key] = (time.time(), facts)
        return facts

    async def _sec_statements(
        self, ticker: str, statement_type: str, period: str, limit: int
    ) -> list[FinancialStatement]:
        if period != "annual":
            raise ValueError("SEC fallback supports annual statements only; set a data API key for quarterly")
        facts = await self._sec_facts(ticker)
        rows = facts_to_statements(facts, limit=limit)[statement_type]
        return [
            FinancialStatement(
                ticker=ticker,
                period=period,
                date=r["date"],
                data=r["data"],
                statement_type=statement_type,
            )
            for r in rows
        ]

    async def get_income_statement(
        self,
        ticker: str,
        period: str = "annual",
        limit: int = 5,
    ) -> list[FinancialStatement]:
        """Fetch income statement data."""
        if self.fmp_key:
            return await self._fmp_income_statement(ticker, period, limit)
        elif self.alpha_vantage_key:
            return await self._av_income_statement(ticker, period, limit)
        return await self._sec_statements(ticker, "income_statement", period, limit)

    async def get_balance_sheet(
        self,
        ticker: str,
        period: str = "annual",
        limit: int = 5,
    ) -> list[FinancialStatement]:
        """Fetch balance sheet data."""
        if self.fmp_key:
            return await self._fmp_balance_sheet(ticker, period, limit)
        elif self.alpha_vantage_key:
            return await self._av_balance_sheet(ticker, period, limit)
        return await self._sec_statements(ticker, "balance_sheet", period, limit)

    async def get_cash_flow(
        self,
        ticker: str,
        period: str = "annual",
        limit: int = 5,
    ) -> list[FinancialStatement]:
        """Fetch cash flow statement data."""
        if self.fmp_key:
            return await self._fmp_cash_flow(ticker, period, limit)
        elif self.alpha_vantage_key:
            return await self._av_cash_flow(ticker, period, limit)
        return await self._sec_statements(ticker, "cash_flow", period, limit)

    async def get_stock_price(
        self,
        ticker: str,
        period: str = "1Y",
    ) -> list[StockPrice]:
        """Fetch historical stock prices."""
        if self.fmp_key:
            return await self._fmp_stock_price(ticker)
        elif self.polygon_key:
            return await self._polygon_stock_price(ticker, period)
        elif self.alpha_vantage_key:
            return await self._av_stock_price(ticker)
        raise ValueError("No financial data API key configured")

    async def get_company_profile(self, ticker: str) -> CompanyProfile:
        """Fetch company profile/overview."""
        if self.fmp_key:
            return await self._fmp_profile(ticker)
        elif self.alpha_vantage_key:
            return await self._av_profile(ticker)
        facts = await self._sec_facts(ticker)
        return CompanyProfile(
            ticker=ticker,
            name=facts.get("entityName", ticker),
            sector="",
            industry="",
            market_cap=0,
            description="",
            exchange="",
        )

    async def get_key_metrics(self, ticker: str, limit: int = 5) -> list[dict]:
        """Fetch key financial metrics (PE, EV/EBITDA, ROE, etc.)."""
        if self.fmp_key:
            url = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?limit={limit}&apikey={self.fmp_key}"
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=15)
                resp.raise_for_status()
                return resp.json()
        return []

    async def get_analyst_estimates(self, ticker: str) -> list[AnalystEstimate]:
        """Fetch analyst consensus estimates."""
        if self.fmp_key:
            return await self._fmp_estimates(ticker)
        elif self.finnhub_key:
            return await self._finnhub_estimates(ticker)
        return []

    async def get_financial_ratios(self, ticker: str, limit: int = 5) -> list[dict]:
        """Fetch pre-calculated financial ratios."""
        if self.fmp_key:
            url = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?limit={limit}&apikey={self.fmp_key}"
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=15)
                resp.raise_for_status()
                return resp.json()
        return []

    async def get_full_financials(self, ticker: str) -> dict:
        """
        Fetch all financial data for a company in parallel.
        Returns a comprehensive financial data package.
        """
        tasks = {
            "profile": self.get_company_profile(ticker),
            "income_statement": self.get_income_statement(ticker),
            "balance_sheet": self.get_balance_sheet(ticker),
            "cash_flow": self.get_cash_flow(ticker),
            "stock_price": self.get_stock_price(ticker),
            "key_metrics": self.get_key_metrics(ticker),
            "ratios": self.get_financial_ratios(ticker),
        }

        results = {}
        gathered = await asyncio.gather(
            *tasks.values(), return_exceptions=True
        )

        for key, result in zip(tasks.keys(), gathered):
            if isinstance(result, Exception):
                logger.warning(f"Failed to fetch {key} for {ticker}: {result}")
                results[key] = None
            else:
                results[key] = result

        return results

    # --- Financial Modeling Prep (FMP) implementations ---

    async def _fmp_income_statement(
        self, ticker: str, period: str, limit: int
    ) -> list[FinancialStatement]:
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period={period}&limit={limit}&apikey={self.fmp_key}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

        return [
            FinancialStatement(
                ticker=ticker,
                period=period,
                date=item.get("date", ""),
                data=item,
                statement_type="income_statement",
            )
            for item in data
        ]

    async def _fmp_balance_sheet(
        self, ticker: str, period: str, limit: int
    ) -> list[FinancialStatement]:
        url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?period={period}&limit={limit}&apikey={self.fmp_key}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

        return [
            FinancialStatement(
                ticker=ticker,
                period=period,
                date=item.get("date", ""),
                data=item,
                statement_type="balance_sheet",
            )
            for item in data
        ]

    async def _fmp_cash_flow(
        self, ticker: str, period: str, limit: int
    ) -> list[FinancialStatement]:
        url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?period={period}&limit={limit}&apikey={self.fmp_key}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

        return [
            FinancialStatement(
                ticker=ticker,
                period=period,
                date=item.get("date", ""),
                data=item,
                statement_type="cash_flow",
            )
            for item in data
        ]

    async def _fmp_stock_price(self, ticker: str) -> list[StockPrice]:
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={self.fmp_key}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

        historical = data.get("historical", [])
        return [
            StockPrice(
                date=item["date"],
                open=item["open"],
                high=item["high"],
                low=item["low"],
                close=item["close"],
                volume=item["volume"],
                adj_close=item.get("adjClose"),
            )
            for item in historical[:365]
        ]

    async def _fmp_profile(self, ticker: str) -> CompanyProfile:
        url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={self.fmp_key}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

        if not data:
            raise ValueError(f"No profile found for {ticker}")

        item = data[0]
        return CompanyProfile(
            ticker=ticker,
            name=item.get("companyName", ""),
            sector=item.get("sector", ""),
            industry=item.get("industry", ""),
            market_cap=item.get("mktCap", 0),
            description=item.get("description", ""),
            exchange=item.get("exchangeShortName", ""),
            ceo=item.get("ceo", ""),
            employees=item.get("fullTimeEmployees", 0),
            website=item.get("website", ""),
            country=item.get("country", ""),
        )

    async def _fmp_estimates(self, ticker: str) -> list[AnalystEstimate]:
        url = f"https://financialmodelingprep.com/api/v3/analyst-estimates/{ticker}?apikey={self.fmp_key}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

        return [
            AnalystEstimate(
                ticker=ticker,
                period=item.get("date", ""),
                revenue_estimate=item.get("estimatedRevenueAvg"),
                eps_estimate=item.get("estimatedEpsAvg"),
                num_analysts=item.get("numberAnalystEstimatedRevenue", 0),
            )
            for item in data[:8]
        ]

    # --- Alpha Vantage implementations ---

    # Alpha Vantage field name -> the FMP-style name the ratio code expects
    _AV_FIELDS = {
        "income_statement": {
            "totalRevenue": "revenue", "costOfRevenue": "costOfRevenue",
            "grossProfit": "grossProfit", "operatingIncome": "operatingIncome",
            "netIncome": "netIncome", "ebitda": "ebitda", "interestExpense": "interestExpense",
        },
        "balance_sheet": {
            "totalAssets": "totalAssets", "totalCurrentAssets": "totalCurrentAssets",
            "totalLiabilities": "totalLiabilities",
            "totalCurrentLiabilities": "totalCurrentLiabilities",
            "totalShareholderEquity": "totalStockholdersEquity",
            "cashAndCashEquivalentsAtCarryingValue": "cashAndCashEquivalents",
            "inventory": "inventory", "currentNetReceivables": "netReceivables",
            "currentAccountsPayable": "accountPayables",
            "longTermDebt": "longTermDebt",
            "shortTermDebt": "shortTermDebt", "currentDebt": "shortTermDebt",
        },
        "cash_flow": {
            "operatingCashflow": "operatingCashFlow",
            "capitalExpenditures": "capitalExpenditure",
        },
    }
    _AV_FUNCTIONS = {
        "income_statement": "INCOME_STATEMENT",
        "balance_sheet": "BALANCE_SHEET",
        "cash_flow": "CASH_FLOW",
    }

    @classmethod
    def _normalize_av(cls, item: dict, statement_type: str) -> dict:
        """Map an Alpha Vantage report to FMP-style numeric fields."""
        out: dict = {"date": item.get("fiscalDateEnding", "")}
        for src, dst in cls._AV_FIELDS[statement_type].items():
            raw = item.get(src)
            if raw in (None, "None", ""):
                continue
            try:
                out.setdefault(dst, float(raw))
            except (TypeError, ValueError):
                continue
        if statement_type == "balance_sheet" and ("longTermDebt" in out or "shortTermDebt" in out):
            out["totalDebt"] = out.get("longTermDebt", 0.0) + out.get("shortTermDebt", 0.0)
        if statement_type == "cash_flow":
            if "capitalExpenditure" in out:
                out["capitalExpenditure"] = -abs(out["capitalExpenditure"])
            if "operatingCashFlow" in out and "capitalExpenditure" in out:
                out["freeCashFlow"] = out["operatingCashFlow"] + out["capitalExpenditure"]
        return out

    async def _av_statement(
        self, ticker: str, statement_type: str, period: str, limit: int
    ) -> list[FinancialStatement]:
        url = (
            f"https://www.alphavantage.co/query?function={self._AV_FUNCTIONS[statement_type]}"
            f"&symbol={ticker}&apikey={self.alpha_vantage_key}"
        )
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

        # Alpha Vantage reports rate limits / errors with HTTP 200
        for notice in ("Note", "Information", "Error Message"):
            if notice in data and not data.get("annualReports"):
                raise ValueError(f"Alpha Vantage: {data[notice]}")

        key = "annualReports" if period == "annual" else "quarterlyReports"
        return [
            FinancialStatement(
                ticker=ticker,
                period=period,
                date=item.get("fiscalDateEnding", ""),
                data=self._normalize_av(item, statement_type),
                statement_type=statement_type,
            )
            for item in data.get(key, [])[:limit]
        ]

    async def _av_income_statement(self, ticker: str, period: str, limit: int = 5):
        return await self._av_statement(ticker, "income_statement", period, limit)

    async def _av_balance_sheet(self, ticker: str, period: str, limit: int = 5):
        return await self._av_statement(ticker, "balance_sheet", period, limit)

    async def _av_cash_flow(self, ticker: str, period: str, limit: int = 5):
        return await self._av_statement(ticker, "cash_flow", period, limit)

    async def _av_stock_price(self, ticker: str) -> list[StockPrice]:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={self.alpha_vantage_key}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

        ts = data.get("Time Series (Daily)", {})
        prices = []
        for date_str, values in sorted(ts.items(), reverse=True)[:365]:
            prices.append(
                StockPrice(
                    date=date_str,
                    open=float(values["1. open"]),
                    high=float(values["2. high"]),
                    low=float(values["3. low"]),
                    close=float(values["4. close"]),
                    volume=int(values["6. volume"]),
                    adj_close=float(values["5. adjusted close"]),
                )
            )
        return prices

    async def _av_profile(self, ticker: str) -> CompanyProfile:
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={self.alpha_vantage_key}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

        return CompanyProfile(
            ticker=ticker,
            name=data.get("Name", ""),
            sector=data.get("Sector", ""),
            industry=data.get("Industry", ""),
            market_cap=float(data.get("MarketCapitalization", 0)),
            description=data.get("Description", ""),
            exchange=data.get("Exchange", ""),
            country=data.get("Country", ""),
        )

    # --- Polygon.io implementations ---

    async def _polygon_stock_price(
        self, ticker: str, period: str
    ) -> list[StockPrice]:
        from datetime import timedelta

        end_date = datetime.now()
        period_days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "5Y": 1825}
        days = period_days.get(period, 365)
        start_date = end_date - timedelta(days=days)

        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
            f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            f"?adjusted=true&sort=desc"
        )

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                url, timeout=15, headers={"Authorization": f"Bearer {self.polygon_key}"}
            )
            resp.raise_for_status()
            data = resp.json()

        return [
            StockPrice(
                date=datetime.fromtimestamp(item["t"] / 1000).strftime("%Y-%m-%d"),
                open=item["o"],
                high=item["h"],
                low=item["l"],
                close=item["c"],
                volume=item["v"],
            )
            for item in data.get("results", [])
        ]

    # --- Finnhub implementations ---

    async def _finnhub_estimates(self, ticker: str) -> list[AnalystEstimate]:
        url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                url, timeout=15, headers={"X-Finnhub-Token": self.finnhub_key}
            )
            resp.raise_for_status()
            data = resp.json()

        estimates = []
        for item in data[:8]:
            total = item.get("buy", 0) + item.get("hold", 0) + item.get("sell", 0)
            if item.get("buy", 0) > item.get("sell", 0):
                rec = "buy"
            elif item.get("sell", 0) > item.get("buy", 0):
                rec = "sell"
            else:
                rec = "hold"

            estimates.append(
                AnalystEstimate(
                    ticker=ticker,
                    period=item.get("period", ""),
                    num_analysts=total,
                    recommendation=rec,
                )
            )
        return estimates

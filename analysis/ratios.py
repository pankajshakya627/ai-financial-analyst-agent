"""
Financial ratio calculator with GAAP-standard computations.
Covers profitability, liquidity, leverage, efficiency, and valuation ratios.
"""

import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RatioResult:
    """A calculated financial ratio."""

    name: str
    value: Optional[float]
    category: str  # profitability, liquidity, leverage, efficiency, valuation
    interpretation: str
    formula: str
    benchmark: Optional[str] = None


class FinancialRatioCalculator:
    """
    Calculates financial ratios from statement data.
    Handles missing data gracefully with None returns.
    """

    @staticmethod
    def safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
        """Safely divide two numbers, returning None if division is impossible."""
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator

    def calculate_all_ratios(
        self,
        income: dict,
        balance: dict,
        cash_flow: dict,
        market_cap: Optional[float] = None,
        stock_price: Optional[float] = None,
        shares_outstanding: Optional[float] = None,
    ) -> list[RatioResult]:
        """Calculate all ratios from financial statement data."""
        ratios = []

        # Profitability
        ratios.extend(self._profitability_ratios(income, balance))

        # Liquidity
        ratios.extend(self._liquidity_ratios(balance))

        # Leverage
        ratios.extend(self._leverage_ratios(income, balance))

        # Efficiency
        ratios.extend(self._efficiency_ratios(income, balance))

        # Valuation (requires market data)
        if market_cap or stock_price:
            ratios.extend(
                self._valuation_ratios(
                    income, balance, cash_flow, market_cap, stock_price, shares_outstanding
                )
            )

        # Cash flow
        ratios.extend(self._cash_flow_ratios(income, cash_flow, balance))

        return [r for r in ratios if r.value is not None]

    def _profitability_ratios(self, income: dict, balance: dict) -> list[RatioResult]:
        """Profitability ratios."""
        revenue = self._get_num(income, "revenue", "totalRevenue")
        gross_profit = self._get_num(income, "grossProfit")
        operating_income = self._get_num(income, "operatingIncome", "operatingExpenses")
        net_income = self._get_num(income, "netIncome", "netIncomeApplicableToCommonShares")
        total_assets = self._get_num(balance, "totalAssets")
        total_equity = self._get_num(balance, "totalStockholdersEquity", "totalShareholderEquity")
        ebitda = self._get_num(income, "ebitda")

        ratios = [
            RatioResult(
                name="Gross Margin",
                value=self.safe_divide(gross_profit, revenue),
                category="profitability",
                interpretation="Percentage of revenue retained after cost of goods sold",
                formula="Gross Profit / Revenue",
                benchmark="Varies by industry; >40% is strong for tech, >20% for manufacturing",
            ),
            RatioResult(
                name="Operating Margin",
                value=self.safe_divide(operating_income, revenue),
                category="profitability",
                interpretation="Percentage of revenue retained after operating expenses",
                formula="Operating Income / Revenue",
                benchmark=">15% is generally good; >25% is excellent",
            ),
            RatioResult(
                name="Net Profit Margin",
                value=self.safe_divide(net_income, revenue),
                category="profitability",
                interpretation="Percentage of revenue retained as net profit",
                formula="Net Income / Revenue",
                benchmark=">10% is solid; varies significantly by industry",
            ),
            RatioResult(
                name="EBITDA Margin",
                value=self.safe_divide(ebitda, revenue),
                category="profitability",
                interpretation="Earnings before interest, taxes, depreciation and amortization as % of revenue",
                formula="EBITDA / Revenue",
                benchmark=">20% is generally healthy",
            ),
            RatioResult(
                name="Return on Assets (ROA)",
                value=self.safe_divide(net_income, total_assets),
                category="profitability",
                interpretation="How effectively the company uses its assets to generate profit",
                formula="Net Income / Total Assets",
                benchmark=">5% is good; >10% is excellent",
            ),
            RatioResult(
                name="Return on Equity (ROE)",
                value=self.safe_divide(net_income, total_equity),
                category="profitability",
                interpretation="Return generated on shareholders' equity investment",
                formula="Net Income / Stockholders' Equity",
                benchmark=">15% is good; watch for leverage-driven inflation",
            ),
        ]
        return ratios

    def _liquidity_ratios(self, balance: dict) -> list[RatioResult]:
        """Liquidity ratios."""
        current_assets = self._get_num(balance, "totalCurrentAssets")
        current_liabilities = self._get_num(balance, "totalCurrentLiabilities")
        cash = self._get_num(balance, "cashAndCashEquivalents", "cashAndShortTermInvestments")
        inventory = self._get_num(balance, "inventory", "netInventory") or 0
        receivables = self._get_num(balance, "netReceivables", "currentNetReceivables") or 0

        quick_assets = None
        if current_assets is not None and inventory is not None:
            quick_assets = current_assets - inventory

        return [
            RatioResult(
                name="Current Ratio",
                value=self.safe_divide(current_assets, current_liabilities),
                category="liquidity",
                interpretation="Ability to pay short-term obligations with current assets",
                formula="Current Assets / Current Liabilities",
                benchmark="1.5-3.0 is healthy; <1 signals potential liquidity issues",
            ),
            RatioResult(
                name="Quick Ratio (Acid Test)",
                value=self.safe_divide(quick_assets, current_liabilities),
                category="liquidity",
                interpretation="Ability to meet obligations without relying on inventory",
                formula="(Current Assets - Inventory) / Current Liabilities",
                benchmark=">1.0 is generally safe",
            ),
            RatioResult(
                name="Cash Ratio",
                value=self.safe_divide(cash, current_liabilities),
                category="liquidity",
                interpretation="Ability to cover obligations with cash alone",
                formula="Cash & Equivalents / Current Liabilities",
                benchmark=">0.5 is conservative",
            ),
        ]

    def _leverage_ratios(self, income: dict, balance: dict) -> list[RatioResult]:
        """Leverage / solvency ratios."""
        total_debt = self._get_num(balance, "totalDebt", "longTermDebt") or 0
        short_term_debt = self._get_num(balance, "shortTermDebt", "currentPortionOfLongTermDebt") or 0
        total_borrowings = total_debt + short_term_debt
        total_assets = self._get_num(balance, "totalAssets")
        total_equity = self._get_num(balance, "totalStockholdersEquity", "totalShareholderEquity")
        ebitda = self._get_num(income, "ebitda")
        interest_expense = self._get_num(income, "interestExpense")
        operating_income = self._get_num(income, "operatingIncome")

        return [
            RatioResult(
                name="Debt-to-Equity",
                value=self.safe_divide(total_borrowings, total_equity),
                category="leverage",
                interpretation="Proportion of debt financing relative to equity",
                formula="Total Debt / Stockholders' Equity",
                benchmark="<1.0 for conservative; <2.0 acceptable for capital-intensive",
            ),
            RatioResult(
                name="Debt-to-Assets",
                value=self.safe_divide(total_borrowings, total_assets),
                category="leverage",
                interpretation="What proportion of assets is financed by debt",
                formula="Total Debt / Total Assets",
                benchmark="<0.5 is conservative",
            ),
            RatioResult(
                name="Debt-to-EBITDA",
                value=self.safe_divide(total_borrowings, ebitda),
                category="leverage",
                interpretation="Years of EBITDA needed to pay off all debt",
                formula="Total Debt / EBITDA",
                benchmark="<3x is healthy; >5x is highly leveraged",
            ),
            RatioResult(
                name="Interest Coverage",
                value=self.safe_divide(operating_income, interest_expense),
                category="leverage",
                interpretation="Ability to service interest payments from operating income",
                formula="Operating Income / Interest Expense",
                benchmark=">3x is safe; <1.5x signals distress",
            ),
        ]

    def _efficiency_ratios(self, income: dict, balance: dict) -> list[RatioResult]:
        """Efficiency / activity ratios."""
        revenue = self._get_num(income, "revenue", "totalRevenue")
        cogs = self._get_num(income, "costOfRevenue", "costOfGoodsSold")
        total_assets = self._get_num(balance, "totalAssets")
        receivables = self._get_num(balance, "netReceivables", "currentNetReceivables")
        inventory = self._get_num(balance, "inventory")
        payables = self._get_num(balance, "accountPayables")

        days_receivable = None
        if receivables and revenue:
            days_receivable = (receivables / revenue) * 365

        days_inventory = None
        if inventory and cogs:
            days_inventory = (inventory / cogs) * 365

        days_payable = None
        if payables and cogs:
            days_payable = (payables / cogs) * 365

        ccc = None
        if days_receivable is not None and days_inventory is not None and days_payable is not None:
            ccc = days_receivable + days_inventory - days_payable

        return [
            RatioResult(
                name="Asset Turnover",
                value=self.safe_divide(revenue, total_assets),
                category="efficiency",
                interpretation="Revenue generated per dollar of assets",
                formula="Revenue / Total Assets",
                benchmark=">1.0 for asset-light; <0.5 for asset-heavy industries",
            ),
            RatioResult(
                name="Days Sales Outstanding (DSO)",
                value=days_receivable,
                category="efficiency",
                interpretation="Average days to collect receivables",
                formula="(Accounts Receivable / Revenue) × 365",
                benchmark="<45 days is good; >60 may indicate collection issues",
            ),
            RatioResult(
                name="Days Inventory Outstanding (DIO)",
                value=days_inventory,
                category="efficiency",
                interpretation="Average days inventory is held before sale",
                formula="(Inventory / COGS) × 365",
                benchmark="Varies by industry; lower is generally better",
            ),
            RatioResult(
                name="Cash Conversion Cycle",
                value=ccc,
                category="efficiency",
                interpretation="Days between cash outflow for materials and cash inflow from sales",
                formula="DSO + DIO - DPO",
                benchmark="Lower is better; negative means supplier-financed operations",
            ),
        ]

    def _valuation_ratios(
        self,
        income: dict,
        balance: dict,
        cash_flow: dict,
        market_cap: Optional[float],
        stock_price: Optional[float],
        shares_outstanding: Optional[float],
    ) -> list[RatioResult]:
        """Valuation ratios (require market data)."""
        net_income = self._get_num(income, "netIncome")
        revenue = self._get_num(income, "revenue", "totalRevenue")
        ebitda = self._get_num(income, "ebitda")
        total_debt = self._get_num(balance, "totalDebt", "longTermDebt") or 0
        cash = self._get_num(balance, "cashAndCashEquivalents") or 0
        book_value = self._get_num(balance, "totalStockholdersEquity", "totalShareholderEquity")
        fcf = self._get_num(cash_flow, "freeCashFlow")

        ev = (market_cap + total_debt - cash) if market_cap else None
        eps = self.safe_divide(net_income, shares_outstanding)

        return [
            RatioResult(
                name="P/E Ratio",
                value=self.safe_divide(stock_price, eps) if eps else self.safe_divide(market_cap, net_income),
                category="valuation",
                interpretation="Price investors pay per dollar of earnings",
                formula="Stock Price / EPS (or Market Cap / Net Income)",
                benchmark="15-25 typical; >30 suggests growth expectations",
            ),
            RatioResult(
                name="P/S Ratio",
                value=self.safe_divide(market_cap, revenue),
                category="valuation",
                interpretation="Price relative to revenue — useful for pre-profit companies",
                formula="Market Cap / Revenue",
                benchmark="<2 is value; >10 is premium growth",
            ),
            RatioResult(
                name="EV/EBITDA",
                value=self.safe_divide(ev, ebitda),
                category="valuation",
                interpretation="Enterprise value relative to cash earnings — capital structure neutral",
                formula="(Market Cap + Debt - Cash) / EBITDA",
                benchmark="<10 is attractive; 10-15 is fair; >20 is premium",
            ),
            RatioResult(
                name="P/B Ratio",
                value=self.safe_divide(market_cap, book_value),
                category="valuation",
                interpretation="Market price relative to accounting book value",
                formula="Market Cap / Book Value of Equity",
                benchmark="<1 may be undervalued; >3 pricing intangibles/growth",
            ),
            RatioResult(
                name="Price-to-FCF",
                value=self.safe_divide(market_cap, fcf),
                category="valuation",
                interpretation="Price relative to free cash flow generation",
                formula="Market Cap / Free Cash Flow",
                benchmark="<15 is attractive; >25 is expensive",
            ),
            RatioResult(
                name="EV/Revenue",
                value=self.safe_divide(ev, revenue),
                category="valuation",
                interpretation="Enterprise value per dollar of revenue",
                formula="Enterprise Value / Revenue",
                benchmark="Context-dependent; compare to industry peers",
            ),
        ]

    def _cash_flow_ratios(
        self, income: dict, cash_flow: dict, balance: dict
    ) -> list[RatioResult]:
        """Cash flow quality ratios."""
        operating_cf = self._get_num(cash_flow, "operatingCashFlow", "totalCashFromOperatingActivities")
        net_income = self._get_num(income, "netIncome")
        capex = self._get_num(cash_flow, "capitalExpenditure", "capitalExpenditures")
        revenue = self._get_num(income, "revenue", "totalRevenue")
        total_debt = self._get_num(balance, "totalDebt", "longTermDebt") or 0

        fcf = None
        if operating_cf is not None and capex is not None:
            fcf = operating_cf - abs(capex)

        return [
            RatioResult(
                name="Operating Cash Flow / Net Income",
                value=self.safe_divide(operating_cf, net_income),
                category="cash_flow",
                interpretation="Cash earnings quality — >1 means earnings are cash-backed",
                formula="Operating Cash Flow / Net Income",
                benchmark=">1.0 indicates quality earnings; <0.8 warrants investigation",
            ),
            RatioResult(
                name="Free Cash Flow Margin",
                value=self.safe_divide(fcf, revenue),
                category="cash_flow",
                interpretation="Percentage of revenue converted to free cash flow",
                formula="FCF / Revenue",
                benchmark=">10% is strong; >20% is excellent",
            ),
            RatioResult(
                name="FCF-to-Debt",
                value=self.safe_divide(fcf, total_debt) if total_debt > 0 else None,
                category="cash_flow",
                interpretation="Ability to pay down debt from free cash flow",
                formula="FCF / Total Debt",
                benchmark=">0.2 suggests manageable debt load",
            ),
            RatioResult(
                name="CapEx / Revenue",
                value=self.safe_divide(abs(capex) if capex else None, revenue),
                category="cash_flow",
                interpretation="Capital intensity of the business",
                formula="Capital Expenditures / Revenue",
                benchmark="<5% is asset-light; >15% is capital-intensive",
            ),
        ]

    @staticmethod
    def _get_num(data: dict, *keys: str) -> Optional[float]:
        """Safely extract a numeric value from a dict, trying multiple keys."""
        for key in keys:
            val = data.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    continue
        return None

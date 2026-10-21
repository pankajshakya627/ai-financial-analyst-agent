"""
Convert SEC XBRL "company facts" into statement dictionaries that use the same
field names as the financial-data API client (FMP-style), so ratios and trends
work without any paid data-API key.
"""

from datetime import date
from typing import Optional

# output field -> us-gaap concepts in priority order (companies change tags over time)
INCOME_CONCEPTS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "costOfRevenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "grossProfit": ["GrossProfit"],
    "operatingIncome": ["OperatingIncomeLoss"],
    "netIncome": ["NetIncomeLoss", "ProfitLoss"],
    "interestExpense": ["InterestExpense", "InterestExpenseDebt"],
    "epsdiluted": ["EarningsPerShareDiluted"],
}

BALANCE_CONCEPTS = {
    "totalAssets": ["Assets"],
    "totalCurrentAssets": ["AssetsCurrent"],
    "totalLiabilities": ["Liabilities"],
    "totalCurrentLiabilities": ["LiabilitiesCurrent"],
    "totalStockholdersEquity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cashAndCashEquivalents": ["CashAndCashEquivalentsAtCarryingValue"],
    "inventory": ["InventoryNet"],
    "netReceivables": ["AccountsReceivableNetCurrent"],
    "accountPayables": ["AccountsPayableCurrent"],
    "longTermDebt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "shortTermDebt": ["DebtCurrent", "LongTermDebtCurrent"],
}

CASH_FLOW_CONCEPTS = {
    "operatingCashFlow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capitalExpenditure": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
}


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _annual_values(facts: dict, concept: str, instant: bool) -> dict[str, float]:
    """Latest-filed annual value per period end date for one concept."""
    node = facts.get("facts", {}).get("us-gaap", {}).get(concept)
    if not node:
        return {}
    best: dict[str, tuple[str, float]] = {}
    for unit_entries in node.get("units", {}).values():
        for e in unit_entries:
            if e.get("form") not in ("10-K", "10-K/A") or "end" not in e:
                continue
            if not instant:
                if "start" not in e or not 340 <= _days(e["start"], e["end"]) <= 380:
                    continue
            filed = e.get("filed", "")
            if e["end"] not in best or filed > best[e["end"]][0]:
                best[e["end"]] = (filed, float(e["val"]))
    return {end: val for end, (_, val) in best.items()}


def _collect(facts: dict, mapping: dict[str, list[str]], instant: bool) -> dict[str, dict]:
    """{period_end: {field: value}} merging concepts in priority order."""
    periods: dict[str, dict] = {}
    for out_field, concepts in mapping.items():
        for concept in concepts:
            for end, val in _annual_values(facts, concept, instant).items():
                periods.setdefault(end, {}).setdefault(out_field, val)
    return periods


def facts_to_statements(facts: dict, limit: int = 5) -> dict[str, list[dict]]:
    """Build annual income, balance and cash flow statements from company facts.

    Returns {"income_statement": [{"date", "data"}, ...], ...}, newest first.
    """
    income = _collect(facts, INCOME_CONCEPTS, instant=False)
    for data in income.values():
        if "grossProfit" not in data and "revenue" in data and "costOfRevenue" in data:
            data["grossProfit"] = data["revenue"] - data["costOfRevenue"]
    balance = _collect(facts, BALANCE_CONCEPTS, instant=True)
    for data in balance.values():
        if "longTermDebt" in data or "shortTermDebt" in data:
            data["totalDebt"] = data.get("longTermDebt", 0) + data.get("shortTermDebt", 0)
    cash_flow = _collect(facts, CASH_FLOW_CONCEPTS, instant=False)
    for data in cash_flow.values():
        if "capitalExpenditure" in data:
            data["capitalExpenditure"] = -abs(data["capitalExpenditure"])
        if "operatingCashFlow" in data and "capitalExpenditure" in data:
            data["freeCashFlow"] = data["operatingCashFlow"] + data["capitalExpenditure"]

    def newest(periods: dict[str, dict], required: Optional[str]) -> list[dict]:
        rows = [
            {"date": end, "data": {**data, "date": end}}
            for end, data in sorted(periods.items(), reverse=True)
            if required is None or required in data
        ]
        return rows[:limit]

    return {
        "income_statement": newest(income, "revenue") or newest(income, "netIncome"),
        "balance_sheet": newest(balance, "totalAssets"),
        "cash_flow": newest(cash_flow, "operatingCashFlow"),
    }

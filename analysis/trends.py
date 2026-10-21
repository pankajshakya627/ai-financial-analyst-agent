"""
Multi-period trend analysis: year-over-year growth, CAGR and margin changes
computed from several fiscal periods of statement data.
"""

from typing import Optional

# metric name -> (statement key, candidate field names)
TREND_METRICS: dict[str, tuple[str, tuple[str, ...]]] = {
    "revenue": ("income_statement", ("revenue", "totalRevenue")),
    "gross_profit": ("income_statement", ("grossProfit",)),
    "operating_income": ("income_statement", ("operatingIncome",)),
    "net_income": ("income_statement", ("netIncome",)),
    "eps": ("income_statement", ("epsdiluted", "eps")),
    "operating_cash_flow": ("cash_flow", ("operatingCashFlow",)),
    "free_cash_flow": ("cash_flow", ("freeCashFlow",)),
    "total_assets": ("balance_sheet", ("totalAssets",)),
    "total_debt": ("balance_sheet", ("totalDebt",)),
    "equity": ("balance_sheet", ("totalStockholdersEquity", "totalShareholderEquity")),
}

# margin name -> (numerator metric, denominator metric)
MARGINS = {
    "gross_margin": ("gross_profit", "revenue"),
    "operating_margin": ("operating_income", "revenue"),
    "net_margin": ("net_income", "revenue"),
    "fcf_margin": ("free_cash_flow", "revenue"),
}


def _to_float(value) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _row(statement) -> tuple[str, dict]:
    """Accept FinancialStatement objects or plain {'date', 'data'} / flat dicts."""
    if hasattr(statement, "data"):
        return getattr(statement, "date", ""), statement.data
    if isinstance(statement, dict) and "data" in statement:
        return statement.get("date", ""), statement["data"]
    data = statement if isinstance(statement, dict) else {}
    return data.get("date", ""), data


def build_series(financials: dict) -> dict[str, list[tuple[str, float]]]:
    """Return {metric: [(date, value), ...]} ordered oldest to newest."""
    series: dict[str, list[tuple[str, float]]] = {}
    for metric, (stmt_key, fields) in TREND_METRICS.items():
        points: dict[str, float] = {}
        for stmt in financials.get(stmt_key) or []:
            date, data = _row(stmt)
            for field_name in fields:
                value = _to_float(data.get(field_name))
                if value is not None:
                    points[date] = value
                    break
        if points:
            series[metric] = sorted(points.items())
    return series


def yoy_growth(points: list[tuple[str, float]]) -> list[tuple[str, Optional[float]]]:
    """Period-over-period growth; None when the prior value is not positive."""
    growth = []
    for (_, prev), (date, cur) in zip(points, points[1:]):
        growth.append((date, (cur - prev) / prev if prev > 0 else None))
    return growth


def cagr(points: list[tuple[str, float]]) -> Optional[float]:
    """Compound annual growth rate across the series (one point per year)."""
    if len(points) < 2:
        return None
    first, last = points[0][1], points[-1][1]
    if first <= 0 or last <= 0:
        return None
    return (last / first) ** (1 / (len(points) - 1)) - 1


def compute_trends(financials: dict) -> dict:
    """Compute growth, CAGR and margin trends from multi-period statements."""
    series = build_series(financials)
    metrics = {}
    for name, points in series.items():
        metrics[name] = {
            "series": [{"date": d, "value": v} for d, v in points],
            "yoy_growth": [{"date": d, "growth": g} for d, g in yoy_growth(points)],
            "cagr": cagr(points),
        }

    margins = {}
    for name, (num, den) in MARGINS.items():
        if num not in series or den not in series:
            continue
        denom = dict(series[den])
        points = [
            (date, value / denom[date]) for date, value in series[num] if denom.get(date)
        ]
        if points:
            change_bps = (points[-1][1] - points[0][1]) * 10_000 if len(points) > 1 else None
            margins[name] = {
                "series": [{"date": d, "value": v} for d, v in points],
                "change_bps": change_bps,
            }

    return {
        "periods": len(next(iter(series.values()))) if series else 0,
        "metrics": metrics,
        "margins": margins,
    }


def format_trends_for_prompt(trends: dict) -> str:
    """Render trends as compact text for an LLM prompt."""
    if not trends or not trends.get("metrics"):
        return "No multi-period trend data available."

    def pct(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{x * 100:.1f}%"

    lines = []
    for name in ("revenue", "net_income", "eps", "free_cash_flow", "operating_cash_flow"):
        m = trends["metrics"].get(name)
        if not m:
            continue
        growth = ", ".join(f"{g['date'][:4]}: {pct(g['growth'])}" for g in m["yoy_growth"])
        lines.append(f"- {name.replace('_', ' ').title()} growth YoY: {growth or 'n/a'} (CAGR {pct(m['cagr'])})")
    for name, m in trends["margins"].items():
        path = " -> ".join(f"{p['date'][:4]}: {pct(p['value'])}" for p in m["series"])
        change = "" if m["change_bps"] is None else f" ({m['change_bps']:+.0f} bps overall)"
        lines.append(f"- {name.replace('_', ' ').title()}: {path}{change}")
    return "\n".join(lines) or "No multi-period trend data available."

"""
Check numbers in LLM-written analysis against the data it was given.

A figure counts as supported when it matches (within rounding) a value from the
structured statements, calculated ratios, trends, or the retrieved context.
Unsupported figures are reported so readers can double-check them; the text is
never changed.
"""

import re
from typing import Iterable, Optional

_SCALE = {
    "trillion": 1e12, "t": 1e12,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mm": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
}

_FIGURE = re.compile(
    r"(?<![\w.\[])(?P<dollar>\$\s?)?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<unit>%|percent\b|trillion\b|billion\b|million\b|thousand\b|bn\b|mm\b|[TBMK]\b))?",
)


def _parse(match: re.Match) -> Optional[dict]:
    raw = match.group("num")
    value = float(raw.replace(",", ""))
    unit = (match.group("unit") or "").lower()
    dollar = bool(match.group("dollar"))
    is_pct = unit in ("%", "percent")

    # Skip things that are rarely financial claims
    tail = match.string[match.end(): match.end() + 3]
    if re.match(r"-[KQkq]\b", tail) or re.match(r"\d", tail):  # 10-K, 10-Q
        return None
    has_decimal = "." in raw
    if not (dollar or unit or has_decimal or value >= 1000):
        return None
    if not (dollar or unit or has_decimal) and 1900 <= value <= 2100:  # years
        return None

    decimals = len(raw.split(".")[1]) if has_decimal else 0
    scale = 1.0 if is_pct else _SCALE.get(unit, 1.0)
    scaled = value * scale
    precision = (10 ** -decimals) * scale  # size of the last printed digit
    return {
        "text": match.group(0).strip(),
        "value": scaled,
        "is_pct": is_pct,
        "precision": precision,
        "start": match.start(),
    }


def extract_figures(text: str) -> list[dict]:
    return [f for f in (_parse(m) for m in _FIGURE.finditer(text or "")) if f]


def _matches(figure: dict, references: Iterable[float]) -> bool:
    value = abs(figure["value"])
    tol = max(0.01 * value, 0.5 * figure["precision"])
    return any(abs(abs(ref) - value) <= tol for ref in references)


def collect_references(financials: dict, ratios: list, trends: dict) -> tuple[set, set]:
    """Return (absolute_values, percentage_points) the analysis may cite."""
    absolute: set[float] = set()
    pct: set[float] = set()

    for key in ("income_statement", "balance_sheet", "cash_flow"):
        for stmt in (financials or {}).get(key) or []:
            data = stmt.data if hasattr(stmt, "data") else (stmt.get("data", stmt) if isinstance(stmt, dict) else {})
            for v in data.values():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    absolute.add(float(v))
    profile = (financials or {}).get("profile")
    if profile is not None and getattr(profile, "market_cap", None):
        absolute.add(float(profile.market_cap))
    for price in (financials or {}).get("stock_price") or []:
        if hasattr(price, "close"):
            absolute.add(float(price.close))

    for r in ratios or []:
        value = getattr(r, "value", None) if not isinstance(r, dict) else r.get("value")
        if value is None:
            continue
        absolute.add(float(value))
        pct.add(float(value) * 100)  # fractions such as margins quoted as percentages

    for metric in (trends or {}).get("metrics", {}).values():
        for point in metric["series"]:
            absolute.add(float(point["value"]))
        for g in metric["yoy_growth"]:
            if g["growth"] is not None:
                pct.add(g["growth"] * 100)
        if metric["cagr"] is not None:
            pct.add(metric["cagr"] * 100)
    for margin in (trends or {}).get("margins", {}).values():
        for point in margin["series"]:
            pct.add(point["value"] * 100)
        if margin["change_bps"] is not None:
            absolute.add(abs(margin["change_bps"]))
            pct.add(margin["change_bps"] / 100)
    return absolute, pct


def verify_figures(
    analysis_text: str,
    financials: dict,
    ratios: list,
    trends: dict,
    context_text: str = "",
    max_reported: int = 20,
) -> dict:
    """Report which figures in the analysis could not be matched to the inputs."""
    absolute, pct = collect_references(financials, ratios, trends)
    for fig in extract_figures(context_text):
        (pct if fig["is_pct"] else absolute).add(fig["value"])

    figures = extract_figures(analysis_text)
    unsupported = []
    for fig in figures:
        refs = pct if fig["is_pct"] else absolute
        if _matches(fig, refs):
            continue
        start = max(0, fig["start"] - 40)
        snippet = analysis_text[start: fig["start"] + 40].replace("\n", " ").strip()
        unsupported.append({"figure": fig["text"], "context": snippet})

    return {
        "checked": len(figures),
        "supported": len(figures) - len(unsupported),
        "unsupported": unsupported[:max_reported],
        "unsupported_count": len(unsupported),
    }

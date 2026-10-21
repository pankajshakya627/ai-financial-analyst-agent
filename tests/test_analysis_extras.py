"""Tests for verification, TTM, Alpha Vantage mapping and speaker splitting."""

import pytest

from analysis.financial_data import FinancialDataClient
from analysis.trends import compute_ttm
from analysis.verification import extract_figures, verify_figures
from core.text_processing import FinancialTextSplitter


FIN = {"income_statement": [{"date": "2024-12-31", "data": {"revenue": 391_035_000_000, "netIncome": 93_736_000_000}}]}


def test_supported_and_unsupported_figures():
    text = "Revenue was $391.0 billion with net income of $93.7B, but margin hit 55.2% and EPS of $9.99."
    result = verify_figures(text, FIN, [], {})
    flagged = {u["figure"] for u in result["unsupported"]}
    assert "$391.0 billion" not in flagged and "$93.7B" not in flagged
    assert "55.2%" in flagged and "$9.99" in flagged


def test_ratio_percentages_and_context_numbers_supported():
    class R:
        value = 0.2397
    text = "Net margin was 24.0% and the filing cites 12.5% growth in services."
    result = verify_figures(text, FIN, [R()], {}, context_text="Services grew 12.5% year over year [S1]")
    assert result["unsupported"] == []


def test_ignores_years_filings_and_citations():
    figs = extract_figures("In 2024 the 10-K [S2] said Q3 results; 5 analysts agreed.")
    assert figs == []


def test_opposite_sign_figures_are_not_supported():
    loss = {"income_statement": [{"date": "2024-12-31", "data": {"netIncome": -10_000_000}}]}
    assert [u["figure"] for u in verify_figures("Net income was $10.0 million.", loss, [], {})["unsupported"]] == ["$10.0 million"]
    assert verify_figures("Net income was -$10.0 million.", loss, [], {})["unsupported"] == []

    trends = {"metrics": {"revenue": {"series": [{"date": "2024", "value": 100}],
                                       "yoy_growth": [{"year": "2024", "growth": -0.10}], "cagr": None}},
              "margins": {}}
    assert [u["figure"] for u in verify_figures("Revenue grew 10.0%.", {}, [], trends)["unsupported"]] == ["10.0%"]
    assert verify_figures("Revenue fell -10.0%.", {}, [], trends)["unsupported"] == []


def _q(date, revenue):
    return {"date": date, "data": {"revenue": revenue}}


def test_ttm_sums_four_quarters_with_growth():
    quarters = [_q(d, v) for d, v in [
        ("2024-12-31", 140), ("2024-09-30", 130), ("2024-06-30", 120), ("2024-03-31", 110),
        ("2023-12-31", 120), ("2023-09-30", 110), ("2023-06-30", 100), ("2023-03-31", 90),
    ]]
    ttm = compute_ttm({"income_statement": quarters})["revenue"]
    assert ttm["ttm"] == 500 and ttm["prior_ttm"] == 420
    assert ttm["growth"] == pytest.approx(80 / 420)


def test_ttm_requires_consecutive_quarters():
    gappy = [_q("2024-12-31", 1), _q("2024-03-31", 1), _q("2023-12-31", 1), _q("2023-03-31", 1)]
    assert compute_ttm({"income_statement": gappy}) == {}


def test_alpha_vantage_normalization():
    balance = FinancialDataClient._normalize_av(
        {"fiscalDateEnding": "2024-12-31", "totalShareholderEquity": "90", "longTermDebt": "50",
         "shortTermDebt": "10", "inventory": "None", "currentNetReceivables": "7"},
        "balance_sheet",
    )
    assert balance["totalStockholdersEquity"] == 90 and balance["totalDebt"] == 60
    assert "inventory" not in balance and balance["netReceivables"] == 7
    cash = FinancialDataClient._normalize_av(
        {"operatingCashflow": "100", "capitalExpenditures": "30"}, "cash_flow")
    assert cash["capitalExpenditure"] == -30 and cash["freeCashFlow"] == 70


def test_earnings_call_split_keeps_every_speaker():
    text = "Intro words here.\nTim Cook: First turn about iPhone sales growth.\nLuca Maestri: Second turn about margins and cash."
    splitter = FinancialTextSplitter(chunk_size=500, chunk_overlap=0)
    chunks = splitter.split_text(text, doc_type="earnings_call")
    sections = [c.section for c in chunks]
    assert "Tim Cook" in sections and "Luca Maestri" in sections and "Introduction" in sections

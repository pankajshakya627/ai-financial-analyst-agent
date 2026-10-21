"""Tests for multi-period trend analysis."""

import pytest

from analysis.trends import cagr, compute_trends, format_trends_for_prompt, yoy_growth


def _financials():
    # newest first, like the data client returns them
    return {
        "income_statement": [
            {"date": "2024-12-31", "data": {"revenue": 121, "grossProfit": 60.5, "netIncome": 24.2}},
            {"date": "2023-12-31", "data": {"revenue": 110, "grossProfit": 55, "netIncome": 22}},
            {"date": "2022-12-31", "data": {"revenue": 100, "grossProfit": 45, "netIncome": 20}},
        ],
        "cash_flow": [
            {"date": "2024-12-31", "data": {"operatingCashFlow": 30, "freeCashFlow": 25}},
            {"date": "2023-12-31", "data": {"operatingCashFlow": 28, "freeCashFlow": 22}},
        ],
    }


def test_yoy_and_cagr():
    points = [("2022", 100.0), ("2023", 110.0), ("2024", 121.0)]
    assert [g for _, g in yoy_growth(points)] == [pytest.approx(0.10), pytest.approx(0.10)]
    assert cagr(points) == pytest.approx(0.10)


def test_growth_undefined_for_non_positive_base():
    assert yoy_growth([("2022", -5.0), ("2023", 10.0)])[0][1] is None
    assert cagr([("2022", 0.0), ("2023", 10.0)]) is None


def test_compute_trends_orders_oldest_first_and_margins():
    trends = compute_trends(_financials())
    revenue = trends["metrics"]["revenue"]
    assert [p["date"] for p in revenue["series"]] == ["2022-12-31", "2023-12-31", "2024-12-31"]
    assert revenue["cagr"] == pytest.approx(0.10)
    gross = trends["margins"]["gross_margin"]
    assert gross["series"][0]["value"] == pytest.approx(0.45)
    assert gross["change_bps"] == pytest.approx(500, abs=1)


def test_missing_data_is_handled():
    assert compute_trends({})["metrics"] == {}
    assert "No multi-period" in format_trends_for_prompt(compute_trends({}))


def test_prompt_text_mentions_growth():
    text = format_trends_for_prompt(compute_trends(_financials()))
    assert "Revenue growth YoY" in text and "10.0%" in text

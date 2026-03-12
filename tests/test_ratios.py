"""Unit tests for financial ratio calculations."""

import pytest
from analysis.ratios import FinancialRatioCalculator


@pytest.fixture
def calculator():
    return FinancialRatioCalculator()


@pytest.fixture
def sample_income():
    return {
        "revenue": 100_000_000,
        "grossProfit": 60_000_000,
        "operatingIncome": 25_000_000,
        "netIncome": 18_000_000,
        "ebitda": 30_000_000,
        "costOfRevenue": 40_000_000,
        "interestExpense": 3_000_000,
    }


@pytest.fixture
def sample_balance():
    return {
        "totalAssets": 200_000_000,
        "totalCurrentAssets": 80_000_000,
        "totalCurrentLiabilities": 50_000_000,
        "cashAndCashEquivalents": 25_000_000,
        "inventory": 15_000_000,
        "netReceivables": 20_000_000,
        "totalDebt": 60_000_000,
        "totalStockholdersEquity": 90_000_000,
        "accountPayables": 12_000_000,
    }


@pytest.fixture
def sample_cash_flow():
    return {
        "operatingCashFlow": 28_000_000,
        "capitalExpenditure": -8_000_000,
        "freeCashFlow": 20_000_000,
    }


class TestSafeDivide:
    def test_normal_division(self, calculator):
        assert calculator.safe_divide(10, 5) == 2.0

    def test_divide_by_zero(self, calculator):
        assert calculator.safe_divide(10, 0) is None

    def test_none_numerator(self, calculator):
        assert calculator.safe_divide(None, 5) is None

    def test_none_denominator(self, calculator):
        assert calculator.safe_divide(10, None) is None


class TestProfitabilityRatios:
    def test_gross_margin(self, calculator, sample_income, sample_balance):
        ratios = calculator._profitability_ratios(sample_income, sample_balance)
        gross_margin = next(r for r in ratios if r.name == "Gross Margin")
        assert gross_margin.value == pytest.approx(0.6, rel=1e-2)

    def test_net_margin(self, calculator, sample_income, sample_balance):
        ratios = calculator._profitability_ratios(sample_income, sample_balance)
        net_margin = next(r for r in ratios if r.name == "Net Profit Margin")
        assert net_margin.value == pytest.approx(0.18, rel=1e-2)

    def test_roe(self, calculator, sample_income, sample_balance):
        ratios = calculator._profitability_ratios(sample_income, sample_balance)
        roe = next(r for r in ratios if r.name == "Return on Equity (ROE)")
        assert roe.value == pytest.approx(0.2, rel=1e-2)


class TestLiquidityRatios:
    def test_current_ratio(self, calculator, sample_balance):
        ratios = calculator._liquidity_ratios(sample_balance)
        current = next(r for r in ratios if r.name == "Current Ratio")
        assert current.value == pytest.approx(1.6, rel=1e-2)

    def test_quick_ratio(self, calculator, sample_balance):
        ratios = calculator._liquidity_ratios(sample_balance)
        quick = next(r for r in ratios if r.name == "Quick Ratio (Acid Test)")
        # (80M - 15M) / 50M = 1.3
        assert quick.value == pytest.approx(1.3, rel=1e-2)


class TestLeverageRatios:
    def test_debt_to_equity(self, calculator, sample_income, sample_balance):
        ratios = calculator._leverage_ratios(sample_income, sample_balance)
        dte = next(r for r in ratios if r.name == "Debt-to-Equity")
        # 60M / 90M = 0.667
        assert dte.value == pytest.approx(0.667, rel=1e-2)

    def test_interest_coverage(self, calculator, sample_income, sample_balance):
        ratios = calculator._leverage_ratios(sample_income, sample_balance)
        ic = next(r for r in ratios if r.name == "Interest Coverage")
        # 25M / 3M = 8.33
        assert ic.value == pytest.approx(8.33, rel=1e-2)


class TestCashFlowRatios:
    def test_fcf_margin(self, calculator, sample_income, sample_cash_flow, sample_balance):
        ratios = calculator._cash_flow_ratios(sample_income, sample_cash_flow, sample_balance)
        fcf_margin = next(r for r in ratios if r.name == "Free Cash Flow Margin")
        # 20M / 100M = 0.2
        assert fcf_margin.value == pytest.approx(0.2, rel=1e-2)

    def test_ocf_to_ni(self, calculator, sample_income, sample_cash_flow, sample_balance):
        ratios = calculator._cash_flow_ratios(sample_income, sample_cash_flow, sample_balance)
        ocf_ni = next(r for r in ratios if r.name == "Operating Cash Flow / Net Income")
        # 28M / 18M = 1.556
        assert ocf_ni.value == pytest.approx(1.556, rel=1e-2)


class TestAllRatios:
    def test_calculate_all(
        self, calculator, sample_income, sample_balance, sample_cash_flow
    ):
        ratios = calculator.calculate_all_ratios(
            income=sample_income,
            balance=sample_balance,
            cash_flow=sample_cash_flow,
            market_cap=500_000_000,
            stock_price=50.0,
        )
        # Should return a non-empty list of valid ratios
        assert len(ratios) > 10
        # All returned ratios should have non-None values
        for r in ratios:
            assert r.value is not None
            assert r.name
            assert r.category

    def test_handles_empty_data(self, calculator):
        ratios = calculator.calculate_all_ratios(
            income={}, balance={}, cash_flow={}
        )
        # Should return empty or only ratios with None filtered out
        assert isinstance(ratios, list)

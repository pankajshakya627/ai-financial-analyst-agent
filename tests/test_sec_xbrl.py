"""Tests for converting SEC XBRL company facts into statements."""

from ingestion.sec_xbrl import facts_to_statements


def _duration(end, start, val, form="10-K", filed="2025-02-01"):
    return {"start": start, "end": end, "val": val, "form": form, "filed": filed}


def _instant(end, val, form="10-K", filed="2025-02-01"):
    return {"end": end, "val": val, "form": form, "filed": filed}


FACTS = {
    "entityName": "Example Corp",
    "facts": {
        "us-gaap": {
            "Revenues": {"units": {"USD": [
                _duration("2024-12-31", "2024-01-01", 1000),
                _duration("2023-12-31", "2023-01-01", 900),
                # quarterly figure inside a 10-K must be ignored
                _duration("2024-12-31", "2024-10-01", 260),
                # 10-Q must be ignored
                _duration("2024-06-30", "2024-01-01", 480, form="10-Q"),
            ]}},
            "CostOfRevenue": {"units": {"USD": [_duration("2024-12-31", "2024-01-01", 400)]}},
            "NetIncomeLoss": {"units": {"USD": [_duration("2024-12-31", "2024-01-01", 150)]}},
            "Assets": {"units": {"USD": [_instant("2024-12-31", 5000), _instant("2023-12-31", 4800)]}},
            "LongTermDebtNoncurrent": {"units": {"USD": [_instant("2024-12-31", 700)]}},
            "DebtCurrent": {"units": {"USD": [_instant("2024-12-31", 100)]}},
            "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [_duration("2024-12-31", "2024-01-01", 220)]}},
            "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [_duration("2024-12-31", "2024-01-01", 50)]}},
        }
    },
}


def test_income_statement_uses_annual_10k_values_only():
    stmts = facts_to_statements(FACTS)
    income = stmts["income_statement"]
    assert [r["date"] for r in income] == ["2024-12-31", "2023-12-31"]
    assert income[0]["data"]["revenue"] == 1000
    assert income[0]["data"]["grossProfit"] == 600  # derived from revenue - cost
    assert income[0]["data"]["netIncome"] == 150


def test_balance_and_cash_flow_derived_fields():
    stmts = facts_to_statements(FACTS)
    assert stmts["balance_sheet"][0]["data"]["totalDebt"] == 800
    cf = stmts["cash_flow"][0]["data"]
    assert cf["capitalExpenditure"] == -50
    assert cf["freeCashFlow"] == 170


def test_limit_applies():
    assert len(facts_to_statements(FACTS, limit=1)["income_statement"]) == 1


def test_empty_facts():
    assert facts_to_statements({})["income_statement"] == []

"""The Yahoo field-unit conversions, tested with made-up info dicts (no internet needed)."""
import pytest

from stocklens.ingest.yahoo import map_fundamentals


def test_dividend_yield_below_one_percent_is_converted():
    """Regression: a 0.3% yield arrives as 0.31 (percent) and used to be left as 0.31 (=31%)."""
    info = {"dividendRate": 1.04, "currentPrice": 352.45, "dividendYield": 0.31}
    assert map_fundamentals("X", info)["dividend_yield"] == pytest.approx(0.00295, abs=1e-4)


def test_dividend_yield_falls_back_to_percent_field():
    assert map_fundamentals("X", {"dividendYield": 4.45})["dividend_yield"] == pytest.approx(0.0445)


def test_no_dividend_gives_none():
    assert map_fundamentals("X", {})["dividend_yield"] is None


def test_debt_to_equity_percent_becomes_ratio_and_missing_stays_none():
    assert map_fundamentals("X", {"debtToEquity": 84.286})["debt_to_equity"] == pytest.approx(0.84286)
    assert map_fundamentals("X", {})["debt_to_equity"] is None


def test_other_fields_pass_through():
    row = map_fundamentals("X", {"sector": "Technology", "trailingPE": 20.0, "priceToBook": 3.0, "returnOnEquity": 0.25})
    assert (row["sector"], row["pe"], row["pb"], row["roe"]) == ("Technology", 20.0, 3.0, 0.25)

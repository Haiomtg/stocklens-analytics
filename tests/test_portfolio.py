"""Hand-calculated examples: if these pass, the P&L maths is doing what a human would do on paper."""
import numpy as np
import pandas as pd
import pytest

from stocklens.portfolio import returns as pret
from stocklens.portfolio import tracker


def tx(rows):
    return tracker.load_transactions(pd.DataFrame(rows, columns=["date", "symbol", "type", "quantity", "price", "fees"]))


def test_average_cost_and_realised_pnl():
    t = tx([
        ("2024-01-02", "AAA", "BUY", 100, 10.0, 10.0),   # cost 1010
        ("2024-02-01", "AAA", "BUY", 100, 20.0, 10.0),   # cost 2010 -> total 3020 for 200 -> avg 15.10
        ("2024-03-01", "AAA", "SELL", 50, 30.0, 5.0),    # proceeds 1495; cost of 50 = 755 -> realised 740
    ])
    h = tracker.holdings_table(t, pd.Series({"AAA": 25.0}))
    r = h.loc["AAA"]
    assert r["quantity"] == 150
    assert r["avg_cost"] == pytest.approx(15.10)
    assert r["realised_pnl"] == pytest.approx(740.0)
    assert r["cost_basis"] == pytest.approx(150 * 15.10)
    assert r["unrealised_pnl"] == pytest.approx(150 * 25 - 150 * 15.10)


def test_dividends_counted_separately():
    t = tx([
        ("2024-01-02", "AAA", "BUY", 100, 10.0, 0.0),
        ("2024-06-01", "AAA", "DIVIDEND", 100, 0.25, 0.0),
    ])
    h = tracker.holdings_table(t, pd.Series({"AAA": 10.0}))
    assert h.loc["AAA", "dividends"] == pytest.approx(25.0)
    assert h.loc["AAA", "total_pnl"] == pytest.approx(25.0)


def test_cannot_sell_more_than_held():
    t = tx([("2024-01-02", "AAA", "BUY", 10, 10.0, 0.0), ("2024-01-03", "AAA", "SELL", 11, 10.0, 0.0)])
    with pytest.raises(ValueError, match="only"):
        tracker.build_positions(t)


def test_bad_input_rejected():
    with pytest.raises(ValueError):
        tx([("2024-01-02", "AAA", "HODL", 10, 10.0, 0.0)])
    with pytest.raises(ValueError):
        tx([("2024-01-02", "AAA", "BUY", -5, 10.0, 0.0)])


def test_fully_sold_position_resets_cost():
    t = tx([("2024-01-02", "AAA", "BUY", 10, 10.0, 0.0), ("2024-01-03", "AAA", "SELL", 10, 12.0, 0.0)])
    h = tracker.holdings_table(t, pd.Series({"AAA": 99.0}))
    assert h.loc["AAA", "quantity"] == 0 and h.loc["AAA", "market_value"] == 0
    assert h.loc["AAA", "realised_pnl"] == pytest.approx(20.0)


def test_twr_ignores_timing_of_deposits():
    """Price goes 100 -> 110 -> 121 (+10% each day). Buying more on day 2 must not change the 10% daily return."""
    idx = pd.bdate_range("2024-01-01", periods=3)
    close = pd.DataFrame({"AAA": [100.0, 110.0, 121.0]}, index=idx)
    t = tx([
        (idx[0], "AAA", "BUY", 10, 100.0, 0.0),
        (idx[1], "AAA", "BUY", 90, 110.0, 0.0),   # big top-up on day 2
    ])
    r = pret.twr_daily_returns(t, close)
    assert r.iloc[-1] == pytest.approx(0.10)       # day 3: +10%
    assert r.iloc[0] == pytest.approx(0.10)        # day 2: the day-1 shares gained 10%; the top-up isn't counted as return


def test_benchmark_comparison_beta():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2024-01-01", periods=300)
    bench = pd.Series(rng.normal(0, 0.01, 300), index=idx)
    port = 1.5 * bench
    out = pret.compare_to_benchmark(port, bench)
    assert out["beta"] == pytest.approx(1.5)

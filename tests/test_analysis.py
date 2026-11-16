import numpy as np
import pandas as pd
import pytest

from stocklens.analysis import indicators, risk, valuation


def test_sma_known_values():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = indicators.sma(s, 3)
    assert out.isna().sum() == 2 and out.iloc[-1] == pytest.approx(4.0)


def test_rsi_extremes_and_bounds(walk):
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert indicators.rsi(up).iloc[-1] == pytest.approx(100.0)
    down = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert indicators.rsi(down).iloc[-1] == pytest.approx(0.0, abs=1e-6)
    r = indicators.rsi(walk).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_macd_hist_is_line_minus_signal(walk):
    m = indicators.macd(walk).dropna()
    assert np.allclose(m["hist"], m["macd"] - m["signal"])


def test_max_drawdown_known():
    # wealth: 1 -> 2 -> 1 -> 1.5 : max drawdown = 50%
    rets = pd.Series([1.0, -0.5, 0.5])
    assert risk.max_drawdown(rets) == pytest.approx(-0.5)


def test_beta_of_leveraged_series(walk):
    bench = walk.pct_change().dropna()
    assert risk.beta(2 * bench, bench) == pytest.approx(2.0)


def test_var_and_es_ordering(walk):
    r = walk.pct_change().dropna()
    var, es = risk.historical_var(r, 0.95), risk.expected_shortfall(r, 0.95)
    assert var > 0 and es >= var  # average tail loss is at least the VaR threshold


def test_sharpe_zero_vol_does_not_crash():
    assert np.isnan(risk.sharpe_ratio(pd.Series([0.0] * 50)))


def test_concentration_warnings():
    h = pd.DataFrame({"market_value": [700.0, 200.0, 100.0], "sector": ["Tech", "Tech", "Banks"]}, index=["A", "B", "C"])
    rep = risk.concentration_report(h, max_position=0.25, max_sector=0.5)
    assert any("A is 70%" in w for w in rep["warnings"])
    assert any("Tech sector is 90%" in w for w in rep["warnings"])
    assert rep["hhi"] == pytest.approx(0.49 + 0.04 + 0.01)


def test_screener_ranks_obviously_better_stock_first():
    idx = pd.bdate_range("2023-01-01", periods=400)
    prices = pd.DataFrame({"GOOD": np.linspace(100, 200, 400), "BAD": np.linspace(100, 60, 400)}, index=idx)
    f = pd.DataFrame(
        {"sector": ["X", "X"], "pe": [10.0, 40.0], "pb": [1.0, 5.0], "roe": [0.25, 0.02], "debt_to_equity": [0.2, 2.0]},
        index=["GOOD", "BAD"],
    )
    out = valuation.score_watchlist(prices, f)
    assert out.index[0] == "GOOD" and out.loc["GOOD", "composite"] > out.loc["BAD", "composite"]


def test_peer_comparison_uses_sector_median():
    f = pd.DataFrame(
        {"sector": ["T", "T", "T", "B"], "pe": [10.0, 20.0, 30.0, 5.0], "pb": [1.0] * 4, "roe": [0.1] * 4,
         "debt_to_equity": [1.0] * 4, "dividend_yield": [0.0] * 4},
        index=["A", "B1", "C", "D"],
    )
    out = valuation.peer_comparison(f, "A")
    assert out.loc["pe", "peer_median"] == pytest.approx(25.0)  # median of 20 and 30
    assert out.attrs["scope"] == "sector peers"

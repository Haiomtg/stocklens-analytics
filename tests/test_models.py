import warnings

import numpy as np
import pandas as pd
import pytest

from stocklens.models import backtest as bt
from stocklens.models import forecasting as fc


def test_signal_cannot_earn_same_day_return():
    """A signal that is 'perfect' with hindsight (long only on up days) must NOT earn those up days,
    because we only act on it the next day. If shift(1) were missing, this would show a huge profit."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=500)
    price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))), index=idx)
    cheating = (price.pct_change() > 0).astype(float)    # peeks at today's return
    res = bt.run_backtest(price, cheating, cost_bps=0)
    assert res.metrics["total_return"] < 0.5             # without look-ahead protection this would be astronomical
    # and the held position on day t is the signal from day t-1
    assert (res.position.iloc[1:].values == cheating.shift(1).iloc[1:].values).all()


def test_costs_reduce_returns_and_scale_with_turnover(walk):
    pos = bt.ma_crossover_position(walk, 10, 50)
    free = bt.run_backtest(walk, pos, cost_bps=0)
    costly = bt.run_backtest(walk, pos, cost_bps=50)
    assert costly.metrics["total_return"] < free.metrics["total_return"]
    expected_drag = (free.returns - costly.returns).sum()
    assert expected_drag == pytest.approx(costly.turnover.sum() * 50 / 1e4)


def test_buy_and_hold_pays_one_entry_cost():
    idx = pd.bdate_range("2024-01-01", periods=5)
    price = pd.Series([100.0] * 5, index=idx)           # flat price: all return is cost
    res = bt.buy_and_hold(price, cost_bps=100)
    assert res.returns.sum() == pytest.approx(-0.01)


def test_portfolio_backtest_weights_sum_and_no_lookahead(walk):
    prices = pd.DataFrame({"A": walk, "B": walk.shift(5).bfill() * 0.9 + 1}, index=walk.index)
    w = bt.cross_sectional_momentum_weights(prices, lookback=60, top_n=1)
    assert ((w.sum(axis=1) - 1).abs() < 1e-9).where(w.sum(axis=1) > 0, True).all()
    res = bt.run_portfolio_backtest(prices, w, 10)
    assert (res.position.iloc[0] == 0).all()             # nothing held on day 1


def test_parameter_sweep_skips_invalid_pairs(walk):
    sweep = bt.parameter_sweep(walk, fast_grid=(10, 50), slow_grid=(50, 100))
    assert not ((sweep["fast"] >= sweep["slow"]).any())


def test_walk_forward_has_no_leakage():
    n, h = 2000, 21
    splits = list(fc.walk_forward_splits(n, h, initial_train=756, test_size=63))
    assert len(splits) > 5
    for tr, te in splits:
        assert tr.max() + h <= te.min()                  # last training label ends before test begins
        assert tr.min() == 0                             # expanding window
    tests = np.concatenate([te for _, te in splits])
    assert len(tests) == len(set(tests))                 # no test row used twice


def test_features_use_only_past_data(walk):
    """Changing FUTURE prices must not change today's features."""
    f1 = fc.make_features(walk)
    altered = walk.copy()
    altered.iloc[-50:] *= 3
    f2 = fc.make_features(altered)
    pd.testing.assert_frame_equal(f1.iloc[:-50], f2.iloc[:-50])


def test_random_walk_gets_honest_negative_verdict():
    """On pure noise there is nothing to learn; the framework must say so instead of finding 'signals'."""
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2012-01-01", periods=2200)
    price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 2200))), index=idx)
    out = fc.run_experiment(price, initial_train=756)
    assert {"logistic", "random_forest", "always_up", "last_month_repeats"} <= set(out["summary"].index)
    assert out["verdict"].startswith("No model clearly beat")


def test_many_stock_summary_applies_multiple_testing_correction():
    import numpy as np
    import pandas as pd
    from stocklens.models import forecasting as fc

    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2015-01-01", periods=1700)
    prices = pd.DataFrame({f"S{i}": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 1700))) for i in range(3)}, index=idx)
    seen = []
    table = fc.run_experiment_many(prices, progress=lambda i, n, s: seen.append(s))
    assert seen == ["S0", "S1", "S2"]
    assert (table["p_adj_all"] >= table["p_adj_models"]).all() and table["p_adj_all"].max() <= 1.0
    text = fc.summarize_many(table)
    assert "Tested 3 stocks" in text


def test_many_stock_summary_handles_nothing_testable():
    import pandas as pd
    from stocklens.models import forecasting as fc

    assert "enough history" in fc.summarize_many(pd.DataFrame({"note": ["x"]}, index=["A"]))

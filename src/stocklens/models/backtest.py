"""Vectorised backtester with transaction costs and NO look-ahead.

The two rules that make a backtest honest:
  1. A signal computed from today's close can only be traded TOMORROW  ->  position.shift(1).
  2. Every change in position pays a cost  ->  cost_bps * |change in position|.

Breaking either rule (especially #1) is the most common reason student backtests look
amazing. tests/test_backtest.py checks both.

Even with these rules, a backtest on one historical path is evidence, not proof:
parameter choices are fitted to that path, and a fixed watchlist has survivorship bias
(stocks that failed or were delisted aren't in it). Say so in the README.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stocklens.analysis import risk, indicators


# ---- signals (each returns a position decided at the close of day t) --------------------

def ma_crossover_position(price: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """Long (1) when the fast SMA is above the slow SMA, otherwise flat (0)."""
    if fast >= slow:
        raise ValueError("fast must be smaller than slow")
    f, s = indicators.sma(price, fast), indicators.sma(price, slow)
    return (f > s).astype(float).where(s.notna(), 0.0)


def time_series_momentum_position(price: pd.Series, lookback: int = 126) -> pd.Series:
    """Long when the past `lookback`-day return is positive, otherwise flat."""
    past = price / price.shift(lookback) - 1
    return (past > 0).astype(float).where(past.notna(), 0.0)


def cross_sectional_momentum_weights(
    prices: pd.DataFrame, lookback: int = 126, top_n: int = 3, rebalance: str = "ME"
) -> pd.DataFrame:
    """Each month-end hold the `top_n` symbols with the best `lookback`-day return, equal weight.
    Weights are decided at the close of the rebalance day and held until the next rebalance."""
    past = prices / prices.shift(lookback) - 1
    # last available trading day of each period (month-end by default)
    reb_dates = prices.groupby(prices.index.to_period(_period_code(rebalance))).tail(1).index
    w = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for d in reb_dates:
        row = past.loc[d].dropna()
        if len(row) < top_n:
            w.loc[d] = 0.0
            continue
        pick = row.nlargest(top_n).index
        w.loc[d] = 0.0
        w.loc[d, pick] = 1.0 / top_n
    return w.ffill().fillna(0.0)


def _period_code(rebalance: str) -> str:
    return {"ME": "M", "M": "M", "W": "W", "QE": "Q", "Q": "Q"}.get(rebalance, "M")


# ---- engine ------------------------------------------------------------------------------

@dataclass
class BacktestResult:
    returns: pd.Series                  # net of costs
    gross_returns: pd.Series
    position: pd.DataFrame | pd.Series  # what was actually held each day (already shifted)
    turnover: pd.Series
    cost_bps: float
    metrics: dict = field(default_factory=dict)


def _metrics(net: pd.Series, turnover: pd.Series, held: pd.Series | pd.DataFrame, periods: int) -> dict:
    exposure = held.abs().sum(axis=1) if isinstance(held, pd.DataFrame) else held.abs()
    years = max(len(net) / periods, 1e-9)
    return {
        "total_return": float((1 + net).prod() - 1),
        "cagr": risk.annualised_return(net, periods),
        "ann_volatility": risk.annualised_volatility(net, periods),
        "sharpe": risk.sharpe_ratio(net, 0.0, periods),
        "max_drawdown": risk.max_drawdown(net),
        "annual_turnover": float(turnover.sum() / years),
        "time_in_market": float((exposure > 0).mean()),
        "n_days": int(len(net)),
    }


def run_backtest(price: pd.Series, position: pd.Series, cost_bps: float = 10.0, periods: int = 252) -> BacktestResult:
    """Backtest a single-asset 0/1 (or -1..1) position series.

    cost_bps is charged on each unit of turnover (10 bps = 0.10% of the traded value). Use a
    realistic number for YOUR broker: include brokerage as a % of a typical trade, plus spread.
    """
    price = price.dropna()
    held = position.reindex(price.index).fillna(0.0).shift(1).fillna(0.0)  # <- no look-ahead
    asset_ret = price.pct_change().fillna(0.0)
    turnover = held.diff().abs()
    turnover.iloc[0] = abs(held.iloc[0])
    gross = held * asset_ret
    net = gross - turnover * cost_bps / 1e4
    res = BacktestResult(net, gross, held, turnover, cost_bps)
    res.metrics = _metrics(net, turnover, held, periods)
    return res


def run_portfolio_backtest(prices: pd.DataFrame, weights: pd.DataFrame, cost_bps: float = 10.0, periods: int = 252) -> BacktestResult:
    prices = prices.sort_index()
    held = weights.reindex(prices.index).fillna(0.0).shift(1).fillna(0.0)  # <- no look-ahead
    asset_ret = prices.pct_change(fill_method=None).fillna(0.0)
    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()
    gross = (held * asset_ret).sum(axis=1)
    net = gross - turnover * cost_bps / 1e4
    res = BacktestResult(net, gross, held, turnover, cost_bps)
    res.metrics = _metrics(net, turnover, held, periods)
    return res


def buy_and_hold(price: pd.Series, cost_bps: float = 10.0) -> BacktestResult:
    return run_backtest(price, pd.Series(1.0, index=price.index), cost_bps)


# ---- honesty tools -----------------------------------------------------------------------

def parameter_sweep(price: pd.Series, fast_grid=(10, 20, 50), slow_grid=(50, 100, 200), cost_bps: float = 10.0) -> pd.DataFrame:
    """Run every MA-crossover combo. If only ONE combo looks good, you've found noise, not a strategy."""
    rows = []
    for f in fast_grid:
        for s in slow_grid:
            if f >= s:
                continue
            m = run_backtest(price, ma_crossover_position(price, f, s), cost_bps).metrics
            rows.append({"fast": f, "slow": s, **{k: m[k] for k in ["cagr", "sharpe", "max_drawdown", "annual_turnover"]}})
    return pd.DataFrame(rows)


def in_sample_vs_out_of_sample(
    price: pd.Series, split: float = 0.6, fast_grid=(10, 20, 50), slow_grid=(50, 100, 200), cost_bps: float = 10.0
) -> dict:
    """Pick the best (fast, slow) by Sharpe on the first `split` of history, then test it on the rest.

    The out-of-sample number is the one to believe."""
    price = price.dropna()
    cut = int(len(price) * split)
    ins, oos = price.iloc[:cut], price.iloc[cut:]
    grid = parameter_sweep(ins, fast_grid, slow_grid, cost_bps)
    if grid.empty:
        raise ValueError("no valid parameter combinations")
    best = grid.sort_values("sharpe", ascending=False).iloc[0]
    f, s = int(best["fast"]), int(best["slow"])
    # compute signals on the FULL series (so the slow SMA is warm at the start of the test), score only the OOS slice
    pos_full = ma_crossover_position(price, f, s)
    oos_res = run_backtest(oos, pos_full.loc[oos.index], cost_bps)
    bh = buy_and_hold(oos, cost_bps)
    return {
        "chosen_fast": f,
        "chosen_slow": s,
        "in_sample_sharpe": float(best["sharpe"]),
        "oos_sharpe": oos_res.metrics["sharpe"],
        "oos_cagr": oos_res.metrics["cagr"],
        "oos_buy_hold_sharpe": bh.metrics["sharpe"],
        "oos_buy_hold_cagr": bh.metrics["cagr"],
        "oos_start": oos.index[0],
    }

"""Portfolio value over time, time-weighted returns, and benchmark comparison.

Time-weighted return (TWR) removes the effect of when YOU add or withdraw money, so it
is the right number to compare against an index. Convention used (end-of-day flows):

    flow_t   = money put into positions on day t   (buys + fees - sell proceeds)
    r_t      = (V_t + dividends_t - flow_t) / V_{t-1} - 1

where V_t is the closing market value of the shares held at the end of day t.
Days before the first purchase have no return (V_{t-1} = 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def positions_over_time(tx: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Shares held per symbol at each date in `index` (end of day)."""
    signed = tx[tx["type"].isin(["BUY", "SELL"])].copy()
    signed["delta"] = np.where(signed["type"] == "BUY", signed["quantity"], -signed["quantity"])
    daily = signed.pivot_table(index="date", columns="symbol", values="delta", aggfunc="sum").fillna(0.0)
    daily = daily.reindex(index.union(daily.index)).fillna(0.0).cumsum()
    return daily.reindex(index)


def cash_flows(tx: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily `flow` (net money invested) and `dividends` aligned to trading days in `index`."""
    t = tx.copy()
    gross = t["quantity"] * t["price"]
    t["flow"] = np.select(
        [t["type"] == "BUY", t["type"] == "SELL"],
        [gross + t["fees"], -(gross - t["fees"])],
        default=0.0,
    )
    t["div"] = np.where(t["type"] == "DIVIDEND", gross - t["fees"], 0.0)
    # trades on non-trading days roll forward to the next trading day
    pos = index.searchsorted(t["date"].values)
    pos = np.clip(pos, 0, len(index) - 1)
    t["day"] = index[pos]
    g = t.groupby("day")[["flow", "div"]].sum()
    out = pd.DataFrame(0.0, index=index, columns=["flow", "dividends"])
    out.loc[g.index, "flow"] = g["flow"]
    out.loc[g.index, "dividends"] = g["div"]
    return out


def portfolio_value(tx: pd.DataFrame, close: pd.DataFrame) -> pd.Series:
    """Daily market value. `close` = wide DataFrame of closing prices (index = trading days)."""
    close = close.ffill()
    held = positions_over_time(tx, close.index)
    cols = [c for c in held.columns if c in close.columns]
    missing = set(held.columns) - set(close.columns)
    if missing:
        raise KeyError(f"no price history for: {sorted(missing)}")
    value = (held[cols] * close[cols]).sum(axis=1)
    return value.rename("value")


def twr_daily_returns(tx: pd.DataFrame, close: pd.DataFrame) -> pd.Series:
    value = portfolio_value(tx, close)
    cf = cash_flows(tx, value.index)
    prev = value.shift(1)
    ret = (value + cf["dividends"] - cf["flow"]) / prev - 1
    ret = ret.where(prev > 0).dropna()
    return ret.rename("portfolio")


def cumulative(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod() - 1


def compare_to_benchmark(port_ret: pd.Series, bench_ret: pd.Series) -> dict:
    """Align on common dates and summarise portfolio vs benchmark."""
    df = pd.concat([port_ret.rename("portfolio"), bench_ret.rename("benchmark")], axis=1, join="inner").dropna()
    if len(df) < 5:
        raise ValueError("not enough overlapping days to compare")
    cum = (1 + df).cumprod() - 1
    active = df["portfolio"] - df["benchmark"]
    cov = df.cov()
    beta = cov.loc["portfolio", "benchmark"] / cov.loc["benchmark", "benchmark"]
    return {
        "cumulative": cum,
        "portfolio_total": float(cum["portfolio"].iloc[-1]),
        "benchmark_total": float(cum["benchmark"].iloc[-1]),
        "excess_total": float(cum["portfolio"].iloc[-1] - cum["benchmark"].iloc[-1]),
        "tracking_error": float(active.std() * np.sqrt(252)),
        "beta": float(beta),
        "n_days": int(len(df)),
    }

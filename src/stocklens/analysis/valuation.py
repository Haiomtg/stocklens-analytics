"""Valuation metrics, peer comparison and the watchlist screener.

IMPORTANT LIMITATION: fundamentals from free sources are a snapshot of TODAY, not
point-in-time history. The screener is therefore a research/ranking tool for the
present. Do NOT use fundamentals in historical backtests - that would be look-ahead
bias (you'd be "knowing" numbers that weren't public yet).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VALUE_COLS = ["pe", "pb"]           # lower is cheaper
QUALITY_COLS = ["roe", "debt_to_equity"]  # higher ROE better, lower D/E better


def peer_comparison(fundamentals: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Compare one stock's ratios with the median of the other stocks in its sector."""
    if symbol not in fundamentals.index:
        raise KeyError(symbol)
    metrics = ["pe", "pb", "roe", "debt_to_equity", "dividend_yield"]
    row = fundamentals.loc[symbol]
    sector = row.get("sector")
    peers = fundamentals[(fundamentals["sector"] == sector) & (fundamentals.index != symbol)]
    scope = "sector peers"
    if peers.empty:  # fall back to everything else on the watchlist
        peers = fundamentals[fundamentals.index != symbol]
        scope = "rest of watchlist"
    out = pd.DataFrame(
        {"value": row[metrics].astype(float), "peer_median": peers[metrics].astype(float).median()}
    )
    out["vs_peers"] = out["value"] / out["peer_median"] - 1
    out.attrs["scope"] = scope
    out.attrs["n_peers"] = len(peers)
    return out


def _pct_rank(s: pd.Series, ascending: bool = True) -> pd.Series:
    """0-1 percentile rank; best = 1. NaN stays NaN."""
    return s.rank(pct=True, ascending=ascending, na_option="keep")


def momentum_12_1(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Return from t-lookback to t-skip (skips the latest month, a standard momentum convention).
    Falls back to a shorter window if there isn't enough history."""
    n = len(prices)
    if n < 60:
        return pd.Series(np.nan, index=prices.columns)
    lookback = min(lookback, n - 1)
    skip = min(skip, max(lookback // 4, 1))
    return prices.iloc[-skip - 1] / prices.iloc[-lookback - 1] - 1


def score_watchlist(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rank symbols by Quality, Value and Momentum (each 0-1, higher = better) plus a composite."""
    weights = weights or {"quality": 1 / 3, "value": 1 / 3, "momentum": 1 / 3}
    syms = list(prices.columns)
    f = fundamentals.reindex(syms)

    value = pd.concat([_pct_rank(f[c], ascending=False) for c in VALUE_COLS if c in f], axis=1).mean(axis=1)
    quality = pd.concat(
        [
            _pct_rank(f["roe"], ascending=True) if "roe" in f else None,
            _pct_rank(f["debt_to_equity"], ascending=False) if "debt_to_equity" in f else None,
        ],
        axis=1,
    ).mean(axis=1)
    mom_raw = momentum_12_1(prices)
    momentum = _pct_rank(mom_raw, ascending=True)

    out = pd.DataFrame({"quality": quality, "value": value, "momentum": momentum, "momentum_return": mom_raw})
    w = pd.Series(weights)
    out["composite"] = (out[["quality", "value", "momentum"]] * w).sum(axis=1, skipna=True) / (
        out[["quality", "value", "momentum"]].notna() * w
    ).sum(axis=1)
    out["rank"] = out["composite"].rank(ascending=False, method="min")
    return out.sort_values("composite", ascending=False)

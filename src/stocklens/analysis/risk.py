"""Risk analytics on RETURN series (daily simple returns unless stated).

periods = trading days per year used to annualise. 252 is fine for US/ASX; Vietnam has
roughly 245-250 trading days, so the difference is small.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def annualised_return(returns: pd.Series, periods: int = 252) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    growth = (1 + r).prod()
    return float(growth ** (periods / len(r)) - 1)


def annualised_volatility(returns: pd.Series, periods: int = 252) -> float:
    return float(returns.dropna().std() * np.sqrt(periods))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods: int = 252) -> float:
    """Annualised Sharpe. `risk_free` is an ANNUAL rate (e.g. 0.04 for 4%)."""
    r = returns.dropna()
    excess = r - risk_free / periods
    sd = excess.std()
    return float(excess.mean() / sd * np.sqrt(periods)) if sd and not np.isnan(sd) else float("nan")


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns.dropna()).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1


def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    return float(dd.min()) if not dd.empty else float("nan")


def beta(returns: pd.Series, bench: pd.Series) -> float:
    df = pd.concat([returns, bench], axis=1, join="inner").dropna()
    if len(df) < 30:
        return float("nan")
    c = df.cov().values
    return float(c[0, 1] / c[1, 1])


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.dropna(how="all").corr()


def historical_var(returns: pd.Series, level: float = 0.95) -> float:
    """1-day historical Value-at-Risk as a POSITIVE loss fraction (0.021 = lose 2.1%)."""
    r = returns.dropna()
    return float(-np.percentile(r, (1 - level) * 100)) if len(r) else float("nan")


def expected_shortfall(returns: pd.Series, level: float = 0.95) -> float:
    """Average loss on the days that are worse than the VaR threshold (a.k.a. CVaR)."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    cutoff = np.percentile(r, (1 - level) * 100)
    tail = r[r <= cutoff]
    return float(-tail.mean())


def risk_summary(returns: pd.Series, bench: pd.Series | None = None, risk_free: float = 0.0, periods: int = 252) -> dict:
    out = {
        "ann_return": annualised_return(returns, periods),
        "ann_volatility": annualised_volatility(returns, periods),
        "sharpe": sharpe_ratio(returns, risk_free, periods),
        "max_drawdown": max_drawdown(returns),
        "var_95_1d": historical_var(returns, 0.95),
        "es_95_1d": expected_shortfall(returns, 0.95),
        "n_days": int(returns.dropna().shape[0]),
    }
    if bench is not None:
        out["beta"] = beta(returns, bench)
    return out


# ---- diversification ----------------------------------------------------------------

def concentration_report(
    holdings: pd.DataFrame,
    max_position: float = 0.25,
    max_sector: float = 0.50,
) -> dict:
    """Weights by position and sector, HHI, and plain-English warnings.

    `holdings` needs columns market_value and (optionally) sector, indexed by symbol.
    """
    h = holdings[holdings["market_value"] > 0].copy()
    total = h["market_value"].sum()
    if total <= 0:
        return {"hhi": float("nan"), "effective_n": float("nan"), "warnings": [], "by_sector": pd.Series(dtype=float)}
    w = h["market_value"] / total
    hhi = float((w**2).sum())
    warnings = [f"{s} is {x:.0%} of the portfolio (limit {max_position:.0%})" for s, x in w.items() if x > max_position]
    by_sector = pd.Series(dtype=float)
    if "sector" in h.columns:
        by_sector = (h.groupby(h["sector"].fillna("Unknown"))["market_value"].sum() / total).sort_values(ascending=False)
        warnings += [f"{s} sector is {x:.0%} of the portfolio (limit {max_sector:.0%})" for s, x in by_sector.items() if x > max_sector]
    return {"hhi": hhi, "effective_n": 1 / hhi, "weights": w, "by_sector": by_sector, "warnings": warnings}

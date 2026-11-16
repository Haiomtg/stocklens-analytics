"""Data-quality checks on daily price data.

Each check returns rows of (date, flag, detail). Checks only FLAG problems - they
never silently "fix" data, so you can decide what to do about each one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FLAG_COLS = ["date", "flag", "detail"]
SPLIT_RATIOS = (2, 3, 4, 5, 10)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=FLAG_COLS)


def find_gaps(dates: pd.Series, max_gap_bdays: int = 5) -> pd.DataFrame:
    """Flag stretches with more than `max_gap_bdays` consecutive missing business days.

    Single missing days are normal (public holidays differ per market), so only long
    gaps are flagged.
    """
    d = pd.to_datetime(pd.Series(dates)).sort_values().reset_index(drop=True)
    if len(d) < 2:
        return _empty()
    rows = []
    for prev, cur in zip(d[:-1], d[1:]):
        missing = len(pd.bdate_range(prev, cur)) - 2
        if missing > max_gap_bdays:
            rows.append((cur, "gap", f"{missing} business days missing before this date (after {prev.date()})"))
    return pd.DataFrame(rows, columns=FLAG_COLS) if rows else _empty()


def find_outliers(close: pd.Series, z: float = 6.0, min_abs_move: float = 0.10) -> pd.DataFrame:
    """Flag daily returns that are both many robust-sigmas away AND large in absolute terms."""
    close = close.dropna()
    ret = close.pct_change().dropna()
    if len(ret) < 30:
        return _empty()
    med = ret.median()
    mad = (ret - med).abs().median() * 1.4826 or ret.std()
    score = (ret - med).abs() / mad
    hit = ret[(score > z) & (ret.abs() > min_abs_move)]
    rows = [(d, "outlier", f"daily return {r:+.1%} ({score[d]:.1f} robust sigmas)") for d, r in hit.items()]
    return pd.DataFrame(rows, columns=FLAG_COLS) if rows else _empty()


def find_suspect_splits(close: pd.Series, tol: float = 0.03) -> pd.DataFrame:
    """Flag overnight drops that look like an UNADJUSTED stock split (e.g. -50% for a 2:1)."""
    ret = close.dropna().pct_change().dropna()
    rows = []
    for d, r in ret.items():
        for k in SPLIT_RATIOS:
            if abs(r - (1 / k - 1)) < tol:
                rows.append((d, "possible_split", f"price fell {r:+.1%}, close to a {k}:1 split"))
                break
    return pd.DataFrame(rows, columns=FLAG_COLS) if rows else _empty()


def find_bad_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Non-positive prices, high < low, close outside the high/low range, zero-volume streaks."""
    rows = []
    d = df.set_index(pd.to_datetime(df["date"]))
    bad_price = d[(d["close"] <= 0) | (d["high"] < d["low"])]
    rows += [(i, "bad_bar", "non-positive price or high < low") for i in bad_price.index]
    tol = 1e-6
    outside = d[(d["close"] > d["high"] * (1 + tol)) | (d["close"] < d["low"] * (1 - tol))]
    rows += [(i, "bad_bar", "close outside high/low range") for i in outside.index]
    return pd.DataFrame(rows, columns=FLAG_COLS) if rows else _empty()


def stale_check(df: pd.DataFrame, today: pd.Timestamp | None = None, max_age_bdays: int = 5) -> pd.DataFrame:
    """Flag if the newest bar is old (broken fetcher / delisted ticker)."""
    if df.empty:
        return pd.DataFrame([(pd.Timestamp.today().normalize(), "empty", "no price data")], columns=FLAG_COLS)
    today = (today or pd.Timestamp.today()).normalize()
    last = pd.to_datetime(df["date"]).max()
    age = len(pd.bdate_range(last, today)) - 1
    if age > max_age_bdays:
        return pd.DataFrame([(last, "stale", f"latest bar is {age} business days old")], columns=FLAG_COLS)
    return _empty()


def run_all(df: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Run every check on one symbol's price frame (columns: date, open..volume)."""
    if df.empty:
        return stale_check(df, today)
    close = df.set_index(pd.to_datetime(df["date"]))["close"]
    parts = [
        find_gaps(df["date"]),
        find_outliers(close),
        find_suspect_splits(close),
        find_bad_bars(df),
        stale_check(df, today),
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return _empty()
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out.sort_values("date").reset_index(drop=True)

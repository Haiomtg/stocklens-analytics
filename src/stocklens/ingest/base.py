"""Fetcher interface.

Free data sources break and change terms. Keeping every source behind this tiny
interface means a broken source is a one-file fix, not a project-wide rewrite.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd

PRICE_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


class DataFetcher(Protocol):
    market: str

    def fetch_prices(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        """Return columns PRICE_COLUMNS, one row per trading day, `date` as datetime."""

    def fetch_dividends(self, symbol: str) -> pd.DataFrame:
        """Return columns [date, amount] (cash per share). Empty frame if none."""

    def fetch_fundamentals(self, symbol: str) -> dict:
        """Return a dict matching the `fundamentals` table (without `as_of`). May be partial."""


def validate_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly if a fetcher returns something the rest of the code can't use."""
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"fetcher output is missing columns: {missing}")
    out = df[PRICE_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out.dropna(subset=["close"]).drop_duplicates("date").sort_values("date")
    return out.reset_index(drop=True)

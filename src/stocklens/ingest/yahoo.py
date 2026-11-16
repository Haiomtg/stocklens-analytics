"""Yahoo Finance fetcher (US and ASX stocks and indices) via the `yfinance` package.

NOTE: yfinance is an unofficial scraper of Yahoo's public endpoints. Yahoo's terms
restrict it to personal use, and it breaks occasionally. That is why the raw
data is cached in SQLite and the public demo ships with synthetic sample data.
"""
from __future__ import annotations

import pandas as pd

from stocklens.ingest.base import validate_price_frame


class YahooFetcher:
    def __init__(self, market: str = "US"):
        self.market = market

    def fetch_prices(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        import yfinance as yf  # imported lazily so tests/demo work without it

        raw = yf.download(
            symbol, start=start, end=end, auto_adjust=False, progress=False, threads=False
        )
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])
        if isinstance(raw.columns, pd.MultiIndex):  # newer yfinance returns (field, ticker)
            raw.columns = raw.columns.get_level_values(0)
        df = raw.reset_index().rename(
            columns={
                "Date": "date", "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
            }
        )
        if "adj_close" not in df:  # indices have no adjusted close
            df["adj_close"] = df["close"]
        return validate_price_frame(df)

    def fetch_dividends(self, symbol: str) -> pd.DataFrame:
        import yfinance as yf

        s = yf.Ticker(symbol).dividends
        if s is None or s.empty:
            return pd.DataFrame(columns=["date", "amount"])
        df = s.reset_index()
        df.columns = ["date", "amount"]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df

    def fetch_fundamentals(self, symbol: str) -> dict:
        import yfinance as yf

        info = yf.Ticker(symbol).info or {}
        dy = info.get("dividendYield")
        # Yahoo has switched between fraction (0.04) and percent (4.0) over time.
        # No real dividend yield is above 100%, so >1 means it arrived as a percent.
        if dy is not None and dy > 1:
            dy = dy / 100
        de = info.get("debtToEquity")  # Yahoo reports this as a percent (45.2 = 0.452)
        return {
            "symbol": symbol,
            "sector": info.get("sector"),
            "market_cap": info.get("marketCap"),
            "pe": info.get("trailingPE"),
            "pb": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"),
            "debt_to_equity": (de / 100) if de is not None else None,
            "dividend_yield": dy,
        }

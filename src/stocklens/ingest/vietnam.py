"""Vietnamese equities fetcher via the `vnstock` package.

!!! vnstock IS CURRENTLY NOT INSTALLABLE FROM PyPI (checked 16 Nov 2026) !!!
PyPI's own project page says vnstock is "quarantined by PyPI admins" pending a security
review, so `pip install vnstock` fails with "from versions: none". Do not work around this
by installing it from a random mirror or fork. Check https://pypi.org/project/vnstock/
again later; if it returns, newer versions may also need a free API key / registration,
so re-read its docs. Until then this fetcher raises a clear error, and the rest of the
project works without it. Safe alternatives: write a CsvFetcher that imports price files you
export yourself from your broker or a data site (same three methods), or another source.

!!! UNTESTED AGAINST THE LIVE API !!!
vnstock's interface has changed between major versions and I could not call the
live service while writing this. The code follows the vnstock v3 pattern:

    from vnstock import Vnstock
    stock = Vnstock().stock(symbol="VRE", source="VCI")
    stock.quote.history(start="2024-01-01", end="2024-12-31", interval="1D")

YOUR JOB (Stage 1): run `python -m stocklens.ingest.vietnam VRE` and fix whatever
differs in your installed version. Things to check:
  * column names (expected: time, open, high, low, close, volume)
  * price units - VCI quotes prices in THOUSANDS of VND (25.5 = 25,500 VND).
    This file leaves them as-is; be consistent with how you log transactions.
  * VN stock prices from this source are typically already adjusted for splits
    and bonus shares, but not dividends, so adj_close == close here.
If vnstock breaks permanently, write a new class with the same three methods and
swap it in `update.py` - nothing else needs to change.
"""
from __future__ import annotations

import sys

import pandas as pd

from stocklens.ingest.base import validate_price_frame


class VietnamFetcher:
    market = "VN"

    def __init__(self, source: str = "VCI"):
        self.source = source

    def _stock(self, symbol: str):
        from vnstock import Vnstock  # lazy import

        return Vnstock().stock(symbol=symbol, source=self.source)

    def fetch_prices(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        raw = self._stock(symbol).quote.history(start=start, end=end, interval="1D")
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])
        df = raw.rename(columns={"time": "date"}).copy()
        df["adj_close"] = df["close"]
        return validate_price_frame(df)

    def fetch_dividends(self, symbol: str) -> pd.DataFrame:
        # Dividend history differs a lot between vnstock versions/sources. Left empty
        # on purpose: log VN dividends in your own transactions CSV instead (type=DIVIDEND).
        return pd.DataFrame(columns=["date", "amount"])

    def fetch_fundamentals(self, symbol: str) -> dict:
        # TODO (Stage 3, optional): pull ratios from vnstock's finance/company modules and map
        # them to: sector, market_cap, pe, pb, roe, debt_to_equity, dividend_yield.
        return {"symbol": symbol}


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "VRE"
    out = VietnamFetcher().fetch_prices(sym, "2024-01-01")
    print(out.tail())
    print(f"{len(out)} rows")

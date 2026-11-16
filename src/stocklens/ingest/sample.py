"""Synthetic sample data for the public demo, tests and CI.

Everything here is RANDOMLY GENERATED. The tickers are fake (SYN-prefixed) so
nobody mistakes them for real securities. This is what lets the Streamlit demo
run without internet and without exposing your real holdings.

Run:  python -m stocklens.ingest.sample
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stocklens import db

# symbol, market, sector, beta to market factor, annual drift, idio vol
SYNTHETIC = [
    ("SYNUS1", "US", "Technology", 1.2, 0.12, 0.22),
    ("SYNUS2", "US", "Technology", 1.1, 0.10, 0.20),
    ("SYNUS3", "US", "Healthcare", 0.8, 0.07, 0.15),
    ("SYNAX1", "ASX", "Materials", 1.0, 0.06, 0.20),
    ("SYNAX2", "ASX", "Financials", 0.9, 0.08, 0.14),
    ("SYNAX3", "ASX", "Financials", 0.9, 0.07, 0.13),
    ("SYNVN1", "VN", "Real Estate", 1.3, 0.09, 0.28),
    ("SYNVN2", "VN", "Utilities", 0.7, 0.07, 0.18),
    ("SYNVN3", "VN", "Consumer", 1.0, 0.10, 0.24),
]
BENCHMARK = "SYNIDX"


def make_prices(n_days: int = 1500, seed: int = 42, end: str | None = None) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end_ts, periods=n_days)
    market = rng.normal(0.07 / 252, 0.15 / np.sqrt(252), n_days)  # common factor
    out: dict[str, pd.DataFrame] = {}

    def to_frame(rets: np.ndarray, start_price: float) -> pd.DataFrame:
        close = start_price * np.exp(np.cumsum(rets))
        noise = rng.uniform(0.002, 0.012, n_days)
        return pd.DataFrame(
            {
                "date": dates,
                "open": close * (1 - noise / 2),
                "high": close * (1 + noise),
                "low": close * (1 - noise),
                "close": close,
                "adj_close": close,
                "volume": rng.integers(200_000, 5_000_000, n_days).astype(float),
            }
        )

    out[BENCHMARK] = to_frame(market, 1000.0)
    for sym, _mkt, _sec, beta, drift, ivol in SYNTHETIC:
        idio = rng.normal(drift / 252, ivol / np.sqrt(252), n_days)
        out[sym] = to_frame(beta * market + idio, float(rng.uniform(10, 120)))
    return out


def make_fundamentals(seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for sym, _mkt, sector, *_ in SYNTHETIC:
        rows.append(
            {
                "symbol": sym,
                "as_of": pd.Timestamp.today().strftime("%Y-%m-%d"),
                "sector": sector,
                "market_cap": float(rng.uniform(1e9, 5e10)),
                "pe": float(rng.uniform(8, 35)),
                "pb": float(rng.uniform(0.8, 6)),
                "roe": float(rng.uniform(0.04, 0.30)),
                "debt_to_equity": float(rng.uniform(0.1, 1.8)),
                "dividend_yield": float(rng.uniform(0.0, 0.06)),
            }
        )
    return rows


def make_transactions() -> pd.DataFrame:
    """A small fake trade log (types: BUY / SELL / DIVIDEND)."""
    today = pd.Timestamp.today().normalize()
    rows = [
        (-700, "SYNUS1", "BUY", 200, None, 9.95),
        (-650, "SYNAX2", "BUY", 200, None, 9.95),
        (-600, "SYNVN1", "BUY", 150, None, 0.0),
        (-500, "SYNUS3", "BUY", 300, None, 9.95),
        (-420, "SYNAX1", "BUY", 150, None, 9.95),
        (-300, "SYNUS1", "SELL", 50, None, 9.95),
        (-250, "SYNVN2", "BUY", 250, None, 0.0),
        (-150, "SYNAX2", "BUY", 100, None, 9.95),
        (-90, "SYNAX2", "DIVIDEND", 300, 0.55, 0.0),
        (-60, "SYNUS2", "BUY", 150, None, 9.95),
    ]
    recs = []
    for off, sym, typ, qty, price, fee in rows:
        d = (today + pd.tseries.offsets.BDay(int(off / 1.4))).normalize()
        recs.append({"date": d, "symbol": sym, "type": typ, "quantity": qty, "price": price, "fees": fee})
    return pd.DataFrame(recs)


def build_sample_db(db_path: str | Path, transactions_path: str | Path | None = None, seed: int = 42) -> None:
    conn = db.connect(db_path)
    prices = make_prices(seed=seed)
    markets = {s: m for s, m, *_ in SYNTHETIC} | {BENCHMARK: "BENCH"}
    for sym, df in prices.items():
        db.upsert_prices(conn, df, sym, markets[sym])
    for row in make_fundamentals():
        db.upsert_fundamentals(conn, row)

    tx = make_transactions()
    # Fill in prices for BUY/SELL rows from the synthetic close on the nearest prior day.
    close = db.load_prices(conn, field="close")
    prices_col = []
    for r in tx.itertuples():
        if pd.notna(r.price):
            prices_col.append(r.price)
        else:
            s = close[r.symbol].loc[:r.date].dropna()
            prices_col.append(round(float(s.iloc[-1]), 2))
    tx["price"] = prices_col
    tx["date"] = tx["date"].dt.strftime("%Y-%m-%d")
    if transactions_path:
        Path(transactions_path).parent.mkdir(parents=True, exist_ok=True)
        tx.to_csv(transactions_path, index=False)
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the synthetic demo database + fake transactions.")
    ap.add_argument("--db", default="data/sample.db")
    ap.add_argument("--transactions", default="data/sample_transactions.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_sample_db(args.db, args.transactions, args.seed)
    print(f"Wrote {args.db} and {args.transactions} (synthetic data, not real securities).")


if __name__ == "__main__":
    main()

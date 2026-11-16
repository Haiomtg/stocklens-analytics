"""SQLite storage layer.

One small database holds prices, dividends, fundamentals snapshots and
data-quality flags. Everything else in the project reads through these helpers,
so swapping SQLite for PostgreSQL later only means changing this file.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_DB = Path(os.environ.get("STOCKLENS_DB", "data/stocklens.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    symbol    TEXT NOT NULL,
    market    TEXT NOT NULL,
    date      TEXT NOT NULL,          -- ISO yyyy-mm-dd
    open      REAL,
    high      REAL,
    low       REAL,
    close     REAL,                   -- split-adjusted, NOT dividend-adjusted
    adj_close REAL,                   -- split + dividend adjusted (use for returns)
    volume    REAL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS dividends (
    symbol TEXT NOT NULL,
    date   TEXT NOT NULL,
    amount REAL NOT NULL,             -- cash per share
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    symbol         TEXT NOT NULL,
    as_of          TEXT NOT NULL,     -- snapshot date: fundamentals are NOT point-in-time
    sector         TEXT,
    market_cap     REAL,
    pe             REAL,
    pb             REAL,
    roe            REAL,              -- fraction (0.15 = 15%)
    debt_to_equity REAL,              -- ratio (0.5 = 50%)
    dividend_yield REAL,              -- fraction (0.04 = 4%)
    PRIMARY KEY (symbol, as_of)
);

CREATE TABLE IF NOT EXISTS quality_flags (
    symbol  TEXT NOT NULL,
    date    TEXT NOT NULL,
    flag    TEXT NOT NULL,
    detail  TEXT,
    PRIMARY KEY (symbol, date, flag)
);
"""

PRICE_COLS = ["open", "high", "low", "close", "adj_close", "volume"]


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(path) if path else DEFAULT_DB
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    return conn


def upsert_prices(conn: sqlite3.Connection, df: pd.DataFrame, symbol: str, market: str) -> int:
    """Insert or replace daily bars. `df` needs a `date` column plus PRICE_COLS."""
    if df.empty:
        return 0
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["symbol"] = symbol
    out["market"] = market
    for col in PRICE_COLS:
        if col not in out:
            out[col] = None
    rows = out[["symbol", "market", "date", *PRICE_COLS]].itertuples(index=False, name=None)
    conn.executemany(
        "INSERT OR REPLACE INTO prices (symbol, market, date, open, high, low, close, adj_close, volume) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        list(rows),
    )
    conn.commit()
    return len(out)


def upsert_dividends(conn: sqlite3.Connection, df: pd.DataFrame, symbol: str) -> int:
    """`df` needs `date` and `amount` columns."""
    if df.empty:
        return 0
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    rows = [(symbol, d, float(a)) for d, a in zip(out["date"], out["amount"])]
    conn.executemany("INSERT OR REPLACE INTO dividends (symbol, date, amount) VALUES (?,?,?)", rows)
    conn.commit()
    return len(rows)


def upsert_fundamentals(conn: sqlite3.Connection, row: dict) -> None:
    cols = ["symbol", "as_of", "sector", "market_cap", "pe", "pb", "roe", "debt_to_equity", "dividend_yield"]
    conn.execute(
        f"INSERT OR REPLACE INTO fundamentals ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        [row.get(c) for c in cols],
    )
    conn.commit()


def replace_quality_flags(conn: sqlite3.Connection, symbol: str, flags: pd.DataFrame) -> None:
    conn.execute("DELETE FROM quality_flags WHERE symbol = ?", (symbol,))
    if not flags.empty:
        conn.executemany(
            "INSERT OR REPLACE INTO quality_flags (symbol, date, flag, detail) VALUES (?,?,?,?)",
            [(symbol, str(r.date), r.flag, r.detail) for r in flags.itertuples()],
        )
    conn.commit()


def last_date(conn: sqlite3.Connection, symbol: str) -> str | None:
    row = conn.execute("SELECT MAX(date) FROM prices WHERE symbol = ?", (symbol,)).fetchone()
    return row[0] if row and row[0] else None


def list_symbols(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT symbol, market, MIN(date) AS first_date, MAX(date) AS last_date, COUNT(*) AS n_rows "
        "FROM prices GROUP BY symbol, market ORDER BY market, symbol",
        conn,
    )


def load_prices(
    conn: sqlite3.Connection,
    symbols: list[str] | None = None,
    field: str = "adj_close",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Wide DataFrame: index = date, columns = symbols, values = `field`."""
    if field not in PRICE_COLS:
        raise ValueError(f"field must be one of {PRICE_COLS}")
    q = f"SELECT symbol, date, {field} AS v FROM prices WHERE 1=1"
    params: list = []
    if symbols:
        q += f" AND symbol IN ({','.join('?' * len(symbols))})"
        params += list(symbols)
    if start:
        q += " AND date >= ?"
        params.append(start)
    if end:
        q += " AND date <= ?"
        params.append(end)
    long = pd.read_sql_query(q, conn, params=params)
    if long.empty:
        return pd.DataFrame()
    wide = long.pivot(index="date", columns="symbol", values="v")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def load_dividends(conn: sqlite3.Connection, symbols: list[str] | None = None) -> pd.DataFrame:
    q = "SELECT symbol, date, amount FROM dividends"
    params: list = []
    if symbols:
        q += f" WHERE symbol IN ({','.join('?' * len(symbols))})"
        params = list(symbols)
    df = pd.read_sql_query(q, conn, params=params)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_fundamentals(conn: sqlite3.Connection) -> pd.DataFrame:
    """Latest fundamentals snapshot per symbol."""
    df = pd.read_sql_query(
        "SELECT f.* FROM fundamentals f "
        "JOIN (SELECT symbol, MAX(as_of) AS m FROM fundamentals GROUP BY symbol) x "
        "ON f.symbol = x.symbol AND f.as_of = x.m",
        conn,
    )
    return df.set_index("symbol") if not df.empty else df


def load_quality_flags(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM quality_flags ORDER BY symbol, date", conn)

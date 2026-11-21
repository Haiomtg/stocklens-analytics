"""Loads everything the dashboard and the weekly report need, in one place."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stocklens import db
from stocklens.analysis import risk
from stocklens.portfolio import returns as pret
from stocklens.portfolio import tracker

DEFAULT_BENCHMARKS = ["SYNIDX", "^AXJO", "^GSPC", "VNINDEX"]


@dataclass
class Context:
    close: pd.DataFrame
    adj: pd.DataFrame
    fundamentals: pd.DataFrame
    symbols: pd.DataFrame
    flags: pd.DataFrame
    transactions: pd.DataFrame | None
    holdings: pd.DataFrame | None
    portfolio_returns: pd.Series | None
    benchmark: str | None
    benchmark_returns: pd.Series | None
    portfolio_error: str | None = None   # why the portfolio could not be built (shown on the pages)

    @property
    def tradable(self) -> list[str]:
        """Symbols that are not benchmarks/indices."""
        return [c for c in self.adj.columns if c != self.benchmark and not c.startswith("^") and c != "VNINDEX" and c != "SYNIDX"]


def pick_benchmark(available: list[str], preferred: str | None = None) -> str | None:
    if preferred and preferred in available:
        return preferred
    return next((b for b in DEFAULT_BENCHMARKS if b in available), None)


def load_context(db_path: str | Path, transactions_path: str | Path | None = None, benchmark: str | None = None) -> Context:
    conn = db.connect(db_path)
    close = db.load_prices(conn, field="close")
    adj = db.load_prices(conn, field="adj_close")
    fundamentals = db.load_fundamentals(conn)
    symbols = db.list_symbols(conn)
    flags = db.load_quality_flags(conn)
    conn.close()

    bench = pick_benchmark(list(adj.columns), benchmark)
    bench_ret = adj[bench].pct_change(fill_method=None) if bench else None

    tx = holdings = port_ret = None
    error = None
    if transactions_path and Path(transactions_path).exists():
        try:
            tx = tracker.load_transactions(transactions_path)
            missing = sorted(set(tx["symbol"]) - set(close.columns))
            if missing:
                raise KeyError(
                    f"your transactions mention symbols that are not in the database: {missing}. "
                    "Use the same ticker names as your watchlist (e.g. BHP.AX), or point the sidebar at the right files."
                )
            sectors = fundamentals["sector"] if "sector" in fundamentals else None
            holdings = tracker.holdings_table(tx, close.ffill().iloc[-1], sectors)
            port_ret = pret.twr_daily_returns(tx, close)
        except (KeyError, ValueError) as exc:  # bad/mismatched transactions must not take the whole app down
            tx = holdings = port_ret = None
            error = str(exc).strip("'\"")
    return Context(close, adj, fundamentals, symbols, flags, tx, holdings, port_ret, bench, bench_ret, error)


def portfolio_risk(ctx: Context) -> dict | None:
    if ctx.portfolio_returns is None:
        return None
    return risk.risk_summary(ctx.portfolio_returns, ctx.benchmark_returns)

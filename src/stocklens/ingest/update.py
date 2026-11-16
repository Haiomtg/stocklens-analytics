"""Daily update job: fetch new prices, dividends, fundamentals; run quality checks.

    python -m stocklens.ingest.update                      # uses config/watchlist.toml
    python -m stocklens.ingest.update --db data/stocklens.db --no-fundamentals

Designed to be re-run safely: it only fetches data after the last stored date, and
one failing symbol never stops the others.
"""
from __future__ import annotations

import argparse
import logging
import sys
import tomllib
from pathlib import Path

import pandas as pd

from stocklens import db
from stocklens.ingest import quality
from stocklens.ingest.vietnam import VietnamFetcher
from stocklens.ingest.yahoo import YahooFetcher

log = logging.getLogger("stocklens.update")


def get_fetcher(market: str):
    if market == "VN":
        return VietnamFetcher()
    return YahooFetcher(market)  # US + ASX


def load_watchlist(path: str | Path) -> tuple[list[dict], dict, str]:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    items = list(cfg.get("symbols", []))
    bench = cfg.get("benchmarks", {})
    for market, sym in bench.items():  # benchmarks are stored like any other symbol
        items.append({"symbol": sym, "market": market, "is_benchmark": True})
    return items, bench, cfg.get("start_date", "2018-01-01")


def update_symbol(conn, item: dict, default_start: str, with_fundamentals: bool) -> dict:
    symbol, market = item["symbol"], item["market"]
    fetcher = get_fetcher(market)
    last = db.last_date(conn, symbol)
    start = (pd.Timestamp(last) - pd.Timedelta(days=7)).strftime("%Y-%m-%d") if last else default_start
    # re-fetch the last week so late corrections / adjustments overwrite stored rows

    prices = fetcher.fetch_prices(symbol, start)
    if prices.empty and not last:
        # yfinance returns an empty frame (not an error) when it is blocked or the ticker is wrong
        raise RuntimeError("no price data returned - check the ticker symbol, your internet, or the data source")
    n = db.upsert_prices(conn, prices, symbol, market)
    if not item.get("is_benchmark"):
        db.upsert_dividends(conn, fetcher.fetch_dividends(symbol), symbol)
        if with_fundamentals:
            f = fetcher.fetch_fundamentals(symbol)
            if len(f) > 1:
                f["as_of"] = pd.Timestamp.today().strftime("%Y-%m-%d")
                db.upsert_fundamentals(conn, f)

    full = pd.read_sql_query("SELECT * FROM prices WHERE symbol = ? ORDER BY date", conn, params=(symbol,))
    flags = quality.run_all(full)
    db.replace_quality_flags(conn, symbol, flags)
    return {"symbol": symbol, "rows": n, "flags": len(flags)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watchlist", default="config/watchlist.toml")
    ap.add_argument("--db", default=str(db.DEFAULT_DB))
    ap.add_argument("--no-fundamentals", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    items, _bench, start = load_watchlist(args.watchlist)
    conn = db.connect(args.db)
    ok, failed = 0, []
    for item in items:
        try:
            res = update_symbol(conn, item, start, not args.no_fundamentals)
            log.info("%s: %d rows upserted, %d quality flags", res["symbol"], res["rows"], res["flags"])
            ok += 1
        except Exception as exc:  # one bad ticker must not stop the job
            log.error("%s FAILED: %s", item["symbol"], exc)
            failed.append(item["symbol"])
    log.info("done: %d ok, %d failed %s", ok, len(failed), failed or "")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

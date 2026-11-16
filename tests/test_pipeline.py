"""End-to-end smoke tests on the synthetic sample data (no internet needed)."""
from stocklens.context import load_context
from stocklens.ingest import sample
from stocklens.reports import weekly


def test_sample_build_context_and_report(tmp_path):
    db_path, tx_path = tmp_path / "s.db", tmp_path / "tx.csv"
    sample.build_sample_db(db_path, tx_path)
    ctx = load_context(db_path, tx_path)
    assert ctx.benchmark == "SYNIDX"
    assert ctx.holdings is not None and (ctx.holdings["market_value"] >= 0).all()
    assert ctx.portfolio_returns is not None and len(ctx.portfolio_returns) > 100
    html = weekly.build_report(ctx)
    assert "StockLens weekly report" in html and "not financial advice" in html.lower()
    assert "SYNVN1" in html  # holdings table rendered


def test_update_cli_survives_failing_symbols(tmp_path, monkeypatch):
    """One broken ticker must not stop the whole job."""
    from stocklens import db
    from stocklens.ingest import update

    wl = tmp_path / "w.toml"
    wl.write_text('start_date="2024-01-01"\n[[symbols]]\nsymbol="GOOD"\nmarket="US"\n[[symbols]]\nsymbol="BAD"\nmarket="US"\n')
    good = sample.make_prices(n_days=300)["SYNUS1"]

    class Fake:
        market = "US"
        def fetch_prices(self, symbol, start, end=None):
            if symbol == "BAD":
                raise RuntimeError("source down")
            return good
        def fetch_dividends(self, symbol):
            import pandas as pd
            return pd.DataFrame(columns=["date", "amount"])
        def fetch_fundamentals(self, symbol):
            return {"symbol": symbol}

    monkeypatch.setattr(update, "get_fetcher", lambda market: Fake())
    code = update.main(["--watchlist", str(wl), "--db", str(tmp_path / "x.db"), "--no-fundamentals"])
    assert code == 0
    conn = db.connect(tmp_path / "x.db")
    assert db.last_date(conn, "GOOD") is not None and db.last_date(conn, "BAD") is None

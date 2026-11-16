"""Weekly summary report -> a single self-contained HTML file.

    python -m stocklens.reports.weekly --db data/sample.db --transactions data/sample_transactions.csv
    python -m stocklens.reports.weekly --db data/stocklens.db --transactions data/private/transactions.csv

The file embeds its charts, so you can email it or open it offline.
(Plotly's JS is loaded from a CDN; for a fully offline report use include_plotlyjs=True.)
"""
from __future__ import annotations

import argparse
import html
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from stocklens.analysis import risk, valuation
from stocklens.context import Context, load_context
from stocklens.portfolio import returns as pret
from stocklens.portfolio import tracker

DISCLAIMER = ("Educational tool - not financial advice. Past performance does not predict future results. "
              "Figures come from free data sources that may contain errors.")

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1f2937}
h1{margin-bottom:0} h2{margin-top:2rem;border-bottom:1px solid #e5e7eb;padding-bottom:.3rem}
table{border-collapse:collapse;width:100%;font-size:.9rem} th,td{padding:.35rem .6rem;text-align:right;border-bottom:1px solid #eee}
th:first-child,td:first-child{text-align:left} .warn{background:#fff7ed;border-left:4px solid #f97316;padding:.5rem .8rem;margin:.4rem 0}
.ok{background:#f0fdf4;border-left:4px solid #22c55e;padding:.5rem .8rem} .muted{color:#6b7280;font-size:.85rem}
.kpis{display:flex;gap:1rem;flex-wrap:wrap} .kpi{background:#f9fafb;border-radius:8px;padding:.7rem 1rem;min-width:140px}
.kpi b{display:block;font-size:1.3rem}
"""


def _pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:+.1%}"


def _money(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:,.0f}"


def _kpi(label: str, value: str) -> str:
    return f'<div class="kpi">{html.escape(label)}<b>{html.escape(value)}</b></div>'


def _table(df: pd.DataFrame, fmt: dict) -> str:
    d = df.copy()
    for col, f in fmt.items():
        if col in d:
            d[col] = d[col].map(f)
    return d.to_html(classes="t", border=0, escape=True)


def week_movers(adj: pd.DataFrame, symbols: list[str], days: int = 5) -> pd.DataFrame:
    r = adj[symbols].iloc[-1] / adj[symbols].iloc[-1 - days] - 1
    return r.sort_values(ascending=False).to_frame("1-week return")


def build_report(ctx: Context) -> str:
    parts: list[str] = [f"<h1>StockLens weekly report</h1><p class='muted'>{date.today():%A %d %B %Y} &middot; data to {ctx.adj.index[-1]:%d %b %Y}</p>"]

    # --- portfolio ---
    if ctx.holdings is not None:
        s = tracker.summary(ctx.holdings)
        parts.append("<h2>Portfolio</h2><div class='kpis'>"
                     + _kpi("Market value", _money(s["market_value"]))
                     + _kpi("Unrealised P&L", f"{_money(s['unrealised_pnl'])} ({_pct(s['unrealised_pct'])})")
                     + _kpi("Realised P&L", _money(s["realised_pnl"]))
                     + _kpi("Dividends", _money(s["dividends"]))
                     + "</div>")
        open_ = ctx.holdings[ctx.holdings["quantity"] > 0][["quantity", "avg_cost", "last_price", "market_value", "unrealised_pnl", "unrealised_pct", "weight"]]
        parts.append(_table(open_, {"quantity": "{:,.0f}".format, "avg_cost": "{:,.2f}".format, "last_price": "{:,.2f}".format,
                                    "market_value": _money, "unrealised_pnl": _money, "unrealised_pct": _pct, "weight": "{:.0%}".format}))

        if ctx.portfolio_returns is not None and ctx.benchmark_returns is not None:
            cmp = pret.compare_to_benchmark(ctx.portfolio_returns, ctx.benchmark_returns)
            fig = go.Figure()
            fig.add_scatter(x=cmp["cumulative"].index, y=cmp["cumulative"]["portfolio"], name="Portfolio (time-weighted)")
            fig.add_scatter(x=cmp["cumulative"].index, y=cmp["cumulative"]["benchmark"], name=f"Benchmark ({ctx.benchmark})")
            fig.update_layout(height=340, yaxis_tickformat=".0%", margin=dict(t=20, b=20), legend=dict(orientation="h"))
            parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
            parts.append(f"<p>Since first purchase: portfolio {_pct(cmp['portfolio_total'])} vs benchmark {_pct(cmp['benchmark_total'])} "
                         f"(excess {_pct(cmp['excess_total'])}).</p>")

        # --- risk ---
        rs = risk.risk_summary(ctx.portfolio_returns, ctx.benchmark_returns) if ctx.portfolio_returns is not None else None
        if rs:
            parts.append("<h2>Risk</h2><div class='kpis'>"
                         + _kpi("Volatility (ann.)", f"{rs['ann_volatility']:.1%}")
                         + _kpi("Max drawdown", f"{rs['max_drawdown']:.1%}")
                         + _kpi("Sharpe", f"{rs['sharpe']:.2f}")
                         + _kpi("1-day VaR 95%", f"{rs['var_95_1d']:.1%}")
                         + _kpi("Beta", f"{rs.get('beta', float('nan')):.2f}")
                         + "</div>")
        conc = risk.concentration_report(ctx.holdings)
        if conc["warnings"]:
            parts += [f"<div class='warn'>&#9888; {html.escape(w)}</div>" for w in conc["warnings"]]
        else:
            parts.append("<div class='ok'>No concentration limits breached.</div>")

    # --- movers + screener ---
    syms = ctx.tradable
    if len(ctx.adj) > 6 and syms:
        parts.append("<h2>Watchlist: last 5 trading days</h2>")
        parts.append(_table(week_movers(ctx.adj, syms), {"1-week return": _pct}))
    if not ctx.fundamentals.empty and syms:
        sc = valuation.score_watchlist(ctx.adj[syms], ctx.fundamentals)
        parts.append("<h2>Screener (quality / value / momentum)</h2>")
        parts.append(_table(sc[["quality", "value", "momentum", "composite", "rank"]], {c: "{:.2f}".format for c in ["quality", "value", "momentum", "composite"]} | {"rank": "{:.0f}".format}))
        parts.append("<p class='muted'>Fundamentals are a snapshot of today, not point-in-time history. Scores rank this watchlist only.</p>")

    # --- data quality ---
    if not ctx.flags.empty:
        recent = ctx.flags.tail(10)
        parts.append("<h2>Data-quality flags (latest 10)</h2>" + recent.to_html(index=False, border=0, escape=True))

    parts.append(f"<hr><p class='muted'>{html.escape(DISCLAIMER)}</p>")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>StockLens weekly report</title><style>{CSS}</style></head><body>{''.join(parts)}</body></html>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/sample.db")
    ap.add_argument("--transactions", default="data/sample_transactions.csv")
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--out", default=f"reports_out/weekly_{date.today():%Y-%m-%d}.html")
    args = ap.parse_args()
    ctx = load_context(args.db, args.transactions, args.benchmark)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(ctx), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

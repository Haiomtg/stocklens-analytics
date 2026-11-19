import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import DISCLAIMER, get_context, pct
from stocklens.portfolio import returns as pret
from stocklens.portfolio import tracker

st.set_page_config(page_title="Portfolio - StockLens", layout="wide")
st.title("Portfolio")
st.caption(DISCLAIMER)
ctx = get_context()

if ctx.holdings is None:
    st.warning(ctx.portfolio_error or "No transactions file found. Add one in the sidebar (columns: date, symbol, type, quantity, price, fees).")
    st.stop()

s = tracker.summary(ctx.holdings)
c = st.columns(5)
c[0].metric("Market value", f"{s['market_value']:,.0f}")
c[1].metric("Unrealised P&L", f"{s['unrealised_pnl']:,.0f}", pct(s["unrealised_pct"]))
c[2].metric("Realised P&L", f"{s['realised_pnl']:,.0f}")
c[3].metric("Dividends", f"{s['dividends']:,.0f}")
c[4].metric("Positions", s["n_positions"])

if ctx.portfolio_returns is not None and ctx.benchmark_returns is not None:
    cmp = pret.compare_to_benchmark(ctx.portfolio_returns, ctx.benchmark_returns)
    fig = go.Figure()
    fig.add_scatter(x=cmp["cumulative"].index, y=cmp["cumulative"]["portfolio"], name="Portfolio (time-weighted)")
    fig.add_scatter(x=cmp["cumulative"].index, y=cmp["cumulative"]["benchmark"], name=f"Benchmark: {ctx.benchmark}")
    fig.update_layout(yaxis_tickformat=".0%", height=380, margin=dict(t=30), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    m = st.columns(4)
    m[0].metric("Portfolio (since first buy)", pct(cmp["portfolio_total"]))
    m[1].metric("Benchmark", pct(cmp["benchmark_total"]))
    m[2].metric("Excess return", pct(cmp["excess_total"]))
    m[3].metric("Beta vs benchmark", f"{cmp['beta']:.2f}")
    st.caption("Time-weighted return: unaffected by when you added money, so it is comparable with an index.")

st.subheader("Holdings")
open_ = ctx.holdings[ctx.holdings["quantity"] > 0]
st.dataframe(
    open_.style.format(
        {"quantity": "{:,.0f}", "avg_cost": "{:,.2f}", "cost_basis": "{:,.0f}", "last_price": "{:,.2f}",
         "market_value": "{:,.0f}", "unrealised_pnl": "{:,.0f}", "unrealised_pct": "{:+.1%}",
         "realised_pnl": "{:,.0f}", "dividends": "{:,.0f}", "total_pnl": "{:,.0f}", "weight": "{:.1%}"}
    ),
    width="stretch",
)
a, b = st.columns(2)
a.plotly_chart(px.pie(open_.reset_index(), names="symbol", values="market_value", title="By position"), width="stretch")
if "sector" in open_:
    b.plotly_chart(px.pie(open_.reset_index(), names="sector", values="market_value", title="By sector"), width="stretch")

with st.expander("Transactions"):
    st.dataframe(ctx.transactions, width="stretch", hide_index=True)
st.caption("Cost method: average cost. Splits and multi-currency are not modelled (see README, limitations).")

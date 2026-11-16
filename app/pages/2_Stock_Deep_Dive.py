import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import DISCLAIMER, get_context, pct
from stocklens.analysis import indicators, risk, valuation

st.set_page_config(page_title="Stock Deep-Dive - StockLens", layout="wide")
st.title("Stock Deep-Dive")
st.caption(DISCLAIMER)
ctx = get_context()

tab1, tab2 = st.tabs(["Single stock", "Watchlist screener"])

with tab1:
    syms = ctx.tradable or list(ctx.adj.columns)
    sym = st.selectbox("Symbol", syms)
    price = ctx.adj[sym].dropna()
    years = st.slider("History (years)", 1, 10, 3)
    ind = indicators.add_all(price).loc[price.index[-1] - __import__("pandas").DateOffset(years=years):]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.2, 0.25], vertical_spacing=0.03)
    for col, name in [("price", "Price"), ("sma50", "SMA 50"), ("sma200", "SMA 200")]:
        fig.add_scatter(x=ind.index, y=ind[col], name=name, row=1, col=1)
    fig.add_scatter(x=ind.index, y=ind["rsi14"], name="RSI 14", row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)
    fig.add_bar(x=ind.index, y=ind["hist"], name="MACD hist", row=3, col=1)
    fig.add_scatter(x=ind.index, y=ind["macd"], name="MACD", row=3, col=1)
    fig.add_scatter(x=ind.index, y=ind["signal"], name="Signal", row=3, col=1)
    fig.update_layout(height=650, margin=dict(t=20), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    r = price.pct_change()
    k = st.columns(4)
    k[0].metric("Volatility (ann.)", f"{risk.annualised_volatility(r):.1%}")
    k[1].metric("Max drawdown", f"{risk.max_drawdown(r):.1%}")
    k[2].metric("RSI 14", f"{ind['rsi14'].iloc[-1]:.0f}")
    k[3].metric("1-year return", pct(price.iloc[-1] / price.iloc[-253] - 1) if len(price) > 253 else "n/a")

    if sym in ctx.fundamentals.index:
        st.subheader("Valuation vs peers")
        peers = valuation.peer_comparison(ctx.fundamentals, sym)
        st.dataframe(peers.style.format({"value": "{:.3f}", "peer_median": "{:.3f}", "vs_peers": "{:+.0%}"}), width="stretch")
        st.caption(f"Compared with {peers.attrs['n_peers']} {peers.attrs['scope']}. Fundamentals are today's snapshot.")

    if ctx.holdings is not None:
        held = [s for s in ctx.holdings.index[ctx.holdings["quantity"] > 0] if s != sym and s in ctx.adj]
        if held:
            st.subheader("Correlation with what you already hold")
            corr = ctx.adj[held].pct_change().corrwith(r).sort_values(ascending=False)
            st.bar_chart(corr)
            st.caption("High correlation means this stock would add little diversification to your portfolio.")

with tab2:
    st.write("Rank the watchlist by **quality** (high ROE, low debt), **value** (low P/E, P/B) and **momentum** (12-1 month return).")
    if ctx.fundamentals.empty:
        st.info("No fundamentals stored yet.")
    else:
        w = st.columns(3)
        wq, wv, wm = (w[0].slider("Quality weight", 0.0, 1.0, 0.33), w[1].slider("Value weight", 0.0, 1.0, 0.33), w[2].slider("Momentum weight", 0.0, 1.0, 0.34))
        if wq + wv + wm == 0:
            st.warning("Set at least one weight above zero.")
        else:
            sc = valuation.score_watchlist(ctx.adj[ctx.tradable], ctx.fundamentals, {"quality": wq, "value": wv, "momentum": wm})
            st.dataframe(sc.style.format({c: "{:.2f}" for c in ["quality", "value", "momentum", "composite"]} | {"momentum_return": "{:+.1%}", "rank": "{:.0f}"}), width="stretch")
            st.caption("Scores are percentile ranks WITHIN this watchlist (1 = best). A ranking tool for research, not a buy list.")

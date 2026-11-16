import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from common import DISCLAIMER, get_context

st.set_page_config(page_title="StockLens Analytics", page_icon="📈", layout="wide")
st.title("📈 StockLens Analytics")
st.caption(DISCLAIMER)

ctx = get_context()

st.markdown(
    """
**Pages** (left sidebar): **Portfolio** - holdings, P&L, vs benchmark - **Stock Deep-Dive** - indicators,
peers, screener - **Risk** - drawdown, VaR, correlation, concentration - **Backtests** - strategies with
costs and an honest ML experiment.
"""
)

c1, c2, c3 = st.columns(3)
c1.metric("Symbols in database", len(ctx.symbols))
c2.metric("Latest data", f"{ctx.adj.index[-1]:%d %b %Y}")
c3.metric("Open data-quality flags", len(ctx.flags))

st.subheader("Data coverage")
st.dataframe(ctx.symbols, width="stretch", hide_index=True)

if not ctx.flags.empty:
    st.subheader("Data-quality flags")
    st.caption("Checks only FLAG issues (gaps, outliers, suspected unadjusted splits, stale data). They never silently change your data.")
    st.dataframe(ctx.flags, width="stretch", hide_index=True)

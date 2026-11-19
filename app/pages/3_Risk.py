import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import DISCLAIMER, get_context
from stocklens.analysis import risk

st.set_page_config(page_title="Risk - StockLens", layout="wide")
st.title("Risk")
st.caption(DISCLAIMER)
ctx = get_context()

if ctx.portfolio_returns is None or ctx.holdings is None:
    st.warning(ctx.portfolio_error or "Risk needs a transactions file (see sidebar).")
    st.stop()

rs = risk.risk_summary(ctx.portfolio_returns, ctx.benchmark_returns)
c = st.columns(6)
c[0].metric("Return (ann.)", f"{rs['ann_return']:.1%}")
c[1].metric("Volatility (ann.)", f"{rs['ann_volatility']:.1%}")
c[2].metric("Sharpe (rf=0)", f"{rs['sharpe']:.2f}")
c[3].metric("Max drawdown", f"{rs['max_drawdown']:.1%}")
c[4].metric("1-day VaR 95%", f"{rs['var_95_1d']:.1%}")
c[5].metric("Beta", f"{rs.get('beta', float('nan')):.2f}")
st.caption("VaR is historical: on 95% of past days the portfolio lost less than this. It says nothing about how bad the other 5% can be - see expected shortfall: "
           f"**{rs['es_95_1d']:.1%}** average loss on those days.")

dd = risk.drawdown_series(ctx.portfolio_returns)
fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", name="Drawdown"))
fig.update_layout(title="Drawdown from previous peak", yaxis_tickformat=".0%", height=320, margin=dict(t=40))
st.plotly_chart(fig, width="stretch")

st.subheader("Diversification check")
conc = risk.concentration_report(ctx.holdings, max_position=st.slider("Max single position", 0.05, 0.6, 0.25, 0.05),
                                 max_sector=st.slider("Max single sector", 0.2, 1.0, 0.5, 0.05))
if conc["warnings"]:
    for w in conc["warnings"]:
        st.warning(w)
else:
    st.success("No concentration limits breached.")
st.write(f"Effective number of equal-sized positions: **{conc['effective_n']:.1f}** (HHI {conc['hhi']:.2f}; more is more diversified).")
if len(conc["by_sector"]):
    st.bar_chart(conc["by_sector"])

st.subheader("Correlation between holdings")
held = [s for s in ctx.holdings.index[ctx.holdings["quantity"] > 0] if s in ctx.adj]
if len(held) >= 2:
    corr = risk.correlation_matrix(ctx.adj[held].pct_change().iloc[-504:])
    st.plotly_chart(px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f", aspect="auto"), width="stretch")
    st.caption("Last ~2 years of daily returns. Correlations rise in market crashes, so diversification is weaker exactly when you need it.")
else:
    st.info("Need at least two open positions.")

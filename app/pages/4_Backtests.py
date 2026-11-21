import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import DISCLAIMER, get_context
from stocklens.models import backtest as bt
from stocklens.models import forecasting as fc

st.set_page_config(page_title="Backtests - StockLens", layout="wide")
st.title("Backtests & ML experiment")
st.caption(DISCLAIMER)
st.warning("Backtests flatter strategies. Signals trade the NEXT day and costs are charged, but results come from one historical path, "
           "parameters are fitted to it, and the watchlist has survivorship bias. Use this to learn, not to size real trades.")
ctx = get_context()
tab1, tab2, tab3 = st.tabs(["Single-stock strategy", "Cross-sectional momentum", "ML direction forecast"])

with tab1:
    sym = st.selectbox("Symbol", ctx.tradable)
    price = ctx.adj[sym].dropna()
    strat = st.radio("Strategy", ["Moving-average crossover", "Time-series momentum"], horizontal=True)
    cost = st.slider("Transaction cost (bps per trade, each way)", 0, 100, 20, help="20 bps = 0.20% of the traded value: brokerage + spread")
    if strat == "Moving-average crossover":
        f = st.slider("Fast SMA", 5, 100, 50)
        s = st.slider("Slow SMA", 20, 300, 200)
        pos = bt.ma_crossover_position(price, f, s) if f < s else None
        if pos is None:
            st.error("Fast must be smaller than slow.")
            st.stop()
    else:
        lb = st.slider("Lookback (days)", 20, 300, 126)
        pos = bt.time_series_momentum_position(price, lb)

    res, bh = bt.run_backtest(price, pos, cost), bt.buy_and_hold(price, cost)
    fig = go.Figure()
    fig.add_scatter(x=res.returns.index, y=(1 + res.returns).cumprod(), name="Strategy (after costs)")
    fig.add_scatter(x=bh.returns.index, y=(1 + bh.returns).cumprod(), name="Buy & hold")
    fig.update_layout(height=380, margin=dict(t=20), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    table = pd.DataFrame({"Strategy": res.metrics, "Buy & hold": bh.metrics}).drop(index="n_days")
    st.dataframe(table.style.format("{:.3f}"), width="stretch")

    if strat == "Moving-average crossover":
        st.subheader("Is it robust, or did you just find noise?")
        sweep = bt.parameter_sweep(price, cost_bps=cost)
        st.write("Sharpe ratio for every parameter pair - a real edge works across neighbours, not at one lucky cell:")
        st.dataframe(sweep.pivot(index="fast", columns="slow", values="sharpe").style.format("{:.2f}", na_rep="-"), width="stretch")
        oos = bt.in_sample_vs_out_of_sample(price, cost_bps=cost)
        st.write(f"Best pair picked on the first 60% of history: **{oos['chosen_fast']}/{oos['chosen_slow']}** "
                 f"(in-sample Sharpe {oos['in_sample_sharpe']:.2f}). On the unseen last 40% (from {oos['oos_start']:%b %Y}): "
                 f"Sharpe **{oos['oos_sharpe']:.2f}** vs buy & hold {oos['oos_buy_hold_sharpe']:.2f}.")

with tab2:
    if len(ctx.tradable) < 3:
        st.info("Cross-sectional momentum needs at least 3 symbols in the database.")
    else:
            st.write("Each month-end, hold the top-N symbols by past return, equal weight. Compared with holding all of them equally.")
            top_n = st.slider("Top N", 1, max(1, len(ctx.tradable) - 1), min(3, len(ctx.tradable)))
            lb2 = st.slider("Lookback (days)", 20, 300, 126, key="lb2")
            cost2 = st.slider("Transaction cost (bps)", 0, 100, 20, key="c2")
            px_ = ctx.adj[ctx.tradable]
            w = bt.cross_sectional_momentum_weights(px_, lb2, top_n)
            r = bt.run_portfolio_backtest(px_, w, cost2)
            ew = bt.run_portfolio_backtest(px_, pd.DataFrame(1 / px_.shape[1], index=px_.index, columns=px_.columns), cost2)
            fig = go.Figure()
            fig.add_scatter(x=r.returns.index, y=(1 + r.returns).cumprod(), name=f"Top-{top_n} momentum")
            fig.add_scatter(x=ew.returns.index, y=(1 + ew.returns).cumprod(), name="Equal-weight all")
            fig.update_layout(height=380, margin=dict(t=20), legend=dict(orientation="h"))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(pd.DataFrame({"Momentum": r.metrics, "Equal-weight": ew.metrics}).drop(index="n_days").style.format("{:.3f}"), width="stretch")

with tab3:
    st.write("Can a model predict whether a stock is higher in ~1 month better than a dumb baseline? "
             "Walk-forward validation, purge gap, two baselines, and an honest verdict.")
    sym3 = st.selectbox("Symbol", ctx.tradable, key="sym3")
    if st.button("Run experiment"):
        with st.spinner("Training walk-forward folds..."):
            try:
                out = fc.run_experiment(ctx.adj[sym3].dropna())
            except ValueError as e:
                st.error(str(e))
                st.stop()
        st.info(out["verdict"])
        st.dataframe(out["summary"].style.format({"accuracy": "{:.3f}", "balanced_accuracy": "{:.3f}", "vs_always_up_pp": "{:+.1f}", "p_value_vs_always_up": "{:.2f}"}), width="stretch")
        st.caption(f"{out['n_folds']} walk-forward folds, {out['n_test']} test days, share of up-months {out['base_rate_up']:.0%}. "
                   "p-values use non-overlapping samples and a 2-model correction.")
        st.line_chart(out["fold_accuracy"])

    st.divider()
    st.subheader("Run on all watchlist stocks")
    st.write("One stock can look good by luck. Testing all of them, with the p-values corrected for the number of "
             "stocks tried, is the honest way to read the result. This takes a minute or two.")
    if st.button("Run on all watchlist stocks"):
        bar = st.progress(0.0, text="Starting...")
        table = fc.run_experiment_many(
            ctx.adj[ctx.tradable],
            progress=lambda i, n, sym: bar.progress(i / n, text=f"Testing {sym} ({i + 1}/{n})"),
        )
        bar.empty()
        st.session_state["ml_all"] = table
    if "ml_all" in st.session_state:
        table = st.session_state["ml_all"]
        st.info(fc.summarize_many(table))
        fmt = {c: "{:.3f}" for c in ["baseline_acc", "logistic_acc", "forest_acc", "p_adj_models", "p_adj_all"] if c in table}
        fmt["best_vs_baseline_pp"] = "{:+.1f}"
        st.dataframe(table.style.format(fmt, na_rep="-"), width="stretch")
        st.caption("p_adj_models: best model's p-value x2 (two models tried). p_adj_all: that x number of stocks (Bonferroni). "
                   "Compare to 0.05. Stocks are not independent (they move together), so even this correction is only a guide.")

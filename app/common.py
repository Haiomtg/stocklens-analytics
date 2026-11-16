"""Shared helpers for every Streamlit page: data location, loading, disclaimer."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from stocklens.context import Context, load_context

ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = "Educational tool - not financial advice. Data is from free sources and may contain errors."


def _default_db() -> str:
    env = os.environ.get("STOCKLENS_DB")
    if env:
        return env
    real = ROOT / "data" / "stocklens.db"
    return str(real if real.exists() else ROOT / "data" / "sample.db")


def _default_tx() -> str:
    env = os.environ.get("STOCKLENS_TRANSACTIONS")
    if env:
        return env
    private = ROOT / "data" / "private" / "transactions.csv"
    return str(private if private.exists() else ROOT / "data" / "sample_transactions.csv")


@st.cache_resource(show_spinner="Loading data...")
def _load(db_path: str, tx_path: str, benchmark: str | None, mtime: float) -> Context:
    return load_context(db_path, tx_path, benchmark)


def get_context() -> Context:
    """Sidebar: choose data files and benchmark; returns the loaded Context."""
    with st.sidebar:
        st.header("Data")
        db_path = st.text_input("Database", _default_db(), key="db_path")
        tx_path = st.text_input("Transactions CSV", _default_tx(), key="tx_path")
        if "sample" in Path(db_path).name:
            st.info("Showing SYNTHETIC sample data. Point the paths at your own files to use real data.")
        st.caption(DISCLAIMER)
    if not Path(db_path).exists():
        st.error(f"Database not found: {db_path}\n\nBuild the demo data with `python -m stocklens.ingest.sample`.")
        st.stop()
    mtime = Path(db_path).stat().st_mtime
    ctx = _load(db_path, tx_path, st.session_state.get("benchmark"), mtime)
    with st.sidebar:
        options = list(ctx.adj.columns)
        idx = options.index(ctx.benchmark) if ctx.benchmark in options else 0
        choice = st.selectbox("Benchmark", options, index=idx)
        if choice != ctx.benchmark:
            st.session_state["benchmark"] = choice
            st.rerun()
    return ctx


def pct(x: float) -> str:
    return "n/a" if x != x else f"{x:+.1%}"

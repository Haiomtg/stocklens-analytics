# StockLens Analytics

> **Educational tool - not financial advice.** Past performance does not predict future results. Data comes from free sources and may contain errors.

A Python analytics backend for ASX, US and Vietnamese equities: automated data pipeline, personal portfolio tracker, risk analytics, backtesting with transaction costs, and an honest walk-forward ML experiment - with a Streamlit dashboard and a weekly HTML report.

It is the numbers layer that sits behind **StockLens** (my document-reading app that extracts figures from company reports).

**Live demo:** _<add your Streamlit Community Cloud link here>_ (runs on synthetic sample data - no real holdings are published)

<!-- Add 2-3 screenshots here: Portfolio page, Risk page, Backtests page -->

## Headline result

> _<Fill this in after Stage 5 - one measurable sentence. Example format:>_
> _"A 50/200-day moving-average crossover on [symbol], after 20 bps costs, returned X% vs Y% buy-and-hold (Sharpe A vs B); chosen on 2018-2022 data, it returned ... out of sample."_
> _"Logistic regression and random forest did / did not beat the always-up baseline on next-month direction (accuracy A% vs B%, p = ...)."_

## What it does

| Area | Features |
|---|---|
| Data | Daily prices, dividends and fundamentals for US/ASX (Yahoo Finance) and Vietnam (vnstock) into SQLite; incremental updates; data-quality checks (gaps, outliers, suspected unadjusted splits, stale data) |
| Portfolio | Transactions CSV -> holdings, average cost, realised / unrealised P&L, dividends, time-weighted return vs a benchmark |
| Analysis | SMA / EMA / RSI / MACD / volatility / Bollinger; valuation vs sector peers; quality-value-momentum watchlist screener |
| Risk | Volatility, max drawdown, Sharpe, beta, correlation matrix, historical VaR and expected shortfall, concentration warnings |
| Models | Vectorised backtester (next-day execution, costs), parameter sweep, in-sample vs out-of-sample test, cross-sectional momentum; walk-forward direction forecasting vs two baselines |
| Output | 4-page Streamlit dashboard, weekly self-contained HTML report |

## Quick start (demo data, no internet needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m stocklens.ingest.sample          # builds data/sample.db + data/sample_transactions.csv (synthetic)
streamlit run app/Home.py                  # dashboard
python -m stocklens.reports.weekly         # reports_out/weekly_<date>.html
pytest                                     # test suite
```

## Using your real data (stays on your computer)

```bash
pip install -e ".[data]"                                   # adds yfinance (vnstock: see note in ingest/vietnam.py)
python -m stocklens.ingest.update                           # fills data/stocklens.db (git-ignored)
cp data/sample_transactions.csv data/private/transactions.csv   # then replace with YOUR trades (git-ignored)
streamlit run app/Home.py
```

Transactions CSV columns: `date, symbol, type, quantity, price, fees`, where `type` is `BUY`, `SELL` or `DIVIDEND` (for dividends: `quantity` = shares held, `price` = cash dividend per share).
`data/private/` and `data/stocklens.db` are in `.gitignore`: **never commit real holdings.**

## Design decisions

- **Next-day execution + costs in every backtest.** A signal from today's close trades tomorrow; every position change pays `cost_bps`. `tests/test_models.py` proves a hindsight-perfect signal cannot profit from the same day's move.
- **Walk-forward validation with a purge gap.** Time series are never shuffled; training stops `horizon` days before each test block because labels look forward.
- **Baselines first.** The ML model must beat "always predict up" and "last month repeats". p-values use non-overlapping samples (overlapping monthly labels would overstate significance) and a 2-model correction.
- **Fetchers behind an interface.** Free data sources break; each is one small class that can be swapped.
- **Time-weighted returns** so the portfolio-vs-benchmark comparison is not distorted by when I add money.

## Limitations (read these)

- **Fundamentals are today's snapshot, not point-in-time**, so they are used for the screener only and never in backtests (that would be look-ahead bias).
- **Survivorship bias:** the watchlist is stocks that exist today; failed or delisted companies are absent, which flatters backtests.
- **One historical path.** Backtest parameters are fitted to it; the sweep and out-of-sample test show robustness but do not prove an edge.
- Average-cost accounting only; **stock splits and multi-currency are not modelled** in the portfolio tracker.
- Free data (Yahoo, vnstock) is unofficial, can be wrong, and can change or disappear.
- The ML result is research, **not a trading signal**.

## Project layout

```
src/stocklens/   ingest/ portfolio/ analysis/ models/ reports/   (db.py, context.py)
app/             Streamlit dashboard (Home.py + pages/)
tests/           pytest suite (maths checked against hand calculations)
docs/            BUILD_GUIDE.md - stage-by-stage plan
.github/         CI tests + daily pipeline health check
```

## License

MIT - see `LICENSE`.

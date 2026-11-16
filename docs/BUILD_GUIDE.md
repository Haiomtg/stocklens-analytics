# BUILD GUIDE - what is done, and what YOU need to do at each stage

Read this top to bottom once. Then work one stage at a time.

## How to read the status labels

- **DONE & TESTED** - code written and checked by automated tests on synthetic data (33 tests pass).
- **DONE, NOT TESTED LIVE** - code written, but it needs real internet/real data, which could not be tried while building it. **You must run it and fix small differences.**
- **YOURS** - only you can do this (your real data, accounts, decisions, writing).

## The honest picture

| Part | Status |
|---|---|
| SQLite database layer, sample-data generator | DONE & TESTED |
| Data-quality checks (gaps, outliers, splits, stale) | DONE & TESTED |
| Yahoo fetcher (US + ASX) | DONE, NOT TESTED LIVE |
| Vietnam fetcher (`vnstock`) | DONE, NOT TESTED LIVE - most likely to need fixes |
| Daily update job | DONE & TESTED (with fake fetchers) |
| Portfolio tracker, P&L, time-weighted return, benchmark | DONE & TESTED (hand-calculated examples) |
| Indicators, valuation, screener, risk metrics | DONE & TESTED |
| Backtester, parameter sweep, out-of-sample test | DONE & TESTED (including a test that proves there is no look-ahead) |
| Walk-forward ML experiment | DONE & TESTED |
| Streamlit dashboard (4 pages) | DONE, runs without errors on sample data. Not yet looked at by a human - YOUR eyes needed |
| Weekly HTML report | DONE & TESTED |
| GitHub Actions (CI + daily health check) | WRITTEN, runs only after you push to GitHub |
| Docker, PostgreSQL, Monte Carlo, optimisation, PDF reports | **Deliberately not built** (see the end) |

## Important rule: you must be able to explain every file

Employers will ask "how does this work?". I wrote the first version so you have a working base, but a project you cannot explain is worse than no project. For each stage below there are **"Explain it"** questions. Do not move on until you can answer them in your own words. A good way to learn: change one thing, break it on purpose, and see which test fails.

## Time plan (evenings and weekends, around work and study)

Aim to finish before Semester 1 2027 starts (capstone + internship). That gives you about 10 to 12 weeks. A realistic plan is **8-10 weeks**:

| Weeks | Stage |
|---|---|
| 1 | Stage 0 + Stage 1 (setup, real data) |
| 2-3 | Stage 2 (your portfolio) |
| 3-4 | Stage 3 (analysis + risk, real fundamentals) |
| 5-6 | Stage 4 (ML + backtests on real data, write-up) |
| 7 | Stage 5 (dashboard review + deploy) |
| 8 | Stage 6 (polish, README, resume) |

If you fall behind, cut from the "optional" lists - never skip Stage 4's honest write-up, it is the most valuable part for employers.

---

## Stage 0 - Set up and see it work (about 2 hours)

**Done for you:** the whole repository, packaging, `.gitignore` (blocks your private data), MIT licence, sample data.

**You do:**

1. Unzip `stocklens-analytics.zip` where you keep projects (for example `~/Projects/`).
2. In Terminal:
   ```bash
   cd ~/Projects/stocklens-analytics
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,data]"
   python -m stocklens.ingest.sample
   pytest
   streamlit run app/Home.py
   ```
   Python 3.11 or newer is required (the project reads TOML files with the standard library).
3. Click through all 5 pages in the browser. Write down anything that looks wrong or confusing. (The data is fake random walks, so the numbers mean nothing - you are checking that the pages work.)
4. Create a **private** GitHub repo first (you can make it public at the end), then:
   ```bash
   git init && git add . && git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your repo URL>
   git push -u origin main
   ```
   Before you push, run `git status` and confirm nothing from `data/private/` or `data/stocklens.db` is listed.

**Definition of done:** `pytest` shows all green, the dashboard opens, the repo is on GitHub, and the CI badge/tab shows a passing run.

**Explain it:** What is `pip install -e .`? Why are the sample tickers called `SYN...`? Why does `.gitignore` list `data/private/*`?

---

## Stage 1 - Real data (Week 1)

**Done for you:** `ingest/yahoo.py`, `ingest/vietnam.py`, `ingest/quality.py`, `ingest/update.py`, `config/watchlist.toml`, SQLite schema in `db.py`.

**You do:**

1. **Edit `config/watchlist.toml`** - put the 10 to 20 stocks you actually follow. US = `AAPL`, ASX = `BHP.AX`, Vietnam = plain ticker (`VRE`, `GAS`, ...). Keep the benchmarks section.
2. **Test the Yahoo fetcher:** `python -m stocklens.ingest.update --no-fundamentals`. Look at the log. Each symbol should say `N rows upserted`. If a symbol fails, read the error - usually a wrong ticker (ASX stocks need `.AX`).
3. **Vietnam data - `vnstock` is quarantined on PyPI right now (checked 16 Nov 2026), so it cannot be installed.** Skip `vnstock` for now: US/ASX data works without it. For Vietnamese stocks, the safest option is a small `CsvFetcher` (same three methods as the other fetchers) that reads price files you export yourself - ask Claude to write it. Re-check https://pypi.org/project/vnstock/ later; do not install it from other places while it is under security review. If it comes back, then **test and fix the Vietnam fetcher.** Run `python -m stocklens.ingest.vietnam VRE`. The `vnstock` package has changed its interface several times. If it errors, open `ingest/vietnam.py`, read the docstring at the top (it lists exactly what to check: column names, price units), and fix the 5 or 6 lines. If you get stuck, paste the error to Claude. Also decide: VCI prices are in thousands of VND (25.5 = 25,500 VND). **Pick one convention and use it in your transactions file too.**
4. Check VN-Index works as a benchmark (`VNINDEX`). If vnstock does not return it for your version, find the right call for index history and add a small branch in `vietnam.py`.
5. Open the dashboard Home page with your real database (`streamlit run app/Home.py`, change the Database path in the sidebar to `data/stocklens.db`). Look at the **data-quality flags**. For each one decide: real event (e.g. a real 30% drop), data error, or unadjusted split. Write 3 or 4 examples in a notes file - they make a great README paragraph.
6. **Schedule your own daily update** (your real data stays on your computer). On a Mac, the simplest way is `cron`:
   ```bash
   crontab -e
   # add this line (change the path), runs 7:30pm on weekdays:
   30 19 * * 1-5 cd ~/Projects/stocklens-analytics && .venv/bin/python -m stocklens.ingest.update >> update.log 2>&1
   ```
   The Mac must be awake at that time. If it is often asleep, run the command by hand once a day instead.
7. **GitHub Actions daily job** (`.github/workflows/daily-update.yml`): after you push, go to the Actions tab and run it once with "Run workflow". Its job is a **pipeline health check** on the public watchlist - it proves the fetchers still work. It does *not* commit the database (a changing binary file would bloat your repo) and never touches your private trades.

**Definition of done:** `data/stocklens.db` holds at least 3 years of prices for your whole watchlist plus both benchmarks; the update command can be re-run safely; you have looked at every quality flag.

**Explain it:** Why store `close` and `adj_close` separately? Why is the update "incremental" and why does it re-fetch the last 7 days? Why does one failing ticker not stop the job? What is a *survivorship bias* problem in a watchlist?

**Warning:** Yahoo Finance's terms limit the data to personal use. That is why the public demo ships synthetic data. Do not publish downloaded price files in your repo.

---

## Stage 2 - Your portfolio (Weeks 2-3)

**Done for you:** `portfolio/tracker.py` (average-cost holdings, realised/unrealised P&L, dividends), `portfolio/returns.py` (daily value, time-weighted return, benchmark comparison), tests with hand-calculated examples.

**You do:**

1. Create `data/private/transactions.csv` (this folder is git-ignored). Copy the header from `data/sample_transactions.csv`:
   `date,symbol,type,quantity,price,fees` with `type` = `BUY`, `SELL` or `DIVIDEND`. For `DIVIDEND`: `quantity` = shares you held, `price` = dividend per share.
2. Enter every trade you have made, with the **same ticker names as your watchlist** (e.g. `BHP.AX`, `VRE`).
3. **Use one currency per file** (limitation of v1). Easiest: keep ASX/US trades in one file, VN trades in another (`data/private/transactions_vn.csv`), and switch files in the dashboard sidebar. Do not mix AUD and VND prices in one file.
4. Open the **Portfolio** page. **Check the numbers against your broker** for each holding: quantity, average cost, unrealised P&L. If something is different, find out why (fees? a split? a dividend you forgot?). This is the most important checking step in the whole project.
5. **Extension (YOURS, optional but impressive):** handle stock splits/bonus shares. Add a `SPLIT` transaction type in `tracker.py` (quantity multiplies, total cost stays the same) and a test with a hand-calculated example. Vietnamese stocks often pay bonus shares, so this is a real problem you will meet.
6. **Extension (optional):** multi-currency - add an FX table to the database and convert VND to AUD for the combined view.

**Definition of done:** your holdings match your broker to the cent (or you know exactly why not), and the portfolio-vs-benchmark chart looks sensible.

**Explain it:** What is average cost vs FIFO? What is a *time-weighted return* and why use it against an index? Walk through `test_twr_ignores_timing_of_deposits`. Why do we sort same-day BUY before SELL?

---

## Stage 3 - Analysis and risk on real stocks (Weeks 3-4)

**Done for you:** `analysis/indicators.py`, `analysis/valuation.py` (peer comparison + screener), `analysis/risk.py`, and the Stock Deep-Dive and Risk pages.

**You do:**

1. **Fundamentals:** check that the `fundamentals` table is filled for US/ASX symbols (`sqlite3 data/stocklens.db "select * from fundamentals limit 5"`). Yahoo units change over time - check one stock by hand: is `roe` a fraction (0.15 = 15%)? is `debt_to_equity` a ratio? is `dividend_yield` a fraction? Fix the conversions in `yahoo.py` if not.
2. **Vietnam fundamentals:** `VietnamFetcher.fetch_fundamentals` is an empty TODO. Pull P/E, P/B, ROE and debt/equity from vnstock (or your own StockLens figures) and map them to the same columns. Without this, VN stocks have no valuation/screener scores. This is your genuine edge - you already analyse VN companies.
3. Choose **sector labels** for VN stocks (the screener's peer comparison uses `sector`).
4. Open **Risk**: read the concentration warnings about your real portfolio. Decide if the limits (25% per stock, 50% per sector) suit you - they are sliders.
5. Read the VaR caption on the Risk page. **Make sure you can explain why VaR 95% does not tell you the worst case** (that is what expected shortfall is for).

**Definition of done:** every watchlist stock has fundamentals; the screener ranks them sensibly (sanity-check: does the top-ranked stock match your own view for obvious reasons?).

**Explain it:** What does RSI measure? What is beta and what does beta = 1.4 mean? Why is `momentum_12_1` skipping the last month? Why must fundamentals NOT be used in backtests (look-ahead bias)?

---

## Stage 4 - Backtests and the ML experiment (Weeks 5-6) - the most important stage

**Done for you:** `models/backtest.py` (next-day execution, costs, parameter sweep, in-sample/out-of-sample test, cross-sectional momentum) and `models/forecasting.py` (walk-forward logistic regression and random forest vs two baselines), plus tests that prove there is no look-ahead leak.

**You do:**

1. Pick **one market and 5 to 10 symbols** with at least 6 years of data. (The ML experiment needs about 4 years minimum: 3 years to train, then test blocks.)
2. Run the **Backtests** page for 2 or 3 stocks. Look at the **parameter sweep**: if only one cell is good, it is noise. Look at the **out-of-sample** line: that is the number to believe.
3. Choose realistic **costs** for your broker. Example: brokerage $X per trade on an average $Y trade, plus spread. Convert to basis points (bps) and write the calculation in your README. A cost of 0 is dishonest.
4. Run the **ML experiment** for each symbol (or `run_experiment_many` in a notebook). Put the results into a table.
5. **Decide your headline result honestly.** The most likely outcome is "no model clearly beat the baseline". That is fine and respected. **Do not** try different features until something "works" - that is p-hacking. If you want to try new features, write down beforehand what you will try, and apply a correction for the number of attempts.
6. Create `notebooks/01_eda.ipynb` (price/return distributions, correlation, stationarity) and `notebooks/02_model_experiments.ipynb` (your experiments, plots of fold accuracy). Notebooks are where you show your thinking.
7. Write the README **Headline result** (2 sentences) and a "What didn't work" section.
8. **Optional extras** (only if ahead of schedule): Monte Carlo simulation of portfolio outcomes; mean-variance optimisation with a cap on weights (be honest that optimisers are unstable); a `statsmodels` check, e.g. ADF test on returns vs prices.

**Definition of done:** one measurable sentence for the README about the strategy (after costs, vs buy-and-hold, with out-of-sample figures) and one about the ML experiment (vs baseline, with p-value), and a "limitations" paragraph you wrote yourself.

**Explain it (interviewers love these):** Why can't you shuffle time-series data for cross-validation? What is a *purge gap*? Why compare to "always up"? What does `shift(1)` in the backtester prevent? What is overfitting in a backtest? What is the difference between the in-sample and out-of-sample Sharpe in your result?

---

## Stage 5 - Dashboard and deployment (Week 7)

**Done for you:** 4 pages + Home, plus the weekly report generator.

**You do:**

1. Use every page with your **real** data on your own computer. Fix anything confusing (labels, formats). Add anything you personally miss (for example a "biggest movers today" table).
2. Take **3 screenshots from the demo (sample data) version** - never from real holdings.
3. Deploy the demo on **Streamlit Community Cloud** (free): sign in with GitHub, "New app", pick your repo, main file `app/Home.py`. It installs from `requirements.txt` and reads the committed `data/sample.db`. Make sure `data/sample.db` and `data/sample_transactions.csv` are committed (they are not git-ignored). Paste the URL into the README.
4. Generate your weekly report: `python -m stocklens.reports.weekly --db data/stocklens.db --transactions data/private/transactions.csv`. Open the HTML in a browser; check numbers; use it weekly for a month.
5. Optional: schedule the weekly report with `cron` (Sunday evening).

**Definition of done:** a public demo link works on your phone; the weekly report opens correctly.

---

## Stage 6 - Polish, README, resume (Week 8)

**You do:**

1. Fill in the README placeholders: demo link, 2-3 screenshots, headline result. Keep the **Limitations** section - it signals maturity.
2. Run `pytest` and make sure CI is green; add a small CI badge to the top of the README if you like.
3. Add type hints/docstrings where you changed code. Delete anything you cannot explain.
4. Record a 2-3 minute screen video: problem -> dashboard -> one honest result. Post it (LinkedIn or YouTube - this also helps your Creator HQ channel, and a Vietnamese-language walkthrough would be unique).
5. **Make the repo public** only after checking `git log` and the file list for any private file. Search the whole repo for your real tickers and account details.
6. **Resume:** come back to the Resume project and ask for the project entry. Give Claude your final numbers; it will turn them into bullets for the **professional track** (e.g. "Built a Python analytics pipeline ... backtested ... walk-forward validation ... 33 automated tests"). Do not list numbers you cannot defend in an interview.

**Definition of done:** a stranger can open the README, understand the problem in 30 seconds, see the demo, and run it in 5 minutes.

---

## What was deliberately NOT built (and why)

| Item | Why left out | When to add |
|---|---|---|
| PostgreSQL | SQLite is enough for one user and keeps setup simple. `db.py` is the only file to change. | If you want to learn it after Stage 6 |
| Docker | Late-stage polish; adds little before the project works | After Stage 6, an hour of work: Dockerfile that runs Streamlit |
| PDF report | HTML is easier, offline, and prints to PDF from the browser | Never needed |
| Monte Carlo / mean-variance optimiser | Easy to do badly; nice-to-have only | Stage 4 extras |
| Dual AUD/VND currency | Adds FX edge cases; separate files avoid them | Stage 2 extension |
| Splits in the tracker | Needs your real examples to test properly | Stage 2 extension |

## Rules to protect yourself

- **Never commit** `data/private/`, `data/stocklens.db`, screenshots of real holdings, or broker statements.
- **Never trade** from the ML output or a backtest alone. The project is for learning and for showing skills.
- Re-check any number you will publish (README, resume, LinkedIn) by recomputing it a second way.
- If a data source breaks, fix the fetcher; do not hand-edit the database.

## When you get stuck

Open the Resume project's chat or any Claude chat and paste: (1) the command you ran, (2) the full error, (3) the file you edited. Ask Claude to explain the fix so you learn it, not just copy it.

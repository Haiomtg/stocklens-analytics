"""Next-month direction forecasting with walk-forward validation and honest baselines.

Question: given only information available today, can a model predict whether the stock
will be higher in `horizon` trading days (~1 month) better than a dumb baseline?

Why it is set up this way:
  * Walk-forward (expanding window): always train on the past, test on the future. Never
    shuffle - shuffled cross-validation leaks the future into training on time series.
  * Purge gap: labels look `horizon` days ahead, so the last `horizon` training rows overlap
    the test period. We stop training `horizon` days before each test block.
  * Baselines: "always predict up (the training majority class)" and "last month's direction
    repeats". Stock direction is close to a coin flip, so beating 50% proves nothing.
  * Overlapping labels inflate significance, so the p-value uses non-overlapping samples.

The most likely outcome on real prices is "no model clearly beats the baseline". That is a
legitimate, publishable result. Report it as such.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stocklens.analysis import indicators

FEATURES = ["ret_1", "ret_5", "ret_21", "vol_21", "rsi_14", "dist_sma50", "macd_hist"]


def make_features(price: pd.Series) -> pd.DataFrame:
    """Features known at the close of day t (only past data)."""
    f = pd.DataFrame(index=price.index)
    f["ret_1"] = price.pct_change(1)
    f["ret_5"] = price.pct_change(5)
    f["ret_21"] = price.pct_change(21)
    f["vol_21"] = price.pct_change().rolling(21).std()
    f["rsi_14"] = indicators.rsi(price, 14) / 100.0
    f["dist_sma50"] = price / indicators.sma(price, 50) - 1
    f["macd_hist"] = indicators.macd(price)["hist"] / price
    return f[FEATURES]


def make_dataset(price: pd.Series, horizon: int = 21) -> tuple[pd.DataFrame, pd.Series]:
    price = price.dropna()
    X = make_features(price)
    fwd = price.shift(-horizon) / price - 1
    y = (fwd > 0).astype(int)
    keep = X.notna().all(axis=1) & fwd.notna()
    return X[keep], y[keep]


def walk_forward_splits(n: int, horizon: int, initial_train: int = 756, test_size: int = 63):
    """Yield (train_idx, test_idx) integer arrays. Train ends `horizon` rows before the test block."""
    start = initial_train
    while start < n:
        end = min(start + test_size, n)
        train_end = start - horizon  # label of row i uses price[i + horizon] -> need i + horizon <= start
        if train_end > 100:
            yield np.arange(0, train_end + 1), np.arange(start, end)
        start = end


def _models(seed: int = 0) -> dict:
    return {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=150, max_depth=4, min_samples_leaf=25, random_state=seed, n_jobs=-1
        ),
    }


def run_experiment(
    price: pd.Series,
    horizon: int = 21,
    initial_train: int = 756,
    test_size: int = 63,
    seed: int = 0,
) -> dict:
    """Walk-forward experiment for one price series. Returns summary table, per-fold table, verdict."""
    X, y = make_dataset(price, horizon)
    n = len(X)
    if n < initial_train + test_size + horizon + 50:
        raise ValueError(f"need more history: have {n} usable rows, need >{initial_train + test_size + horizon + 50}")

    preds: dict[str, list] = {name: [] for name in [*_models(seed), "always_up", "last_month_repeats"]}
    truth, test_pos, fold_ids = [], [], []
    for k, (tr, te) in enumerate(walk_forward_splits(n, horizon, initial_train, test_size)):
        assert tr.max() + horizon <= te.min(), "train/test overlap (leak)"
        Xtr, ytr, Xte = X.iloc[tr], y.iloc[tr], X.iloc[te]
        for name, make in _models(seed).items():
            m = make().fit(Xtr, ytr)
            preds[name] += list(m.predict(Xte))
        preds["always_up"] += [int(ytr.mean() >= 0.5)] * len(te)
        preds["last_month_repeats"] += list((Xte["ret_21"] > 0).astype(int))
        truth += list(y.iloc[te])
        test_pos += list(te)
        fold_ids += [k] * len(te)

    truth_a = np.array(truth)
    pred_df = pd.DataFrame(preds, index=X.index[test_pos]).assign(actual=truth_a, fold=fold_ids)

    rows = []
    base = pred_df["always_up"].values
    sub = np.arange(0, len(pred_df), horizon)  # non-overlapping labels
    for name in preds:
        p = pred_df[name].values
        acc = float((p == truth_a).mean())
        bal = float(balanced_accuracy_score(truth_a, p))
        wins = int(((p[sub] == truth_a[sub]) & (base[sub] != truth_a[sub])).sum())
        losses = int(((p[sub] != truth_a[sub]) & (base[sub] == truth_a[sub])).sum())
        pval = binomtest(wins, wins + losses, 0.5).pvalue if (wins + losses) > 0 and name != "always_up" else np.nan
        rows.append(
            {"model": name, "accuracy": acc, "balanced_accuracy": bal,
             "vs_always_up_pp": (acc - float((base == truth_a).mean())) * 100, "p_value_vs_always_up": pval}
        )
    summary = pd.DataFrame(rows).set_index("model")

    fold_acc = pred_df.groupby("fold").apply(
        lambda g: pd.Series({m: (g[m] == g["actual"]).mean() for m in preds}), include_groups=False
    )
    return {
        "summary": summary,
        "fold_accuracy": fold_acc,
        "predictions": pred_df,
        "n_test": int(len(pred_df)),
        "n_folds": int(pred_df["fold"].nunique()),
        "horizon": horizon,
        "base_rate_up": float(truth_a.mean()),
        "verdict": verdict(summary),
    }


def verdict(summary: pd.DataFrame, alpha: float = 0.05) -> str:
    """Plain-English honest conclusion."""
    ml = summary.loc[["logistic", "random_forest"]]
    best = ml["vs_always_up_pp"].idxmax()
    pp = ml.loc[best, "vs_always_up_pp"]
    p = min(1.0, ml.loc[best, "p_value_vs_always_up"] * len(ml))  # Bonferroni: we looked at 2 models
    if pp > 0 and p < alpha:
        return (f"{best} beat the always-up baseline by {pp:+.1f} pp (adjusted p={p:.3f} on non-overlapping samples). "
                "Treat as a lead to re-test on other periods/markets, not a trading signal.")
    return (f"No model clearly beat the always-up baseline (best: {best}, {pp:+.1f} pp, p={p:.2f}). "
            "Stock direction over this horizon is close to unpredictable from these features - an honest negative result.")


def run_experiment_many(prices: pd.DataFrame, progress=None, **kw) -> pd.DataFrame:
    """Run the experiment on every symbol; one row per symbol.

    Columns: best_model / best_vs_baseline_pp (the better of logistic and forest, as in `verdict`),
    p_adj_models (that model's p-value x2, because two models were tried) and p_adj_all
    (p_adj_models x number of stocks tested: Bonferroni, because many stocks were tried).
    `progress(i, n, symbol)` is called before each symbol so a UI can show a progress bar."""
    rows = []
    syms = list(prices.columns)
    for i, sym in enumerate(syms):
        if progress:
            progress(i, len(syms), sym)
        try:
            res = run_experiment(prices[sym].dropna(), **kw)
        except ValueError as exc:
            rows.append({"symbol": sym, "note": str(exc)})
            continue
        s = res["summary"]
        ml = s.loc[["logistic", "random_forest"]]
        best = ml["vs_always_up_pp"].idxmax()
        p = ml.loc[best, "p_value_vs_always_up"]
        p = 1.0 if pd.isna(p) else float(p)  # NaN = predictions never differed from the baseline
        rows.append(
            {
                "symbol": sym,
                "n_test": res["n_test"],
                "baseline_acc": s.loc["always_up", "accuracy"],
                "logistic_acc": s.loc["logistic", "accuracy"],
                "forest_acc": s.loc["random_forest", "accuracy"],
                "best_model": best,
                "best_vs_baseline_pp": float(ml.loc[best, "vs_always_up_pp"]),
                "p_adj_models": min(1.0, p * 2),
            }
        )
    out = pd.DataFrame(rows).set_index("symbol")
    if "p_adj_models" in out:
        n_tested = int(out["p_adj_models"].notna().sum())
        out["p_adj_all"] = (out["p_adj_models"] * n_tested).clip(upper=1.0)
    return out


def summarize_many(table: pd.DataFrame, alpha: float = 0.05) -> str:
    """Plain-English reading of `run_experiment_many`, including how many wins luck alone would give."""
    if "p_adj_all" not in table or table["p_adj_all"].notna().sum() == 0:
        return "No stock had enough history to test."
    t = table.dropna(subset=["p_adj_all"])
    n = len(t)
    ahead = int((t["best_vs_baseline_pp"] > 0).sum())
    hits_each = int(((t["best_vs_baseline_pp"] > 0) & (t["p_adj_models"] < alpha)).sum())
    hits_all = int(((t["best_vs_baseline_pp"] > 0) & (t["p_adj_all"] < alpha)).sum())
    return (
        f"Tested {n} stocks. A model was ahead of the always-up baseline on {ahead} of them (even a coin flip "
        f"would do that on about half). {hits_each} looked significant on their own (p<{alpha}); "
        f"{hits_all} stayed significant after correcting for testing {n} stocks. "
        f"By luck alone you would expect about {alpha * n:.1f} 'significant' stocks without that correction. "
        + ("Nothing survives the correction: an honest negative result." if hits_all == 0
           else "Treat any survivor as a lead to re-test on other periods or markets, not a trading signal.")
    )

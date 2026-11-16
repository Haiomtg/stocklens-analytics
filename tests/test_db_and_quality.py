import numpy as np
import pandas as pd

from stocklens import db
from stocklens.ingest import quality


def _bars(n=60, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=n)
    close = pd.Series(100.0 + np.arange(n) * 0.1, index=idx)
    return pd.DataFrame({"date": idx, "open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "adj_close": close, "volume": 1000.0}).reset_index(drop=True)


def test_upsert_is_idempotent_and_load_roundtrip(conn):
    df = _bars()
    db.upsert_prices(conn, df, "AAA", "US")
    db.upsert_prices(conn, df, "AAA", "US")  # second insert must not duplicate
    assert conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0] == len(df)
    wide = db.load_prices(conn, ["AAA"], field="close")
    assert list(wide.columns) == ["AAA"]
    assert np.isclose(wide["AAA"].iloc[0], 100.0)
    assert db.last_date(conn, "AAA") == df["date"].max().strftime("%Y-%m-%d")


def test_clean_data_has_no_flags():
    df = _bars()
    flags = quality.run_all(df, today=df["date"].max())
    assert flags.empty


def test_gap_detected_but_single_holiday_ignored():
    df = _bars(60)
    holiday = df.drop(index=df.index[10])                     # one missing day: normal
    assert quality.find_gaps(holiday["date"]).empty
    long_gap = df.drop(index=df.index[10:20])         # 10 missing business days
    flags = quality.find_gaps(long_gap["date"])
    assert len(flags) == 1 and flags.iloc[0]["flag"] == "gap"


def test_unadjusted_split_flagged():
    df = _bars(80)
    df.loc[40:, ["open", "high", "low", "close", "adj_close"]] /= 2  # looks like a 2:1 split
    flags = quality.find_suspect_splits(df.set_index("date")["close"])
    assert len(flags) == 1 and flags.iloc[0]["flag"] == "possible_split"


def test_outlier_flagged():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2023-01-01", periods=300)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=idx)
    close.iloc[150:] *= 1.5  # +50% jump
    flags = quality.find_outliers(close)
    assert len(flags) == 1


def test_stale_data_flagged():
    df = _bars()
    flags = quality.stale_check(df, today=df["date"].max() + pd.Timedelta(days=30))
    assert flags.iloc[0]["flag"] == "stale"

"""Price files you export yourself (broker or data site) -> the same three-method fetcher interface.

Why this exists: the Vietnam package `vnstock` is not installable right now (see vietnam.py), and
Yahoo does not carry Vietnamese stocks. Importing CSV files has no terms-of-service risk, no
network, and cannot break when a website changes.

HOW TO USE
  1. Put one file per ticker in  data/vn_csv/  named after the ticker:  FPT.csv, SAB.csv, VNINDEX.csv ...
  2. Each file needs a date column and a close column. Open/high/low/volume are optional.
  3. Run  python -m stocklens.ingest.update --watchlist config/private.toml

Recognised headers (case, accents and spaces are ignored), English or Vietnamese:
  date    : date, time, ngay, ngay giao dich
  open    : open, gia mo cua, mo cua
  high    : high, cao nhat, gia cao nhat
  low     : low, thap nhat, gia thap nhat
  close   : close, price, gia dong cua, dong cua, gia dieu chinh (see note), gia
  volume  : volume, vol, vol., khoi luong, kl, khoi luong khop lenh
  adj_close (optional): adj close, adj_close

Things this file handles so you do not have to:
  * dates written 31/12/2025 (day first) or 2025-12-31
  * numbers like 46,000  or  1.2M / 350K  (volume suffixes)
  * UNITS: Vietnamese quotes are often in thousands of VND (46.0 = 46,000 VND) and sometimes in
    full VND (46000). Everything is stored in THOUSANDS of VND, matching your broker app.
    The rule: if the median close is above 1000, divide prices by 1000. No HOSE/HNX stock trades
    above 1,000,000 VND, so this cannot misfire on real data.

Limitation: dividends and fundamentals are not read from CSV (empty). Log VN dividends in your
transactions file (type=DIVIDEND). Also: use prices adjusted for splits/bonus shares if the site
offers it, otherwise the data-quality check will flag the jump as a suspected split.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

from stocklens.ingest.base import PRICE_COLUMNS, validate_price_frame

log = logging.getLogger("stocklens.csv")

DEFAULT_FOLDER = Path("data/vn_csv")

_ALIASES = {
    "date": {"date", "time", "ngay", "ngay giao dich"},
    "open": {"open", "gia mo cua", "mo cua"},
    "high": {"high", "cao nhat", "gia cao nhat"},
    "low": {"low", "thap nhat", "gia thap nhat"},
    "close": {"close", "price", "gia dong cua", "dong cua", "gia"},
    "adj_close": {"adj close", "adj_close", "adjclose", "gia dieu chinh"},
    "volume": {"volume", "vol", "vol.", "khoi luong", "kl", "khoi luong khop lenh"},
}


def _norm(header: str) -> str:
    text = unicodedata.normalize("NFKD", str(header)).replace("đ", "d").replace("Đ", "D")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip().lower()


def _to_number(series: pd.Series) -> pd.Series:
    """'46,000' -> 46000, '1.2M' -> 1200000, '350K' -> 350000, '45,5' -> 45.5, '1.234.567' -> 1234567."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    def one(value) -> float:
        if pd.isna(value):
            return float("nan")
        s = str(value).strip().replace(" ", "").replace(" ", "")
        if s in ("", "-", "--"):
            return float("nan")
        mult = 1.0
        if s[-1] in "KkMmBb":
            mult = {"k": 1e3, "m": 1e6, "b": 1e9}[s[-1].lower()]
            s = s[:-1]
        if "," in s and "." in s:  # whichever comes last is the decimal mark
            dec = "," if s.rfind(",") > s.rfind(".") else "."
            thou = "." if dec == "," else ","
            s = s.replace(thou, "").replace(dec, ".")
        elif "," in s:
            s = s.replace(",", "") if re.fullmatch(r"\d{1,3}(,\d{3})+", s) else s.replace(",", ".")
        elif s.count(".") > 1:  # 1.234.567 -> thousands separators
            s = s.replace(".", "")
        try:
            return float(s) * mult
        except ValueError:
            return float("nan")

    return series.map(one)


def _parse_dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    text = series.astype(str).str.strip()
    iso = text.str.fullmatch(r"\d{4}-\d{2}-\d{2}.*")
    if iso.all():
        return pd.to_datetime(text.str[:10], format="%Y-%m-%d")
    return pd.to_datetime(text, dayfirst=True, format="mixed")


class CsvFetcher:
    market = "VN"

    def __init__(self, folder: str | Path = DEFAULT_FOLDER):
        self.folder = Path(folder)

    def path_for(self, symbol: str) -> Path | None:
        if not self.folder.is_dir():
            return None
        wanted = f"{symbol}.csv".lower()
        for p in self.folder.iterdir():
            if p.name.lower() == wanted:
                return p
        return None

    def has(self, symbol: str) -> bool:
        return self.path_for(symbol) is not None

    def fetch_prices(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        path = self.path_for(symbol)
        if path is None:
            raise FileNotFoundError(f"no price file for {symbol}: expected {self.folder / (symbol + '.csv')}")
        raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
        lookup = {}
        for col in raw.columns:
            for field, names in _ALIASES.items():
                if _norm(col) in names and field not in lookup:
                    lookup[field] = col
        for needed in ("date", "close"):
            if needed not in lookup:
                raise ValueError(f"{path.name}: cannot find a '{needed}' column. Columns found: {list(raw.columns)}")

        df = pd.DataFrame({"date": _parse_dates(raw[lookup["date"]])})
        for field in ("open", "high", "low", "close", "adj_close", "volume"):
            df[field] = _to_number(raw[lookup[field]]) if field in lookup else float("nan")
        for field in ("open", "high", "low"):  # a missing O/H/L column falls back to close
            df[field] = df[field].fillna(df["close"])
        df["adj_close"] = df["adj_close"].fillna(df["close"])
        df["volume"] = df["volume"].fillna(0)

        if df["close"].median() > 1000:  # full VND -> thousands of VND
            log.info("%s: prices look like full VND, converting to thousands", symbol)
            for field in ("open", "high", "low", "close", "adj_close"):
                df[field] = df[field] / 1000

        df = df[df["date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["date"] <= pd.Timestamp(end)]
        if df.empty:
            return pd.DataFrame(columns=PRICE_COLUMNS)
        return validate_price_frame(df)

    def fetch_dividends(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "amount"])

    def fetch_fundamentals(self, symbol: str) -> dict:
        return {"symbol": symbol}

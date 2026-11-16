"""Technical indicators. All functions take a price Series and return Series/DataFrames
aligned to the same index. They only use past data (no look-ahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(price: pd.Series, window: int) -> pd.Series:
    return price.rolling(window, min_periods=window).mean()


def ema(price: pd.Series, span: int) -> pd.Series:
    return price.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(price: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI, 0-100."""
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out[(avg_loss == 0) & (avg_gain > 0)] = 100.0  # all gains, no losses
    return out


def macd(price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(price, fast) - ema(price, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def rolling_volatility(price: pd.Series, window: int = 21, periods: int = 252) -> pd.Series:
    """Annualised rolling volatility of daily log returns."""
    r = np.log(price).diff()
    return r.rolling(window, min_periods=window).std() * np.sqrt(periods)


def bollinger(price: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(price, window)
    sd = price.rolling(window, min_periods=window).std()
    return pd.DataFrame({"mid": mid, "upper": mid + k * sd, "lower": mid - k * sd})


def add_all(price: pd.Series) -> pd.DataFrame:
    """Convenience bundle used by the Stock Deep-Dive page."""
    out = pd.DataFrame({"price": price})
    out["sma50"] = sma(price, 50)
    out["sma200"] = sma(price, 200)
    out["rsi14"] = rsi(price, 14)
    out = out.join(macd(price))
    out["vol21"] = rolling_volatility(price, 21)
    return out

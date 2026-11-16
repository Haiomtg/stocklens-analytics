"""Transactions -> holdings, average cost, realised / unrealised P&L, dividends.

Transaction CSV columns (see data/sample_transactions.csv):
    date, symbol, type, quantity, price, fees

    type = BUY       quantity shares bought at `price` each, plus `fees` in total
    type = SELL      quantity shares sold at `price` each, minus `fees` in total
    type = DIVIDEND  `quantity` = shares you held, `price` = cash dividend PER SHARE, fees usually 0

Cost method: AVERAGE COST (not FIFO). Simple, and what most brokers show.

KNOWN LIMITATIONS (documented on purpose - fix them in Stage 2 if you want):
  * Stock splits / bonus shares are not modelled. Log them by hand as a SELL of the old
    position and a BUY of the new one at the same total cost, or extend this module.
  * Single currency: all prices are treated as being in the same currency.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REQUIRED = ["date", "symbol", "type", "quantity", "price", "fees"]
_TYPE_ORDER = {"BUY": 0, "DIVIDEND": 1, "SELL": 2}  # same-day: buy first, so you can't sell what you don't own


def load_transactions(source) -> pd.DataFrame:
    """Read a CSV path or DataFrame and validate it. Raises ValueError with a clear message."""
    df = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"transactions file is missing columns: {missing}")
    df = df[REQUIRED].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip().str.upper()
    bad = set(df["type"]) - set(_TYPE_ORDER)
    if bad:
        raise ValueError(f"unknown transaction type(s): {sorted(bad)} (use BUY, SELL, DIVIDEND)")
    for c in ["quantity", "price", "fees"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    df["fees"] = df["fees"].fillna(0.0)
    if (df["quantity"] <= 0).any():
        raise ValueError("quantity must be positive (use type=SELL for sales)")
    if (df.loc[df["type"].isin(["BUY", "SELL"]), "price"].fillna(-1) <= 0).any():
        raise ValueError("BUY/SELL rows need a positive price")
    df["_o"] = df["type"].map(_TYPE_ORDER)
    df = df.sort_values(["date", "_o"], kind="stable").drop(columns="_o").reset_index(drop=True)
    return df


@dataclass
class _Position:
    qty: float = 0.0
    cost: float = 0.0       # total cost basis of shares currently held
    realised: float = 0.0
    dividends: float = 0.0
    fees: float = 0.0

    @property
    def avg_cost(self) -> float:
        return self.cost / self.qty if self.qty else 0.0


def build_positions(tx: pd.DataFrame) -> dict[str, _Position]:
    pos: dict[str, _Position] = {}
    for r in tx.itertuples():
        p = pos.setdefault(r.symbol, _Position())
        if r.type == "BUY":
            p.qty += r.quantity
            p.cost += r.quantity * r.price + r.fees
            p.fees += r.fees
        elif r.type == "SELL":
            if r.quantity > p.qty + 1e-9:
                raise ValueError(f"{r.date.date()}: selling {r.quantity} {r.symbol} but only {p.qty} held")
            avg = p.avg_cost
            proceeds = r.quantity * r.price - r.fees
            p.realised += proceeds - r.quantity * avg
            p.cost -= r.quantity * avg
            p.qty -= r.quantity
            p.fees += r.fees
            if p.qty < 1e-9:
                p.qty, p.cost = 0.0, 0.0
        else:  # DIVIDEND
            p.dividends += r.quantity * r.price - r.fees
    return pos


def holdings_table(tx: pd.DataFrame, last_prices: pd.Series, sectors: pd.Series | None = None) -> pd.DataFrame:
    """One row per symbol ever traded. `last_prices` = latest price per symbol (same currency as trades)."""
    pos = build_positions(tx)
    rows = []
    for sym, p in pos.items():
        px = float(last_prices.get(sym, float("nan")))
        mv = p.qty * px if p.qty else 0.0
        unreal = mv - p.cost if p.qty else 0.0
        rows.append(
            {
                "symbol": sym,
                "quantity": p.qty,
                "avg_cost": p.avg_cost,
                "cost_basis": p.cost,
                "last_price": px,
                "market_value": mv,
                "unrealised_pnl": unreal,
                "unrealised_pct": unreal / p.cost if p.cost else float("nan"),
                "realised_pnl": p.realised,
                "dividends": p.dividends,
                "total_pnl": unreal + p.realised + p.dividends,
            }
        )
    df = pd.DataFrame(rows).set_index("symbol")
    total_mv = df["market_value"].sum()
    df["weight"] = df["market_value"] / total_mv if total_mv else 0.0
    if sectors is not None:
        df["sector"] = sectors.reindex(df.index)
    return df.sort_values("market_value", ascending=False)


def summary(holdings: pd.DataFrame) -> dict[str, float]:
    open_ = holdings[holdings["quantity"] > 0]
    cost = open_["cost_basis"].sum()
    return {
        "market_value": float(open_["market_value"].sum()),
        "cost_basis": float(cost),
        "unrealised_pnl": float(open_["unrealised_pnl"].sum()),
        "unrealised_pct": float(open_["unrealised_pnl"].sum() / cost) if cost else float("nan"),
        "realised_pnl": float(holdings["realised_pnl"].sum()),
        "dividends": float(holdings["dividends"].sum()),
        "total_pnl": float(holdings["total_pnl"].sum()),
        "n_positions": int(len(open_)),
    }

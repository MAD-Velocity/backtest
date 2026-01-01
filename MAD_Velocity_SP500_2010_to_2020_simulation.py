#!/usr/bin/env python3
"""
MAD-Velocity Portfolio Simulation (GitHub-ready, NO AWS credentials)

Universe:
  - Symbols CSV (public URL)
  - Per-symbol indicator CSVs (public URL directory)

Requirements in each indicator CSV:
  - date column (default: "date")
  - a price column: one of ["close","adj close","adj_close","adjclose"] (case-insensitive)
  - buy-signals column (default: "buy-signals")
  - sell-signals column (default: "sell-signals")

Simulation rules:
  - Start cash = $10,000
  - Each NEW BUY uses risk_fraction (default 0.02) of CURRENT portfolio equity
  - Hold until sell signal
  - Multiple concurrent positions allowed by default

Outputs (saved locally, same directory you run from):
  - {BASE}_trades.csv
  - {BASE}_equity.csv
  - {BASE}_summary.json

Example run (no args needed):
  python MAD_Velocity_SP500_2010_to_2020_simulation.py

Optional args:
  python script.py --start-cash 10000 --risk-fraction 0.02 --single-position-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -------- Optional progress bar --------
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # fallback handled below

# -------- HTTP download --------
try:
    import requests
except ImportError as e:
    raise SystemExit(
        "Missing dependency: requests. Install with:\n"
        "  pip install requests\n"
    ) from e


# -------------------------
# Defaults (public URLs)
# -------------------------
DEFAULT_SYMBOLS_URL = (
    "https://mad-velocity-backtest-data.s3.us-east-1.amazonaws.com/"
    "sp500_constituents_continuous_2010_2020_tickers_downloaded.csv"
)

DEFAULT_INDICATORS_BASE_URL = (
    "https://mad-velocity-backtest-data.s3.us-east-1.amazonaws.com/indicators/"
)

# Output base string (requested)
DEFAULT_OUT_BASE = "MAD_Velocity_SP500_2010_to_2020_simulation"

# Simulation defaults
START_CASH_DEFAULT = 10_000.0
RISK_FRACTION_DEFAULT = 0.02  # 2% of *current* equity per new BUY

# Column defaults
DATE_COL_DEFAULT = "date"
BUY_COL_DEFAULT = "buy-signals"
SELL_COL_DEFAULT = "sell-signals"
PRICE_COL_CANDIDATES_DEFAULT = ["close", "adj close", "adj_close", "adjclose"]


# -------------------------
# Types
# -------------------------
@dataclass
class Position:
    shares: float
    entry_price: float
    entry_date: pd.Timestamp


# -------------------------
# Helpers
# -------------------------
def _pbar(iterable, **kwargs):
    """tqdm wrapper that degrades gracefully."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def http_get_bytes(url: str, timeout: int = 60) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def read_csv_from_url(url: str) -> pd.DataFrame:
    content = http_get_bytes(url)
    # pandas can read from bytes via BytesIO
    from io import BytesIO
    return pd.read_csv(BytesIO(content))


def normalize_symbol(sym: str) -> str:
    return str(sym).strip().upper()


def infer_symbol_column(df: pd.DataFrame) -> str:
    for c in ["symbol", "ticker", "Symbol", "Ticker"]:
        if c in df.columns:
            return c
    if df.shape[1] == 1:
        return df.columns[0]
    raise ValueError(f"Could not infer symbol column. Columns: {list(df.columns)}")


def infer_price_col(df: pd.DataFrame, candidates: List[str]) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise ValueError(
        f"Could not find a price column. Tried {candidates}. Available: {list(df.columns)}"
    )


def safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def load_universe_symbols(symbols_url: str, limit: Optional[int] = None) -> List[str]:
    df = read_csv_from_url(symbols_url)
    sym_col = infer_symbol_column(df)
    syms = [normalize_symbol(x) for x in df[sym_col].dropna().tolist()]
    syms = [s for s in syms if s and s != "NAN"]
    if limit is not None:
        syms = syms[: int(limit)]
    return syms


def load_indicator_df(
    indicators_base_url: str,
    symbol: str,
    date_col: str,
    buy_col: str,
    sell_col: str,
    price_candidates: List[str],
) -> Tuple[pd.DataFrame, str]:
    url = f"{indicators_base_url}{symbol}.csv"
    df = read_csv_from_url(url)

    if date_col not in df.columns:
        raise ValueError(f"{symbol}: missing required date column '{date_col}'")

    if buy_col not in df.columns or sell_col not in df.columns:
        raise ValueError(f"{symbol}: missing '{buy_col}' and/or '{sell_col}'")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    price_col = infer_price_col(df, price_candidates)
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # buy/sell columns: coerce to int-ish, preserve NaNs if any
    # then treat nonzero as True in simulation
    df[buy_col] = pd.to_numeric(df[buy_col], errors="coerce").fillna(0).astype(int)
    df[sell_col] = pd.to_numeric(df[sell_col], errors="coerce").fillna(0).astype(int)

    # Keep only what we need
    df = df[[date_col, price_col, buy_col, sell_col]].copy()
    df = df.dropna(subset=[price_col])

    # index by date for fast access
    df = df.set_index(date_col)
    return df, price_col


def compute_portfolio_equity(
    cash: float,
    positions: Dict[str, Position],
    data_by_sym: Dict[str, pd.DataFrame],
    price_col_by_sym: Dict[str, str],
    dt: pd.Timestamp,
) -> Tuple[float, float]:
    holdings_value = 0.0
    for sym, pos in positions.items():
        df = data_by_sym[sym]
        px_col = price_col_by_sym[sym]
        if dt in df.index:
            px = safe_float(df.at[dt, px_col])
            if px is not None:
                holdings_value += pos.shares * px
    equity = cash + holdings_value
    return equity, holdings_value


def simulate(
    symbols: List[str],
    data_by_sym: Dict[str, pd.DataFrame],
    price_col_by_sym: Dict[str, str],
    buy_col: str,
    sell_col: str,
    start_cash: float,
    risk_fraction: float,
    allow_multiple_positions: bool,
    progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    # master calendar = union of all dates
    all_dates = set()
    for sym in symbols:
        all_dates.update(data_by_sym[sym].index.tolist())
    calendar = sorted(all_dates)
    if not calendar:
        raise ValueError("No dates found across indicator CSVs.")

    cash = float(start_cash)
    positions: Dict[str, Position] = {}
    trades: List[Dict] = []
    equity_rows: List[Dict] = []

    day_iter = calendar
    if progress and tqdm is not None:
        day_iter = tqdm(calendar, desc="Simulating days", unit="day")

    for dt in day_iter:
        # mark-to-market
        equity, holdings_value = compute_portfolio_equity(cash, positions, data_by_sym, price_col_by_sym, dt)

        # --- SELLS first ---
        to_close: List[str] = []
        for sym, pos in list(positions.items()):
            df = data_by_sym[sym]
            px_col = price_col_by_sym[sym]
            if dt not in df.index:
                continue

            sell_sig = int(df.at[dt, sell_col])
            px = safe_float(df.at[dt, px_col])
            if px is None:
                continue

            if sell_sig != 0:
                proceeds = pos.shares * px
                cash += proceeds
                pnl = (px - pos.entry_price) * pos.shares
                trades.append(
                    {
                        "date": dt.strftime("%Y-%m-%d"),
                        "ticker": sym,
                        "side": "SELL",
                        "price": float(px),
                        "shares": float(pos.shares),
                        "notional": float(proceeds),
                        "pnl": float(pnl),
                        "entry_date": pos.entry_date.strftime("%Y-%m-%d"),
                        "entry_price": float(pos.entry_price),
                        "cash_after": float(cash),
                    }
                )
                to_close.append(sym)

        for sym in to_close:
            positions.pop(sym, None)

        # recompute equity after sells
        equity, holdings_value = compute_portfolio_equity(cash, positions, data_by_sym, price_col_by_sym, dt)

        # --- BUYS ---
        if (not allow_multiple_positions) and positions:
            # skip buys if single-position mode and we already have one
            pass
        else:
            # iterate in fixed ticker order (deterministic)
            for sym in symbols:
                if sym in positions:
                    continue
                if (not allow_multiple_positions) and positions:
                    break

                df = data_by_sym[sym]
                px_col = price_col_by_sym[sym]
                if dt not in df.index:
                    continue

                buy_sig = int(df.at[dt, buy_col])
                if buy_sig == 0:
                    continue

                px = safe_float(df.at[dt, px_col])
                if px is None or px <= 0:
                    continue

                # size = current equity * risk_fraction
                equity_now, _ = compute_portfolio_equity(cash, positions, data_by_sym, price_col_by_sym, dt)
                trade_dollars = equity_now * float(risk_fraction)

                if trade_dollars <= 0 or cash < trade_dollars:
                    continue

                shares = trade_dollars / px
                cash -= trade_dollars

                positions[sym] = Position(
                    shares=float(shares),
                    entry_price=float(px),
                    entry_date=pd.Timestamp(dt),
                )

                trades.append(
                    {
                        "date": dt.strftime("%Y-%m-%d"),
                        "ticker": sym,
                        "side": "BUY",
                        "price": float(px),
                        "shares": float(shares),
                        "notional": float(trade_dollars),
                        "cash_after": float(cash),
                    }
                )

        # end-of-day equity snapshot
        equity, holdings_value = compute_portfolio_equity(cash, positions, data_by_sym, price_col_by_sym, dt)
        equity_rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "cash": float(cash),
                "holdings_value": float(holdings_value),
                "equity": float(equity),
                "n_positions": int(len(positions)),
            }
        )

    # liquidate remaining at last date
    last_dt = calendar[-1]
    liquidation_pnl = 0.0
    for sym, pos in list(positions.items()):
        df = data_by_sym[sym]
        px_col = price_col_by_sym[sym]
        if last_dt in df.index:
            px = safe_float(df.at[last_dt, px_col])
            if px is not None and px > 0:
                proceeds = pos.shares * px
                cash += proceeds
                pnl = (px - pos.entry_price) * pos.shares
                liquidation_pnl += pnl
                trades.append(
                    {
                        "date": last_dt.strftime("%Y-%m-%d"),
                        "ticker": sym,
                        "side": "LIQUIDATE",
                        "price": float(px),
                        "shares": float(pos.shares),
                        "notional": float(proceeds),
                        "pnl": float(pnl),
                        "entry_date": pos.entry_date.strftime("%Y-%m-%d"),
                        "entry_price": float(pos.entry_price),
                        "cash_after": float(cash),
                    }
                )
        positions.pop(sym, None)

    # Build DataFrames
    trades_df = pd.DataFrame(trades)

    equity_df = pd.DataFrame(equity_rows).drop_duplicates(subset=["date"], keep="last")
    equity_df["date"] = pd.to_datetime(equity_df["date"])
    equity_df = equity_df.sort_values("date").reset_index(drop=True)

    # Summary metrics
    start_eq = float(start_cash)
    end_eq = float(equity_df["equity"].iloc[-1]) if len(equity_df) else float("nan")
    days = int((equity_df["date"].iloc[-1] - equity_df["date"].iloc[0]).days) if len(equity_df) else 0
    years = max(days / 365.25, 1e-9)
    cagr = (end_eq / start_eq) ** (1 / years) - 1 if start_eq > 0 and end_eq > 0 else np.nan

    eq_series = equity_df.set_index("date")["equity"].astype(float)
    daily_ret = eq_series.pct_change(fill_method=None).dropna()
    sharpe = np.nan
    if len(daily_ret) > 2 and float(daily_ret.std(ddof=1)) > 0:
        sharpe = float((daily_ret.mean() / daily_ret.std(ddof=1)) * np.sqrt(252))

    roll_max = eq_series.cummax()
    dd = (eq_series / roll_max) - 1.0
    max_dd = float(dd.min()) if len(dd) else np.nan

    summary = {
        "start_cash": start_cash,
        "risk_fraction": risk_fraction,
        "allow_multiple_positions": allow_multiple_positions,
        "start_date": equity_df["date"].iloc[0].strftime("%Y-%m-%d") if len(equity_df) else None,
        "end_date": equity_df["date"].iloc[-1].strftime("%Y-%m-%d") if len(equity_df) else None,
        "days": days,
        "final_equity": end_eq,
        "CAGR": float(cagr) if np.isfinite(cagr) else None,
        "Sharpe": float(sharpe) if np.isfinite(sharpe) else None,
        "MaxDD": float(max_dd) if np.isfinite(max_dd) else None,
        "n_trade_rows": int(len(trades_df)),
        "n_buys": int((trades_df.get("side") == "BUY").sum()) if not trades_df.empty else 0,
        "n_sells": int((trades_df.get("side") == "SELL").sum()) if not trades_df.empty else 0,
        "n_liquidations": int((trades_df.get("side") == "LIQUIDATE").sum()) if not trades_df.empty else 0,
        "liquidation_pnl": float(liquidation_pnl),
        "universe_size": int(len(symbols)),
    }

    return trades_df, equity_df, summary


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--symbols-url", default=DEFAULT_SYMBOLS_URL, help="Public URL to symbols CSV")
    ap.add_argument("--indicators-base-url", default=DEFAULT_INDICATORS_BASE_URL, help="Public base URL to indicators/ (must end with /)")
    ap.add_argument("--out-base", default=DEFAULT_OUT_BASE, help="Base filename prefix for local outputs")

    ap.add_argument("--start-cash", type=float, default=START_CASH_DEFAULT)
    ap.add_argument("--risk-fraction", type=float, default=RISK_FRACTION_DEFAULT)

    ap.add_argument("--date-col", default=DATE_COL_DEFAULT)
    ap.add_argument("--buy-col", default=BUY_COL_DEFAULT)
    ap.add_argument("--sell-col", default=SELL_COL_DEFAULT)
    ap.add_argument("--price-cols", nargs="+", default=PRICE_COL_CANDIDATES_DEFAULT)

    ap.add_argument("--single-position-only", action="store_true", default=False)
    ap.add_argument("--limit-symbols", type=int, default=None)

    args = ap.parse_args()

    if not args.indicators_base_url.endswith("/"):
        args.indicators_base_url += "/"

    allow_multiple = not args.single_position_only

    # 1) Load universe
    symbols = load_universe_symbols(args.symbols_url, limit=args.limit_symbols)
    print(f"Loaded {len(symbols)} symbols from {args.symbols_url}")

    # 2) Load indicator CSVs
    data_by_sym: Dict[str, pd.DataFrame] = {}
    price_col_by_sym: Dict[str, str] = {}
    failures: List[Tuple[str, str]] = []

    sym_iter = symbols
    if tqdm is not None:
        sym_iter = tqdm(symbols, desc="Downloading indicator CSVs", unit="ticker")

    for sym in sym_iter:
        try:
            df, px_col = load_indicator_df(
                indicators_base_url=args.indicators_base_url,
                symbol=sym,
                date_col=args.date_col,
                buy_col=args.buy_col,
                sell_col=args.sell_col,
                price_candidates=args.price_cols,
            )
            if len(df) == 0:
                raise ValueError("empty indicator dataframe after cleaning")
            data_by_sym[sym] = df
            price_col_by_sym[sym] = px_col
        except Exception as e:
            failures.append((sym, str(e)))

    ok_syms = list(data_by_sym.keys())
    print(f"Indicator load done: ok={len(ok_syms)} missing/bad={len(failures)}")
    if failures:
        print("First 10 failures:")
        for s, err in failures[:10]:
            print(f"  - {s}: {err}")

    if not ok_syms:
        print("ERROR: No usable indicator CSVs. Exiting.")
        return 2

    # 3) Run simulation
    print(
        f"Starting simulation: start_cash={args.start_cash:.2f}, "
        f"risk_fraction={args.risk_fraction:.4f}, allow_multiple_positions={allow_multiple}"
    )

    trades_df, equity_df, summary = simulate(
        symbols=ok_syms,
        data_by_sym=data_by_sym,
        price_col_by_sym=price_col_by_sym,
        buy_col=args.buy_col,
        sell_col=args.sell_col,
        start_cash=args.start_cash,
        risk_fraction=args.risk_fraction,
        allow_multiple_positions=allow_multiple,
        progress=True,
    )

    print(f"Done. Final equity: {summary['final_equity']:.2f} | Sharpe: {summary['Sharpe']} | MaxDD: {summary['MaxDD']}")

    # 4) Write outputs locally
    out_trades = f"{args.out_base}_trades.csv"
    out_equity = f"{args.out_base}_equity.csv"
    out_summary = f"{args.out_base}_summary.json"

    trades_df.to_csv(out_trades, index=False)
    equity_df.assign(date=equity_df["date"].dt.strftime("%Y-%m-%d")).to_csv(out_equity, index=False)
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nWrote local outputs:")
    print(f"  - {out_trades}  ({len(trades_df)} rows)")
    print(f"  - {out_equity}  ({len(equity_df)} rows)")
    print(f"  - {out_summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
MAD-Velocity Backtest (public, no credentials)

What this script does
- Loads the *continuous* S&P 500 (2010–2020) symbol list from a public CSV URL
- For each symbol, loads its indicator CSV from a public "indicators/" URL directory
- Computes buy/sell signals from the *regime* column:
    Buy  = regime moves from -1 to  1
    Sell = regime moves from  2 to  1
- Runs a simple single-asset backtest per symbol (one position at a time, long-only)
- Outputs per-symbol metrics:
    Symbol, CAGR, Sharpe, MaxDD, Final_Equity, Trades, Signal-to-Profit
- Saves the consolidated results locally (CSV)

Assumptions (kept simple for GitHub reproducibility)
- Uses close-to-close execution (enter/exit at the close of the signal day).
  This avoids look-ahead complexity and works consistently across indicator CSVs.

Run (works in terminal AND IPython/Jupyter):
    python backtest_mad_velocity_public.py

Optional overrides:
    python backtest_mad_velocity_public.py --out MAD-Velocity_backtest.csv
    python backtest_mad_velocity_public.py --progress-every 25
    python backtest_mad_velocity_public.py --symbols-url <URL> --indicators-base-url <URL>
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd


# -----------------------
# Defaults (public URLs)
# -----------------------
DEFAULT_SYMBOLS_URL = (
    "https://mad-velocity-backtest-data.s3.us-east-1.amazonaws.com/"
    "sp500_constituents_continuous_2010_2020_tickers_downloaded.csv"
)
DEFAULT_INDICATORS_BASE_URL = (
    "https://mad-velocity-backtest-data.s3.us-east-1.amazonaws.com/indicators/"
)

# Signal definitions (per your spec)
BUY_FROM, BUY_TO = -1, 1
SELL_FROM, SELL_TO = 2, 1


# -----------------------
# Helpers
# -----------------------
def _norm_colmap(df: pd.DataFrame) -> Dict[str, str]:
    """Map lowercase column name -> actual column name."""
    return {c.lower(): c for c in df.columns}


def _pick_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    cmap = _norm_colmap(df)
    for c in candidates:
        if c.lower() in cmap:
            return cmap[c.lower()]
    return None


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _annualized_sharpe(daily_rets: pd.Series) -> float:
    daily_rets = pd.to_numeric(daily_rets, errors="coerce").dropna()
    if len(daily_rets) < 2:
        return float("nan")
    mu = daily_rets.mean()
    sd = daily_rets.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return float("nan")
    return float(np.sqrt(252.0) * (mu / sd))


def _max_drawdown(equity: pd.Series) -> float:
    equity = pd.to_numeric(equity, errors="coerce").dropna()
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def _cagr(final_equity: float, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> float:
    if not np.isfinite(final_equity) or final_equity <= 0:
        return float("nan")
    days = (end_dt - start_dt).days
    if days <= 0:
        return float("nan")
    years = days / 365.25
    return float(final_equity ** (1.0 / years) - 1.0)


@dataclass
class BacktestResult:
    symbol: str
    cagr: float
    sharpe: float
    maxdd: float
    final_equity: float
    trades: int
    signal_to_profit: float


# -----------------------
# Core Backtest
# -----------------------
def load_symbols(symbols_url: str) -> list[str]:
    df = pd.read_csv(symbols_url)
    # common column names: Symbol, symbol, ticker, Ticker
    sym_col = _pick_col(df, ("symbol", "ticker", "tickers"))
    if sym_col is None:
        raise ValueError(
            f"Could not find a symbol column in symbols CSV. Columns: {list(df.columns)}"
        )
    syms = (
        df[sym_col]
        .astype(str)
        .str.strip()
        .replace({"": np.nan})
        .dropna()
        .unique()
        .tolist()
    )
    return syms


def load_indicator_csv(indicators_base_url: str, symbol: str) -> pd.DataFrame:
    # filenames are typically like AAPL.csv, BRK.B.csv, etc.
    # Need URL-encoding for special chars.
    filename = f"{symbol}.csv"
    url = indicators_base_url.rstrip("/") + "/" + quote(filename)

    df = pd.read_csv(url)

    # Find date column
    date_col = _pick_col(df, ("date", "datetime", "time"))
    if date_col is None:
        raise ValueError(f"{symbol}: missing date column (expected 'date'). Columns: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    # Normalize to standard column names we’ll use internally
    df = df.rename(columns={date_col: "date"})

    # regime column
    reg_col = _pick_col(df, ("regime",))
    if reg_col is None:
        raise ValueError(f"{symbol}: missing 'regime' column. Columns: {list(df.columns)}")
    if reg_col != "regime":
        df = df.rename(columns={reg_col: "regime"})

    # close column (we use close-to-close)
    close_col = _pick_col(df, ("close", "adj close", "adj_close", "adjclose"))
    if close_col is None:
        raise ValueError(f"{symbol}: missing close/adj close column. Columns: {list(df.columns)}")

    # Keep both if present; but we’ll use CLOSE_COL for pricing
    if close_col != "close":
        df = df.rename(columns={close_col: "close"})

    # Ensure numeric
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["regime"] = pd.to_numeric(df["regime"], errors="coerce")

    df = df.dropna(subset=["close", "regime"]).reset_index(drop=True)
    return df


def backtest_single_symbol(df: pd.DataFrame, symbol: str) -> BacktestResult:
    """
    Close-to-close execution:
      - Signal computed on day t, enter/exit at day t close.
    """
    if df.empty or len(df) < 50:
        return BacktestResult(symbol, np.nan, np.nan, np.nan, np.nan, 0, np.nan)

    prev = df["regime"].shift(1)
    cur = df["regime"]

    buy = (prev == BUY_FROM) & (cur == BUY_TO)
    sell = (prev == SELL_FROM) & (cur == SELL_TO)

    # Position state and equity curve
    in_pos = False
    entry_price = np.nan
    trade_pnls = []

    equity = 1.0
    equity_curve = []
    equity_dates = []

    # Daily returns applied only when in position
    close = df["close"].values
    dates = df["date"].values
    buy_arr = buy.values
    sell_arr = sell.values

    # Precompute close-to-close returns (t / t-1 - 1)
    c2c = np.empty_like(close, dtype=float)
    c2c[:] = np.nan
    c2c[1:] = (close[1:] / close[:-1]) - 1.0

    for i in range(len(df)):
        # Apply return for day i (from i-1 to i) if we were in position over that interval
        if i > 0 and in_pos and np.isfinite(c2c[i]):
            equity *= (1.0 + c2c[i])

        # Record curve point
        equity_curve.append(equity)
        equity_dates.append(dates[i])

        # Evaluate signals at the close of day i
        if not in_pos and buy_arr[i]:
            in_pos = True
            entry_price = close[i]  # enter at close
        elif in_pos and sell_arr[i]:
            exit_price = close[i]   # exit at close
            pnl = (exit_price / entry_price) - 1.0 if (np.isfinite(entry_price) and entry_price > 0) else np.nan
            trade_pnls.append(pnl)
            in_pos = False
            entry_price = np.nan

    # If still holding at end, close at last close (count it as a trade close)
    if in_pos and np.isfinite(entry_price) and entry_price > 0:
        exit_price = close[-1]
        pnl = (exit_price / entry_price) - 1.0
        trade_pnls.append(pnl)

    eq = pd.Series(equity_curve, index=pd.to_datetime(equity_dates)).sort_index()
    daily_rets = eq.pct_change().fillna(0.0)

    final_equity = float(eq.iloc[-1])
    cagr = _cagr(final_equity, eq.index[0], eq.index[-1])
    sharpe = _annualized_sharpe(daily_rets)
    maxdd = _max_drawdown(eq)

    trades = int(len(trade_pnls))
    if trades > 0:
        wins = sum(1 for p in trade_pnls if np.isfinite(p) and p > 0)
        stp = float(wins / trades)
    else:
        stp = float("nan")

    return BacktestResult(
        symbol=symbol,
        cagr=cagr,
        sharpe=sharpe,
        maxdd=maxdd,
        final_equity=final_equity,
        trades=trades,
        signal_to_profit=stp,
    )


# -----------------------
# Main
# -----------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MAD-Velocity backtest (public indicator CSVs).")

    # IMPORTANT: defaults make this run inside IPython without required args
    ap.add_argument(
        "--symbols-url",
        default=DEFAULT_SYMBOLS_URL,
        help="Public URL to the symbols CSV.",
    )
    ap.add_argument(
        "--indicators-base-url",
        default=DEFAULT_INDICATORS_BASE_URL,
        help="Base URL to the indicators directory (must end with /indicators/ or equivalent).",
    )
    ap.add_argument(
        "--out",
        default="MAD_Velocity_SP500_2010_to_2020_backtest.csv",
        help="Local output CSV filename.",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N symbols.",
    )

    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    print(f"Loading symbols from: {args.symbols_url}")
    symbols = load_symbols(args.symbols_url)
    n = len(symbols)
    print(f"Loaded {n} symbols.")

    results: list[BacktestResult] = []
    missing_or_bad = 0

    for i, sym in enumerate(symbols, start=1):
        try:
            df = load_indicator_csv(args.indicators_base_url, sym)
            res = backtest_single_symbol(df, sym)
            results.append(res)
        except Exception as e:
            missing_or_bad += 1
            # Keep it readable but useful
            print(f"WARNING [{i}/{n}] {sym}: skipped ({type(e).__name__}: {e})")

        if (i % max(1, args.progress_every) == 0) or (i == n):
            ok = len(results)
            print(f"Progress: {i}/{n} processed | ok={ok} | missing/bad={missing_or_bad}")

    out_df = pd.DataFrame(
        [
            {
                "Symbol": r.symbol,
                "CAGR": r.cagr,
                "Sharpe": r.sharpe,
                "MaxDD": r.maxdd,
                "Final_Equity": r.final_equity,
                "Trades": r.trades,
                "Signal-to-Profit": r.signal_to_profit,
            }
            for r in results
        ]
    )

    # Nice sorting for review
    if not out_df.empty:
        out_df = out_df.sort_values(["Final_Equity", "Sharpe"], ascending=False).reset_index(drop=True)

    out_df.to_csv(args.out, index=False)
    print(f"\nWrote local results to: {args.out}")
    print(f"Done. ok={len(results)} missing/bad={missing_or_bad}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

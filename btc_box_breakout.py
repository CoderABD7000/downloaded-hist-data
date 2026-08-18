"""
btc_box_breakout.py

Tests the "4-candle box breakout" pattern as an independent, standalone
signal (no ADX, no delta -- purely this pattern on its own merits):

  1. Find 4 consecutive candles whose high-low RANGES are reasonably
     similar to each other (not one huge candle mixed with tiny ones --
     max_range/min_range within a configurable ratio, default 2.5x).
  2. Mark the box: Box_High = highest high across all 4 candles,
     Box_Low = lowest low across all 4 candles.
  3. Wait for the first subsequent candle whose CLOSE breaks outside
     the box (above Box_High = bullish breakout, below Box_Low =
     bearish breakout).
  4. Enter in the breakout direction with a fixed SL/TP (same proven
     exit mechanism used throughout this project).

This is tested completely independently -- if it has real edge, it
should show up on its own, with no other indicator's help.

USAGE:
    python btc_box_breakout.py --sl 400 --tp 700
    python btc_box_breakout.py --range-ratio 2.5
"""

import argparse
import numpy as np
import pandas as pd

DEFAULT_PARQUET = r"D:\data\btc\BTCUSDT_1m_ohlcv_oi_delta_cvd_5y.parquet"


def load_and_resample_15m(parquet_path, days=None):
    print(f"Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    if days is not None:
        cutoff = df.index.max() - pd.Timedelta(days=days)
        df = df[df.index >= cutoff]
    agg = df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna()
    agg = agg.reset_index()
    print(f"Resampled to {len(agg):,} x 15m candles, {agg['timestamp'].iloc[0]} to {agg['timestamp'].iloc[-1]}")
    return agg


def find_box_breakouts(bars: pd.DataFrame, range_ratio_max: float):
    """Scans for valid 4-candle boxes and their first breakout candle.
    Returns a list of (breakout_idx, direction, box_high, box_low)."""
    high = bars["high"].values
    low = bars["low"].values
    close = bars["close"].values
    n = len(bars)
    ranges = high - low

    breakouts = []
    i = 3
    while i < n:
        box_ranges = ranges[i-3:i+1]
        if box_ranges.min() > 0:
            ratio = box_ranges.max() / box_ranges.min()
            if ratio <= range_ratio_max:
                box_high = high[i-3:i+1].max()
                box_low = low[i-3:i+1].min()
                j = i + 1
                while j < n:
                    if close[j] > box_high:
                        breakouts.append((j, "BUY", box_high, box_low))
                        i = j
                        break
                    elif close[j] < box_low:
                        breakouts.append((j, "SELL", box_high, box_low))
                        i = j
                        break
                    elif j - i > 20:
                        break
                    j += 1
        i += 1
    return breakouts


def run_backtest(bars: pd.DataFrame, breakouts: list, sl_usdt: float, tp_usdt: float, size_btc: float):
    high = bars["high"].values
    low = bars["low"].values
    close = bars["close"].values
    timestamps = bars["timestamp"].values
    trade_log = []

    for idx, direction, box_high, box_low in breakouts:
        entry_price = close[idx]
        sl_price = entry_price - sl_usdt if direction == "BUY" else entry_price + sl_usdt
        tp_price = entry_price + tp_usdt if direction == "BUY" else entry_price - tp_usdt

        for k in range(idx + 1, len(bars)):
            if direction == "BUY":
                if low[k] <= sl_price:
                    trade_log.append((timestamps[idx], timestamps[k], direction, "Stop Loss",
                                       (sl_price - entry_price) * size_btc))
                    break
                elif high[k] >= tp_price:
                    trade_log.append((timestamps[idx], timestamps[k], direction, "Take Profit",
                                       (tp_price - entry_price) * size_btc))
                    break
            else:
                if high[k] >= sl_price:
                    trade_log.append((timestamps[idx], timestamps[k], direction, "Stop Loss",
                                       (entry_price - sl_price) * size_btc))
                    break
                elif low[k] <= tp_price:
                    trade_log.append((timestamps[idx], timestamps[k], direction, "Take Profit",
                                       (entry_price - tp_price) * size_btc))
                    break

    return pd.DataFrame(trade_log, columns=["entry_time", "exit_time", "side", "reason", "pnl"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--sl", type=float, default=400.0)
    parser.add_argument("--tp", type=float, default=700.0)
    parser.add_argument("--size", type=float, default=1.0)
    parser.add_argument("--range-ratio", type=float, default=2.5,
                         help="Max allowed ratio between largest and smallest candle range in the box")
    args = parser.parse_args()

    bars = load_and_resample_15m(args.parquet, args.days)

    print(f"Scanning for 4-candle boxes (range ratio <= {args.range_ratio}x) and their breakouts...")
    breakouts = find_box_breakouts(bars, args.range_ratio)
    print(f"Found {len(breakouts)} valid box breakouts")

    trades = run_backtest(bars, breakouts, args.sl, args.tp, args.size)

    print("\n" + "=" * 70)
    print("BACKTEST REPORT -- 4-Candle Box Breakout Pattern (standalone)")
    print("=" * 70)
    print(f"Period: {bars['timestamp'].iloc[0]} to {bars['timestamp'].iloc[-1]}")
    print(f"Stop: ${args.sl} | Target: ${args.tp} | Size: {args.size} BTC | Range ratio max: {args.range_ratio}x")

    if trades.empty:
        print("\nNo trades triggered.")
        return

    wins = trades[trades["pnl"] > 0]
    win_rate = len(wins) / len(trades) * 100
    total_pnl = trades["pnl"].sum()
    cum = trades.sort_values("exit_time")["pnl"].cumsum()
    max_dd = (cum - cum.cummax()).min()
    rr = args.tp / args.sl
    breakeven = 1 / (1 + rr) * 100

    print(f"\nTotal trades: {len(trades)}")
    print(f"Wins:

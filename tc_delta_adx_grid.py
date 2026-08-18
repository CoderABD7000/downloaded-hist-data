"""
btc_delta_adx_grid.py

Grid search over SL/TP combinations for the ORIGINAL, unmodified
delta+ADX signal -- no filters (RSI, POC, etc. all tested and found
to hurt performance, so back to the validated baseline signal only).

Tests every combination of:
  SL: 300, 350, 400, 450, 500, 550, 600, 650
  TP: 500, 550, 600, 650, 700, 750, 800, 850
(64 combinations total)

The signal (delta+ADX, same thresholds, same dedup state machine) is
computed ONCE -- only the SL/TP exit levels change per grid cell, so
this is far faster than re-running the full script 64 times.

USAGE:
    python btc_delta_adx_grid.py
"""

import argparse
import numpy as np
import pandas as pd

DEFAULT_PARQUET = r"D:\data\btc\BTCUSDT_1m_ohlcv_oi_delta_cvd_5y.parquet"
WEIGHTS = dict(delta=8, adx=35)
TH = dict(adx_period=14, adx_min=18, buy=58, sell=42, min_gap=6, cooldown=2, pressure_smooth=3)
POSITION_SIZE_BTC = 1.0

SL_GRID = [300, 350, 400, 450, 500, 550, 600, 650]
TP_GRID = [500, 550, 600, 650, 700, 750, 800, 850]


def load_and_resample_15m(parquet_path: str, days=None) -> pd.DataFrame:
    print(f"Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    if days is not None:
        cutoff = df.index.max() - pd.Timedelta(days=days)
        df = df[df.index >= cutoff]
    agg = df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "taker_buy_vol": "sum",
    }).dropna()
    agg = agg.rename(columns={"taker_buy_vol": "taker_buy_base"})
    agg = agg.reset_index()
    print(f"Resampled to {len(agg):,} x 15m candles, {agg['timestamp'].iloc[0]} to {agg['timestamp'].iloc[-1]}")
    return agg[["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]]


def wilder_smooth(x, period):
    out = np.full(len(x), np.nan)
    if len(x) < period:
        return out
    out[period - 1] = np.nansum(x[:period])
    for i in range(period, len(x)):
        out[i] = out[i - 1] - (out[i - 1] / period) + x[i]
    return out


def compute_adx(H, L, C, period=14):
    n = len(H)
    TR = np.zeros(n); pDM = np.zeros(n); nDM = np.zeros(n)
    TR[0] = H[0] - L[0]
    for i in range(1, n):
        up, down = H[i] - H[i - 1], L[i - 1] - L[i]
        pDM[i] = up if (up > down and up > 0) else 0
        nDM[i] = down if (down > up and down > 0) else 0
        TR[i] = max(H[i] - L[i], abs(H[i] - C[i - 1]), abs(L[i] - C[i - 1]))
    pS, nS, trS = wilder_smooth(pDM, period), wilder_smooth(nDM, period), wilder_smooth(TR, period)
    diPlus = np.zeros(n); diMinus = np.zeros(n); dx = np.zeros(n)
    valid = (~np.isnan(trS)) & (trS > 0)
    diPlus[valid] = 100 * pS[valid] / trS[valid]
    diMinus[valid] = 100 * nS[valid] / trS[valid]
    tot = diPlus + diMinus
    nz = tot != 0
    dx[nz] = 100 * np.abs(diPlus[nz] - diMinus[nz]) / tot[nz]
    adx = wilder_smooth(dx, period) / period
    return diPlus, diMinus, adx


def ema(arr, span):
    out = np.empty(len(arr)); out[0] = arr[0]
    alpha = 2 / (span + 1)
    for i in range(1, len(arr)):
        out[i] = (arr[i] - out[i - 1]) * alpha + out[i - 1]
    return out


def compute_signal(bars: pd.DataFrame):
    """Original, unmodified delta+ADX signal -- no filters."""
    H, L, C, V = bars["high"].values, bars["low"].values, bars["close"].values, bars["volume"].values
    TB = bars["taker_buy_base"].values

    diPlus, diMinus, adx = compute_adx(H, L, C, TH["adx_period"])

    buy_pct = np.where(V > 0, np.clip(TB / np.where(V == 0, 1, V), 0, 1), 0.5)
    adx_val = np.clip(np.nan_to_num(adx, nan=0) / 50, 0, 1)
    tot_di = diPlus + diMinus
    di_ratio = np.where(tot_di > 0, diPlus / np.where(tot_di == 0, 1, tot_di), 0.5)
    adx_contrib = 0.5 + (di_ratio - 0.5) * adx_val

    total_w = WEIGHTS["delta"] + WEIGHTS["adx"]
    score = buy_pct * WEIGHTS["delta"] + adx_contrib * WEIGHTS["adx"]
    pressure_raw = score / total_w * 100
    pressure = ema(pressure_raw, TH["pressure_smooth"])

    ok_adx = (~np.isnan(adx)) & (adx >= TH["adx_min"])
    gap_ok = np.abs(pressure - 50) >= TH["min_gap"]
    buy_raw = ok_adx & gap_ok & (pressure >= TH["buy"])
    sell_raw = ok_adx & gap_ok & (pressure <= TH["sell"])

    return buy_raw, sell_raw, C


class BotState:
    def __init__(self):
        self.last_dir = None
        self.bars_since = TH["cooldown"]
        self.last_dot = None

    def apply_final_type_rule(self, base_type: str) -> str:
        if base_type == "OTHER":
            self.bars_since += 1
            final = "OTHER"
        elif self.last_dir is None or base_type != self.last_dir:
            if self.bars_since >= TH["cooldown"]:
                self.last_dir, self.bars_since = base_type, 0
                final = base_type
            else:
                self.bars_since += 1
                final = "OTHER"
        else:
            final = base_type
            self.bars_since = 0

        c = "G" if final == "BUY" else ("R" if final == "SELL" else "Y")
        if c in ("G", "R") and c == self.last_dot:
            c = "Y"
        self.last_dot = c
        return c


def get_signal_directions(buy_raw, sell_raw, warmup):
    """Pre-compute the dedup'd signal direction for every candle ONCE --
    this is identical regardless of SL/TP, so it only needs to run once
    for the whole grid search, not once per grid cell."""
    n = len(buy_raw)
    directions = [None] * n
    state = BotState()
    for i in range(warmup, n):
        base_type = "BUY" if buy_raw[i] else ("SELL" if sell_raw[i] else "OTHER")
        dot = state.apply_final_type_rule(base_type)
        directions[i] = "BUY" if dot == "G" else ("SELL" if dot == "R" else None)
    return directions


def run_backtest_fixed_signal(bars, directions, close, sl_usdt, tp_usdt, warmup):
    """Replays the SAME pre-computed signal directions against a given
    SL/TP pair. Only entry/exit bookkeeping changes per grid cell.
    Uses raw numpy arrays throughout (no pandas .iloc in the loop) --
    .iloc has meaningful per-call overhead that adds up significantly
    across 64 grid cells x 176k candles each."""
    high = bars["high"].values
    low = bars["low"].values
    timestamps = bars["timestamp"].values
    trade_log = []
    open_pos = None

    for i in range(warmup, len(close)):
        close_price = close[i]

        if open_pos is not None:
            closed, pnl = False, 0.0
            if open_pos["side"] == "BUY":
                if low[i] <= open_pos["sl"]:
                    closed = True
                    pnl = (open_pos["sl"] - open_pos["entry_price"]) * open_pos["qty"]
                elif high[i] >= open_pos["tp"]:
                    closed = True
                    pnl = (open_pos["tp"] - open_pos["entry_price"]) * open_pos["qty"]
            else:
                if high[i] >= open_pos["sl"]:
                    closed = True
                    pnl = (open_pos["entry_price"] - open_pos["sl"]) * open_pos["qty"]
                elif low[i] <= open_pos["tp"]:
                    closed = True
                    pnl = (open_pos["entry_price"] - open_pos["tp"]) * open_pos["qty"]
            if closed:
                trade_log.append((timestamps[i], pnl))
                open_pos = None

        signal_dir = directions[i]
        if signal_dir and open_pos is None:
            sl_price = close_price - sl_usdt if signal_dir == "BUY" else close_price + sl_usdt
            tp_price = close_price + tp_usdt if signal_dir == "BUY" else close_price - tp_usdt
            open_pos = dict(side=signal_dir, qty=POSITION_SIZE_BTC, entry_price=close_price,
                             sl=sl_price, tp=tp_price)

    return pd.DataFrame(trade_log, columns=["exit_time", "pnl"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET)
    parser.add_argument("--days", type=int, default=None)
    args = parser.parse_args()

    bars = load_and_resample_15m(args.parquet, args.days)
    buy_raw, sell_raw, close = compute_signal(bars)
    warmup = TH["adx_period"] + 5

    print("Pre-computing signal directions (once, shared across the whole grid)...")
    directions = get_signal_directions(buy_raw, sell_raw, warmup)

    results = []
    total = len(SL_GRID) * len(TP_GRID)
    count = 0
    for sl in SL_GRID:
        for tp in TP_GRID:
            count += 1
            trades = run_backtest_fixed_signal(bars, directions, close, sl, tp, warmup)
            if trades.empty:
                continue
            wins = trades[trades["pnl"] > 0]
            win_rate = len(wins) / len(trades) * 100
            total_pnl = trades["pnl"].sum()
            cum = trades["pnl"].cumsum()
            max_dd = (cum - cum.cummax()).min()
            rr = tp / sl
            breakeven = 1 / (1 + rr) * 100
            edge = win_rate - breakeven
            results.append(dict(sl=sl, tp=tp, rr=rr, trades=len(trades), win_rate=win_rate,
                                 total_pnl=total_pnl, max_dd=max_dd, breakeven=breakeven, edge=edge))
            print(f"  [{count}/{total}] SL={sl} TP={tp}: {len(trades)} trades, "
                  f"{win_rate:.1f}% WR, ${total_pnl:,.0f} PnL")

    results_df = pd.DataFrame(results).sort_values("total_pnl", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 100)
    print("GRID SEARCH RESULTS -- sorted by Total PnL (best first)")
    print("=" * 100)
    print(f"{'SL':>6} {'TP':>6} {'R:R':>6} {'Trades':>8} {'WinRate':>9} {'TotalPnL':>14} {'MaxDD':>12} {'Edge(pts)':>10}")
    print("-" * 100)
    for _, r in results_df.iterrows():
        print(f"{r['sl']:>6.0f} {r['tp']:>6.0f} {r['rr']:>6.2f} {r['trades']:>8.0f} "
              f"{r['win_rate']:>8.1f}% ${r['total_pnl']:>12,.0f} ${r['max_dd']:>10,.0f} {r['edge']:>+9.1f}")

    results_df.to_csv("grid_search_results.c

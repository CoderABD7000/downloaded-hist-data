"""
btc_triple_poc_adx.py

Entry only when price closes above (or below) ALL THREE POC levels
simultaneously -- Weekly, Daily, AND 4H -- confirmed by ADX.

POC calculation is the exact, already-validated no-look-ahead engine
from build_poc_dataset.py:
  - Weekly POC: previous week's completed POC, constant for the ENTIRE
    current week, replaced at each new week (Monday 00:00 UTC).
  - Daily POC: previous day's completed POC, constant for the entire
    current day.
  - 4H POC: previous 4H block's completed POC. Six blocks/day, first
    starting 00:00 UTC. Each row uses whichever of the 6 4H-POC values
    is valid for its own time slot.

Entry:
  BUY  when close > W-POC AND close > D-POC AND close > active 4H-POC
       AND ADX confirms
  SELL when close < W-POC AND close < D-POC AND close < active 4H-POC
       AND ADX confirms

Same validated exit mechanism (fixed $ SL/TP, dedup/cooldown state
machine) as the other strategies tested today.

USAGE:
    python btc_triple_poc_adx.py --sl 400 --tp 750
"""

import argparse
from datetime import timedelta

import numpy as np
import pandas as pd

DEFAULT_PARQUET = r"D:\data\btc\BTCUSDT_1m_ohlcv_oi_delta_cvd_5y.parquet"
BIN_SIZE = 25.0
ADX_MIN = 18
COOLDOWN_BARS = 2
SL_USDT = None
TP_USDT = None
POSITION_SIZE_BTC = None


def compute_poc(df_1m: pd.DataFrame, bin_size: float):
    if df_1m.empty:
        return None
    lo = np.floor(df_1m["low"].values / bin_size).astype(np.int64)
    hi = np.floor(df_1m["high"].values / bin_size).astype(np.int64)
    vol = df_1m["volume"].values
    volume_by_bin = {}
    for l, h, v in zip(lo, hi, vol):
        n_bins = h - l + 1
        if n_bins <= 0:
            continue
        share = v / n_bins
        for b in range(l, h + 1):
            volume_by_bin[b] = volume_by_bin.get(b, 0.0) + share
    if not volume_by_bin:
        return None
    best_bin = max(volume_by_bin, key=volume_by_bin.get)
    return best_bin * bin_size + bin_size / 2.0


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


def build_dataset(parquet_path: str, days=None) -> pd.DataFrame:
    print(f"Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if days is not None:
        cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    print("Computing Weekly POCs...")
    df["week_start"] = df["timestamp"].dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")
    week_pocs = {}
    for wk, grp in df.groupby("week_start"):
        week_pocs[wk] = compute_poc(grp, BIN_SIZE)
    week_to_next_week_poc = {wk + timedelta(days=7): poc for wk, poc in week_pocs.items()}

    print("Computing Daily POCs...")
    df["day_start"] = df["timestamp"].dt.floor("D")
    day_pocs = {}
    for day, grp in df.groupby("day_start"):
        day_pocs[day] = compute_poc(grp, BIN_SIZE)
    day_to_next_day_poc = {d + timedelta(days=1): poc for d, poc in day_pocs.items()}

    print("Computing 4H POCs...")
    df["block_start"] = df["timestamp"].dt.floor("4h")
    block_pocs = {}
    for blk, grp in df.groupby("block_start"):
        block_pocs[blk] = compute_poc(grp, BIN_SIZE)
    block_to_next_block_poc = {blk + timedelta(hours=4): poc for blk, poc in block_pocs.items()}

    df_idx = df.set_index("timestamp")
    agg = df_idx.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna().reset_index()

    agg["week_start"] = agg["timestamp"].dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")
    agg["W_POC"] = agg["week_start"].map(week_to_next_week_poc)

    agg["day_start"] = agg["timestamp"].dt.floor("D")
    agg["D_POC"] = agg["day_start"].map(day_to_next_day_poc)

    agg["block_start"] = agg["timestamp"].dt.floor("4h")
    agg["H4_POC"] = agg["block_start"].map(block_to_next_block_poc)

    agg = agg.dropna(subset=["W_POC", "D_POC", "H4_POC"]).reset_index(drop=True)

    print(f"Resampled to {len(agg):,} x 15m candles with all 3 POCs valid, "
          f"{agg['timestamp'].iloc[0]} to {agg['timestamp'].iloc[-1]}")
    return agg


class BotState:
    def __init__(self):
        self.last_dir = None
        self.bars_since = COOLDOWN_BARS
        self.last_dot = None

    def apply_final_type_rule(self, base_type: str) -> str:
        if base_type == "OTHER":
            self.bars_since += 1
            final = "OTHER"
        elif self.last_dir is None or base_type != self.last_dir:
            if self.bars_since >= COOLDOWN_BARS:
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


def run_backtest(bars: pd.DataFrame):
    close = bars["close"].values
    high = bars["high"].values
    low = bars["low"].values
    w_poc = bars["W_POC"].values
    d_poc = bars["D_POC"].values
    h4_poc = bars["H4_POC"].values

    _, _, adx = compute_adx(high, low, close, period=14)

    state = BotState()
    trade_log = []
    open_pos = None
    warmup = 20

    for i in range(warmup, len(bars)):
        close_price = close[i]
        ts_now = bars["timestamp"].iloc[i]

        if open_pos is not None:
            closed, pnl, reason, exit_price = False, 0.0, "", close_price
            if open_pos["side"] == "BUY":
                if low[i] <= open_pos["sl"]:
                    closed, reason, exit_price = True, "Stop Loss", open_pos["sl"]
                    pnl = (open_pos["sl"] - open_pos["entry_price"]) * open_pos["qty"]
                elif high[i] >= open_pos["tp"]:
                    closed, reason, exit_price = True, "Take Profit", open_pos["tp"]
                    pnl = (open_pos["tp"] - open_pos["entry_price"]) * open_pos["qty"]
            else:
                if high[i] >= open_pos["sl"]:
                    closed, reason, exit_price = True, "Stop Loss", open_pos["sl"]
                    pnl = (open_pos["entry_price"] - open_pos["sl"]) * open_pos["qty"]
                elif low[i] <= open_pos["tp"]:
                    closed, reason, exit_price = True, "Take Profit", open_pos["tp"]
                    pnl = (open_pos["entry_price"] - open_pos["tp"]) * open_pos["qty"]
            if closed:
                trade_log.append(dict(entry_time=open_pos["entry_time"], exit_time=ts_now, side=open_pos["side"],
                                       entry=open_pos["entry_price"], exit=exit_price, reason=reason, pnl=pnl))
                open_pos = None

        base_type = "OTHER"
        if not np.isnan(adx[i]) and adx[i] >= ADX_MIN:
            above_all = close_price > w_poc[i] and close_price > d_poc[i] and close_price > h4_poc[i]
            below_all = close_price < w_poc[i] and close_price < d_poc[i] and close_price < h4_poc[i]
            if above_all:
                base_type = "BUY"
            elif below_all:
                base_type = "SELL"

        dot = state.apply_final_type_rule(base_type)
        signal_dir = "BUY" if dot == "G" else ("SELL" if dot == "R" else None)

        if signal_dir and open_pos is None:
            sl_price = close_price - SL_USDT if signal_dir == "BUY" else close_price + SL_USDT
            tp_price = close_price + TP_USDT if signal_dir == "BUY" else close_price - TP_USDT
            open_pos = dict(side=signal_dir, qty=POSITION_SIZE_BTC, entry_price=close_price,
                             sl=sl_price, tp=tp_price, entry_time=ts_now)

    return pd.DataFrame(trade_log)


def print_report(trades: pd.DataFrame, bars: pd.DataFrame):
    print("\n" + "=" * 70)
    print("BACKTEST REPORT -- BTC Triple-POC (W/D/4H) + ADX Strategy")
    print("=" * 70)
    print(f"Period: {bars['timestamp'].iloc[0]} to {bars['timestamp'].iloc[-1]}")
    print(f"Candles: {len(bars)} (15m) | Stop: ${SL_USDT} | Target: ${TP_USDT} | Size: {POSITION_SIZE_BTC} BTC")
    if trades.empty:
        print("\nNo trades triggered over this period.")
        return

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    print(f"\nTotal trades: {len(trades)}")
    print(f"Wins: {len(wins)} | Losses: {len(losses)} | Win rate: {len(wins)/len(trades)*100:.1f}%")
    print(f"Total PnL: ${trades['pnl'].sum():,.2f}")
    print(f"\nBy exit reason:")
    print(trades.groupby("reason")["pnl"].agg(["count", "sum", "mean"]))
    print(f"\nBy side:")
    print(trades.groupby("side")["pnl"].agg(["count", "sum", "mean"]))

    trades_sorted = trades.sort_values("exit_time").reset_index(drop=True)
    cum = trades_sorted["pnl"].cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    print(f"\nMax drawdown: ${drawdown.min():,.2f}")
    print(f"Final cumulative PnL: ${cum.iloc[-1]:,.2f}")

    rr = TP_USDT / SL_USDT
    breakeven_wr = 1 / (1 + rr) * 100
    actual_wr = len(wins) / len(trades) * 100
    print(f"\nBreakeven win rate at this {rr:.2f}:1 R:R: {breakeven_wr:.1f}%")
    print(f"Actual win rate: {actual_wr:.1f}% ({'ABOVE' if actual_wr > breakeven_wr else 'BELOW'} breakeven by {abs(actual_wr-breakeven_wr):.1f} points)")

    trades_sorted.to_csv("triple_poc_adx_trades.csv", index=False)
    print(f"\nFull trade log written to triple_poc_adx_trades.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--sl", type=float, default=400.0)
    parser.add_argument("--tp", type=float, default=750.0)
    parser.add_argument("--size", type=float, default=1.0)
    args = parser.parse_args()

    SL_USDT = args.sl
    TP_USDT = args.tp
    POSITION_SIZE_BTC = args.size

    bars = build_dataset(args.parquet, args.days)
    trades = run_backtest(bars)
    print_report(trades, bars)

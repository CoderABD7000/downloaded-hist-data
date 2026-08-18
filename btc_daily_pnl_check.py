"""
btc_daily_pnl_check.py

Shows the REAL day-by-day PnL distribution for the delta+ADX strategy
at a given SL/TP and position size -- not just the average, since
averages hide crucial variance: how many days are $0 (no trade
closed), how many are losing days, and what a realistic "typical"
day actually looks like versus the average.

USAGE:
    python btc_daily_pnl_check.py --sl 400 --tp 700 --size 1.0
    python btc_daily_pnl_check.py --sl 400 --tp 700 --size 6.1
"""

import argparse
import numpy as np
import pandas as pd

DEFAULT_PARQUET = r"D:\data\btc\BTCUSDT_1m_ohlcv_oi_delta_cvd_5y.parquet"
WEIGHTS = dict(delta=8, adx=35)
TH = dict(adx_period=14, adx_min=18, buy=58, sell=42, min_gap=6, cooldown=2, pressure_smooth=3)


def load_and_resample_15m(parquet_path, days=None):
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
    return agg.reset_index()


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


def compute_signal(bars):
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
    pressure = ema(score / total_w * 100, TH["pressure_smooth"])
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

    def apply_final_type_rule(self, base_type):
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


def run_backtest(bars, buy_raw, sell_raw, close, sl_usdt, tp_usdt, size_btc):
    state = BotState()
    trade_log = []
    open_pos = None
    high = bars["high"].values
    low = bars["low"].values
    timestamps = bars["timestamp"].values
    warmup = TH["adx_period"] + 5

    for i in range(warmup, len(close)):
        close_price = close[i]
        if open_pos is not None:
            closed, pnl = False, 0.0
            if open_pos["side"] == "BUY":
                if low[i] <= open_pos["sl"]:
                    closed = True; pnl = (open_pos["sl"] - open_pos["entry_price"]) * size_btc
                elif high[i] >= open_pos["tp"]:
                    closed = True; pnl = (open_pos["tp"] - open_pos["entry_price"]) * size_btc
            else:
                if high[i] >= open_pos["sl"]:
                    closed = True; pnl = (open_pos["entry_price"] - open_pos["sl"]) * size_btc
                elif low[i] <= open_pos["tp"]:
                    closed = True; pnl = (open_pos["entry_price"] - open_pos["tp"]) * size_btc
            if closed:
                trade_log.append((timestamps[i], pnl))
                open_pos = None
        base_type = "BUY" if buy_raw[i] else ("SELL" if sell_raw[i] else "OTHER")
        dot = state.apply_final_type_rule(base_type)
        signal_dir = "BUY" if dot == "G" else ("SELL" if dot == "R" else None)
        if signal_dir and open_pos is None:
            sl_price = close_price - sl_usdt if signal_dir == "BUY" else close_price + sl_usdt
            tp_price = close_price + tp_usdt if signal_dir == "BUY" else close_price - tp_usdt
            open_pos = dict(side=signal_dir, entry_price=close_price, sl=sl_price, tp=tp_price)
    return pd.DataFrame(trade_log, columns=["exit_time", "pnl"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--sl", type=float, default=400.0)
    parser.add_argument("--tp", type=float, default=700.0)
    parser.add_argument("--size", type=float, default=1.0)
    args = parser.parse_args()

    bars = load_and_resample_15m(args.parquet, args.days)
    buy_raw, sell_raw, close = compute_signal(bars)
    trades = run_backtest(bars, buy_raw, sell_raw, close, args.sl, args.tp, args.size)

    trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    trades["date"] = trades["exit_time"].dt.date

    full_range = pd.date_range(trades["exit_time"].min().date(), trades["exit_time"].max().date(), freq="D")
    daily_pnl = trades.groupby("date")["pnl"].sum()
    daily_pnl = daily_pnl.reindex(full_range.date, fill_value=0.0)

    print("\n" + "=" * 70)
    print(f"DAILY PnL DISTRIBUTION -- SL=${args.sl} TP=${args.tp} Size={args.size} BTC")
    print("=" * 70)
    print(f"Total days in period: {len(daily_pnl)}")
    print(f"Average PnL/day: ${daily_pnl.mean():,.2f}")
    print(f"Median PnL/day: ${daily_pnl.median():,.2f}")
    print(f"Days with $0 (no trade closed): {(daily_pnl == 0).sum()} ({(daily_pnl==0).mean()*100:.1f}%)")
    print(f"Days with negative PnL: {(daily_pnl < 0).sum()} ({(daily_pnl<0).mean()*100:.1f}%)")
    print(f"Days with positive PnL: {(daily_pnl > 0).sum()} ({(daily_pnl>0).mean()*100:.1f}%)")
    print(f"Days >= $500: {(daily_pnl >= 500).sum()} ({(daily_pnl>=500).mean()*100:.1f}%)")
    print(f"Worst single day: ${daily_pnl.min():,.2f}")
    print(f"Best single day: ${daily_pnl.max():,.2f}")
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  {p}th percentile: ${daily_pnl.quantile(p/100):,.2f}")

    daily_pnl.to_csv("daily_pnl.csv", header=["pnl"])
    print("\nFull daily PnL series written to daily_pnl.csv")


if __name__ == "__main__":
    main()

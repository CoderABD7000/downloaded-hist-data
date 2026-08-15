"""
backtest_btc_bot_v3.py

Standalone backtest of BTC_BOT_V3's exact live_trader01.py logic against
real historical Binance klines.

This is NOT a simplification -- it replicates:
  - The exact pressure-score formula (including the oi/liq dead-weight
    bug found in the live code)
  - The exact BotState.apply_final_type_rule() dedup/cooldown state
    machine (only fires once per fresh direction streak)
  - Fixed $500/$750 stop/target (not ATR-based)
  - The portfolio-level profit lock (arm at $100, flatten on $50 giveback)
  - The 3-consecutive-loss breaker (30-bar cooldown per side)
  - 6-position concurrency cap
  - 0.1 BTC per trade
"""

import argparse
from collections import deque

import numpy as np
import pandas as pd

SYMBOL = "BTCUSDT"
INTERVAL = "15m"

WEIGHTS = dict(delta=8, oi=5, adx=35, liq=2.5, vwap=30, cvd=26)
TH = dict(adx_period=14, adx_min=18, buy=58, sell=42, min_gap=6, cooldown=2,
          pressure_smooth=3, cvd_period=50,
          breaker_loss_count=3, breaker_cooldown_bars=30,
          sl_usdt=500, tp_usdt=750)
POSITION_SIZE_BTC = 0.1
MAX_CONCURRENT_POSITIONS = 6
PROFIT_LOCK_ARM_USD = 100.0
PROFIT_LOCK_GIVEBACK_USD = 50.0


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
    print(f"Resampled to {len(agg)} x 15m candles, {agg['timestamp'].iloc[0]} to {agg['timestamp'].iloc[-1]}")
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


def compute_vwap_dist(H, L, C, V, dates):
    n = len(C)
    tp = (H + L + C) / 3
    dist = np.zeros(n)
    cum_vol = cum_pv = 0.0
    cur_day = None
    for i in range(n):
        if dates[i] != cur_day:
            cum_vol = cum_pv = 0.0
            cur_day = dates[i]
        cum_vol += V[i]; cum_pv += tp[i] * V[i]
        vwap = cum_pv / cum_vol if cum_vol > 0 else C[i]
        dist[i] = (C[i] - vwap) / vwap * 100
    return dist


def ema(arr, span):
    out = np.empty(len(arr)); out[0] = arr[0]
    alpha = 2 / (span + 1)
    for i in range(1, len(arr)):
        out[i] = (arr[i] - out[i - 1]) * alpha + out[i - 1]
    return out


def compute_pressure_and_signal(bars: pd.DataFrame):
    H, L, C, V = bars["high"].values, bars["low"].values, bars["close"].values, bars["volume"].values
    TB = bars["taker_buy_base"].values

    delta_arr = 2 * TB - V
    cvd = np.cumsum(delta_arr)

    diPlus, diMinus, adx = compute_adx(H, L, C, TH["adx_period"])
    dates = bars["timestamp"].dt.date.values
    vwap_dist = compute_vwap_dist(H, L, C, V, dates)

    W = dict(WEIGHTS)
    total_w = sum(W.values())

    buy_pct = np.where(V > 0, np.clip(TB / np.where(V == 0, 1, V), 0, 1), 0.5)
    adx_val = np.clip(np.nan_to_num(adx, nan=0) / 50, 0, 1)
    tot_di = diPlus + diMinus
    di_ratio = np.where(tot_di > 0, diPlus / np.where(tot_di == 0, 1, tot_di), 0.5)
    adx_contrib = 0.5 + (di_ratio - 0.5) * adx_val
    vwap_s = np.clip(0.5 + (vwap_dist / 0.5) * 0.5, 0, 1)

    period = TH["cvd_period"]
    cvd_series = pd.Series(cvd)
    roll_mean = cvd_series.rolling(period, min_periods=period).mean().values
    roll_std = cvd_series.rolling(period, min_periods=period).std().values
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(roll_std > 0, (cvd - roll_mean) / roll_std, 0)
    z = np.nan_to_num(z, nan=0.0)
    cvd_s = np.clip(0.5 + z * 0.2, 0, 1)

    score = (buy_pct * W["delta"] + adx_contrib * W["adx"] + vwap_s * W["vwap"] + cvd_s * W["cvd"])

    pressure_raw = score / total_w * 100
    pressure = ema(pressure_raw, TH["pressure_smooth"])

    ok_adx = (~np.isnan(adx)) & (adx >= TH["adx_min"])
    gap_ok = np.abs(pressure - 50) >= TH["min_gap"]
    buy_raw = ok_adx & gap_ok & (pressure >= TH["buy"])
    sell_raw = ok_adx & gap_ok & (pressure <= TH["sell"])
    vwap_rel = np.where(vwap_dist > 0.05, "Above", np.where(vwap_dist < -0.05, "Below", "At"))
    buy_raw &= (vwap_rel != "Below")
    sell_raw &= (vwap_rel != "Above")

    return dict(buy_raw=buy_raw, sell_raw=sell_raw, pressure=pressure, close=C)


class BotState:
    def __init__(self):
        self.last_dir = None
        self.bars_since = TH["cooldown"]
        self.last_dot = None
        self.breaker_cd = {"BUY": 0, "SELL": 0}
        self.breaker_consec = {"BUY": 0, "SELL": 0}
        self.open_positions = deque()
        self.realized_pnl = 0.0
        self.floating_pnl_peak = 0.0
        self.profit_lock_armed = False

    def decrement_cooldowns(self):
        for k in ("BUY", "SELL"):
            if self.breaker_cd[k] > 0:
                self.breaker_cd[k] -= 1

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


def compute_floating_pnl(positions, current_price):
    return sum(
        (current_price - p["entry_price"]) * p["qty"] * (1 if p["side"] == "BUY" else -1)
        for p in positions
    )


def run_backtest(bars: pd.DataFrame):
    feats = compute_pressure_and_signal(bars)
    state = BotState()
    trade_log = []
    warmup = max(TH["adx_period"], TH["cvd_period"]) + 5

    for i in range(warmup, len(bars)):
        close_price = feats["close"][i]
        ts_now = bars["timestamp"].iloc[i]

        base_type = "BUY" if feats["buy_raw"][i] else ("SELL" if feats["sell_raw"][i] else "OTHER")
        state.decrement_cooldowns()
        dot = state.apply_final_type_rule(base_type)

        remaining = deque()
        for pos in state.open_positions:
            closed, pnl, reason, exit_price = False, 0.0, "", close_price
            if pos["side"] == "BUY":
                if bars["low"].iloc[i] <= pos["sl"]:
                    closed, reason, exit_price = True, "Stop Loss", pos["sl"]
                    pnl = (pos["sl"] - pos["entry_price"]) * pos["qty"]
                elif bars["high"].iloc[i] >= pos["tp"]:
                    closed, reason, exit_price = True, "Take Profit", pos["tp"]
                    pnl = (pos["tp"] - pos["entry_price"]) * pos["qty"]
            else:
                if bars["high"].iloc[i] >= pos["sl"]:
                    closed, reason, exit_price = True, "Stop Loss", pos["sl"]
                    pnl = (pos["entry_price"] - pos["sl"]) * pos["qty"]
                elif bars["low"].iloc[i] <= pos["tp"]:
                    closed, reason, exit_price = True, "Take Profit", pos["tp"]
                    pnl = (pos["entry_price"] - pos["tp"]) * pos["qty"]

            if closed:
                state.realized_pnl += pnl
                if pnl < 0:
                    state.breaker_consec[pos["side"]] += 1
                    if state.breaker_consec[pos["side"]] >= TH["breaker_loss_count"]:
                        state.breaker_cd[pos["side"]] = TH["breaker_cooldown_bars"]
                else:
                    state.breaker_consec[pos["side"]] = 0
                trade_log.append(dict(entry_time=pos["entry_time"], exit_time=ts_now, side=pos["side"],
                                       entry=pos["entry_price"], exit=exit_price, reason=reason, pnl=pnl))
            else:
                remaining.append(pos)
        state.open_positions = remaining

        signal_dir = "BUY" if dot == "G" else ("SELL" if dot == "R" else None)
        if signal_dir and state.breaker_cd[signal_dir] == 0 and len(state.open_positions) < MAX_CONCURRENT_POSITIONS:
            entry_side = signal_dir
            sl_price = close_price - TH["sl_usdt"] if entry_side == "BUY" else close_price + TH["sl_usdt"]
            tp_price = close_price + TH["tp_usdt"] if entry_side == "BUY" else close_price - TH["tp_usdt"]
            state.open_positions.append(dict(side=entry_side, qty=POSITION_SIZE_BTC, entry_price=close_price,
                                              sl=sl_price, tp=tp_price, entry_time=ts_now))

        unrealized = compute_floating_pnl(state.open_positions, close_price)
        if not state.open_positions:
            state.floating_pnl_peak = 0.0
            state.profit_lock_armed = False
        else:
            if unrealized > state.floating_pnl_peak:
                state.floating_pnl_peak = unrealized
            if state.floating_pnl_peak >= PROFIT_LOCK_ARM_USD:
                state.profit_lock_armed = True
            giveback = state.floating_pnl_peak - unrealized
            if state.profit_lock_armed and giveback >= PROFIT_LOCK_GIVEBACK_USD:
                for pos in state.open_positions:
                    pnl = (close_price - pos["entry_price"]) * pos["qty"] if pos["side"] == "BUY" else (pos["entry_price"] - close_price) * pos["qty"]
                    state.realized_pnl += pnl
                    trade_log.append(dict(entry_time=pos["entry_time"], exit_time=ts_now, side=pos["side"],
                                           entry=pos["entry_price"], exit=close_price, reason="Profit Lock", pnl=pnl))
                state.open_positions = deque()
                state.floating_pnl_peak = 0.0
                state.profit_lock_armed = False

    return pd.DataFrame(trade_log), state


def print_report(trades: pd.DataFrame, bars: pd.DataFrame):
    print("\n" + "=" * 60)
    print("BACKTEST REPORT -- BTC_BOT_V3 exact logic replay")
    print("=" * 60)
    print(f"Period: {bars['timestamp'].iloc[0]} to {bars['timestamp'].iloc[-1]}")
    print(f"Candles: {len(bars)} ({INTERVAL})")
    if trades.empty:
        print("\nNo trades triggered over this period.")
        return

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    print(f"\nTotal trades: {len(trades)}")
    print(f"Wins: {len(wins)} | Losses: {len(losses)} | Win rate: {len(wins)/len(trades)*100:.1f}%")
    print(f"Total PnL: ${trades['pnl'].sum():,.2f}")
    print(f"Avg win: ${wins['pnl'].mean():,.2f}" if len(wins) else "Avg win: n/a")
    print(f"Avg loss: ${losses['pnl'].mean():,.2f}" if len(losses) else "Avg loss: n/a")
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

    trades_sorted.to_csv("backtest_trades.csv", index=False)
    print(f"\nFull trade log written to backtest_trades.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=r"D:\data\btc\BTCUSDT_1m_ohlcv_oi_delta_cvd_5y.parquet")
    parser.add_argument("--days", type=int, default=None)
    args = parser.parse_args()

    bars = load_and_resample_15m(args.parquet, args.days)
    trades, final_state = run_backtest(bars)
    print_report(trades, bars)

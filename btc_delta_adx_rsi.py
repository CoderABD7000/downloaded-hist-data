"""
btc_delta_adx_rsi.py

Adds an RSI EXHAUSTION filter on top of the validated delta+ADX
strategy: block BUY entries when RSI > 70 (already overbought, move
may be overextended), block SELL entries when RSI < 30 (already
oversold). This is deliberately NOT a redundant directional
confirmation (which the POC test just showed hurts more than helps) --
it adds a genuinely different dimension: how stretched the move
already is, not whether a trend exists.

Same validated delta+ADX signal, same $400/$750 exit, same dedup state
machine -- only the RSI filter is new, for a clean comparison.

USAGE:
    python btc_delta_adx_rsi.py --sl 400 --tp 750
    python btc_delta_adx_rsi.py --rsi-overbought 70 --rsi-oversold 30
"""

import argparse
import numpy as np
import pandas as pd

DEFAULT_PARQUET = r"D:\data\btc\BTCUSDT_1m_ohlcv_oi_delta_cvd_5y.parquet"
WEIGHTS = dict(delta=8, adx=35)
TH = dict(adx_period=14, adx_min=18, buy=58, sell=42, min_gap=6, cooldown=2, pressure_smooth=3)
RSI_PERIOD = 14
SL_USDT = None
TP_USDT = None
POSITION_SIZE_BTC = None
RSI_OVERBOUGHT = None
RSI_OVERSOLD = None


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


def compute_rsi(close: np.ndarray, period=14):
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = wilder_smooth(gain, period) / period
    avg_loss = wilder_smooth(loss, period) / period
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100.0, rsi)
    return rsi


def ema(arr, span):
    out = np.empty(len(arr)); out[0] = arr[0]
    alpha = 2 / (span + 1)
    for i in range(1, len(arr)):
        out[i] = (arr[i] - out[i - 1]) * alpha + out[i - 1]
    return out


def compute_signal(bars: pd.DataFrame):
    H, L, C, V = bars["high"].values, bars["low"].values, bars["close"].values, bars["volume"].values
    TB = bars["taker_buy_base"].values

    diPlus, diMinus, adx = compute_adx(H, L, C, TH["adx_period"])
    rsi = compute_rsi(C, RSI_PERIOD)

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

    buy_raw = buy_raw & (rsi < RSI_OVERBOUGHT)
    sell_raw = sell_raw & (rsi > RSI_OVERSOLD)

    return dict(buy_raw=buy_raw, sell_raw=sell_raw, close=C)


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


def run_backtest(bars: pd.DataFrame):
    feats = compute_signal(bars)
    state = BotState()
    trade_log = []
    open_pos = None
    warmup = max(TH["adx_period"], RSI_PERIOD) + 5

    for i in range(warmup, len(bars)):
        close_price = feats["close"][i]
        ts_now = bars["timestamp"].iloc[i]

        if open_pos is not None:
            closed, pnl, reason, exit_price = False, 0.0, "", close_price
            if open_pos["side"] == "BUY":
                if bars["low"].iloc[i] <= open_pos["sl"]:
                    closed, reason, exit_price = True, "Stop Loss", open_pos["sl"]
                    pnl = (open_pos["sl"] - open_pos["entry_price"]) * open_pos["qty"]
                elif bars["high"].iloc[i] >= open_pos["tp"]:
                    closed, reason, exit_price = True, "Take Profit", open_pos["tp"]
                    pnl = (open_pos["tp"] - open_pos["entry_price"]) * open_pos["qty"]
            else:
                if bars["high"].iloc[i] >= open_pos["sl"]:
                    closed, reason, exit_price = True, "Stop Loss", open_pos["sl"]
                    pnl = (open_pos["entry_price"] - open_pos["sl"]) * open_pos["qty"]
                elif bars["low"].iloc[i] <= open_pos["tp"]:
                    closed, reason, exit_price = True, "Take Profit", open_pos["tp"]
                    pnl = (open_pos["entry_price"] - open_pos["tp"]) * open_pos["qty"]
            if closed:
                trade_log.append(dict(entry_time=open_pos["entry_time"], exit_time=ts_now, side=open_pos["side"],
                                       entry=open_pos["entry_price"], exit=exit_price, reason=reason, pnl=pnl))
                open_pos = None

        base_type = "BUY" if feats["buy_raw"][i] else ("SELL" if feats["sell_raw"][i] else "OTHER")
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
    print("BACKTEST REPORT -- BTC delta+ADX + RSI Exhaustion Filter")
    print("=" * 70)
    print(f"Period: {bars['timestamp'].iloc[0]} to {bars['timestamp'].iloc[-1]}")
    print(f"Candles: {len(bars)} (15m) | Stop: ${SL_USDT} | Target: ${TP_USDT} | Size: {POSITION_SIZE_BTC} BTC")
    print(f"RSI filter: block BUY if RSI>{RSI_OVERBOUGHT}, block SELL if RSI<{RSI_OVERSOLD}")
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

    trades_sorted.to_csv("delta_adx_rsi_trades.csv", index=False)
    print(f"\nFull trade log written to delta_adx_rsi_trades.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--sl", type=float, default=400.0)
    parser.add_argument("--tp", type=float, default=750.0)
    parser.add_argument("--size", type=float, default=1.0)
    parser.add_argument("--rsi-overbought", type=float, default=70.0)
    parser.add_argument("--rsi-oversold", type=float, default=30.0)
    args = parser.parse_args()

    SL_USDT = args.sl
    TP_USDT = args.tp
    POSITION_SIZE_BTC = args.size
    RSI_OVERBOUGHT = args.rsi_overbought
    RSI_OVERSOLD = args.rsi_oversold

    bars = load_and_resample_15m(args.parquet, args.days)
    trades = run_backtest(bars)
    print_report(trades, bars)

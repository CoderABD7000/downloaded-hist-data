"""
btc_delta_adx_vwap_v2.py

Thorough test of 5 VWAP-based variants against the validated delta+ADX
baseline, not just one -- since our last 3 single-shot filter attempts
(POC breakout, RSI exhaustion, triple-POC confluence) all failed for
the same underlying reason: they re-confirmed DIRECTION, which ADX
already knows, and ended up cutting some of the best trades (strong
trends naturally trigger every directional confirmation at once).

Variants tested:
  0. BASELINE -- no VWAP filter at all
  1. SIDE -- close > VWAP for BUY, close < VWAP for SELL (the naive,
     likely-redundant version, included for a fair comparison)
  2. DISTANCE_NEAR -- only trade when price is CLOSE to VWAP (< 0.3%
     away), avoiding chasing an already-extended move
  3. DISTANCE_FAR -- the OPPOSITE hypothesis: only trade when price
     has ALREADY moved meaningfully away from VWAP (> 0.3%), i.e.
     require the intraday move to already be underway with real
     conviction before entering
  4. SLOPE -- only trade when VWAP ITSELF is sloping in the trade's
     direction (VWAP now > VWAP 4 bars ago for BUY), i.e. the day's
     aggregate flow is moving the right way, not just current price
     position

Uses SL=400/TP=700 (3rd-best in the grid, best drawdown among leaders).

USAGE:
    python btc_delta_adx_vwap_v2.py --sl 400 --tp 700
"""

import argparse
import numpy as np
import pandas as pd

DEFAULT_PARQUET = r"D:\data\btc\BTCUSDT_1m_ohlcv_oi_delta_cvd_5y.parquet"
WEIGHTS = dict(delta=8, adx=35)
TH = dict(adx_period=14, adx_min=18, buy=58, sell=42, min_gap=6, cooldown=2, pressure_smooth=3)
DISTANCE_THRESHOLD_PCT = 0.3
SLOPE_LOOKBACK = 4
SL_USDT = None
TP_USDT = None
POSITION_SIZE_BTC = None


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


def compute_daily_vwap(bars: pd.DataFrame) -> np.ndarray:
    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
    day = bars["timestamp"].dt.floor("D")
    pv = typical_price * bars["volume"]
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = bars["volume"].groupby(day).cumsum()
    return (cum_pv / cum_vol).values


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


def compute_base_signal(bars: pd.DataFrame):
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


def apply_variant(buy_raw, sell_raw, close, vwap, variant: str):
    buy_raw = buy_raw.copy()
    sell_raw = sell_raw.copy()
    dist_pct = (close - vwap) / vwap * 100

    if variant == "BASELINE":
        pass
    elif variant == "SIDE":
        buy_raw &= (close > vwap)
        sell_raw &= (close < vwap)
    elif variant == "DISTANCE_NEAR":
        near = np.abs(dist_pct) < DISTANCE_THRESHOLD_PCT
        buy_raw &= near
        sell_raw &= near
    elif variant == "DISTANCE_FAR":
        far_up = dist_pct > DISTANCE_THRESHOLD_PCT
        far_down = dist_pct < -DISTANCE_THRESHOLD_PCT
        buy_raw &= far_up
        sell_raw &= far_down
    elif variant == "SLOPE":
        vwap_slope_up = np.zeros(len(vwap), dtype=bool)
        vwap_slope_down = np.zeros(len(vwap), dtype=bool)
        vwap_slope_up[SLOPE_LOOKBACK:] = vwap[SLOPE_LOOKBACK:] > vwap[:-SLOPE_LOOKBACK]
        vwap_slope_down[SLOPE_LOOKBACK:] = vwap[SLOPE_LOOKBACK:] < vwap[:-SLOPE_LOOKBACK]
        buy_raw &= vwap_slope_up
        sell_raw &= vwap_slope_down

    return buy_raw, sell_raw


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


def run_backtest(bars, buy_raw, sell_raw, close, sl_usdt, tp_usdt):
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
                    closed = True; pnl = (open_pos["sl"] - open_pos["entry_price"]) * open_pos["qty"]
                elif high[i] >= open_pos["tp"]:
                    closed = True; pnl = (open_pos["tp"] - open_pos["entry_price"]) * open_pos["qty"]
            else:
                if high[i] >= open_pos["sl"]:
                    closed = True; pnl = (open_pos["entry_price"] - open_pos["sl"]) * open_pos["qty"]
                elif low[i] <= open_pos["tp"]:
                    closed = True; pnl = (open_pos["entry_price"] - open_pos["tp"]) * open_pos["qty"]
            if closed:
                trade_log.append((timestamps[i], pnl))
                open_pos = None

        base_type = "BUY" if buy_raw[i] else ("SELL" if sell_raw[i] else "OTHER")
        dot = state.apply_final_type_rule(base_type)
        signal_dir = "BUY" if dot == "G" else ("SELL" if dot == "R" else None)

        if signal_dir and open_pos is None:
            sl_price = close_price - sl_usdt if signal_dir == "BUY" else close_price + sl_usdt
            tp_price = close_price + tp_usdt if signal_dir == "BUY" else close_price - tp_usdt
            open_pos = dict(side=signal_dir, qty=POSITION_SIZE_BTC, entry_price=close_price, sl=sl_price, tp=tp_price)

    return pd.DataFrame(trade_log, columns=["exit_time", "pnl"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--sl", type=float, default=400.0)
    parser.add_argument("--tp", type=float, default=700.0)
    parser.add_argument("--size", type=float, default=1.0)
    args = parser.parse_args()

    global POSITION_SIZE_BTC
    POSITION_SIZE_BTC = args.size

    bars = load_and_resample_15m(args.parquet, args.days)
    buy_raw_base, sell_raw_base, close = compute_base_signal(bars)
    vwap = compute_daily_vwap(bars)

    variants = ["BASELINE", "SIDE", "DISTANCE_NEAR", "DISTANCE_FAR", "SLOPE"]
    results = []

    print("\n" + "=" * 90)
    print(f"TESTING 5 VWAP VARIANTS at SL=${args.sl} TP=${args.tp}")
    print("=" * 90)

    for variant in variants:
        buy_raw, sell_raw = apply_variant(buy_raw_base, sell_raw_base, close, vwap, variant)
        trades = run_backtest(bars, buy_raw, sell_raw, close, args.sl, args.tp)
        if trades.empty:
            print(f"{variant:<16}: no trades")
            continue
        wins = trades[trades["pnl"] > 0]
        wr = len(wins) / len(trades) * 100
        pnl = trades["pnl"].sum()
        cum = trades["pnl"].cumsum()
        dd = (cum - cum.cummax()).min()
        results.append(dict(variant=variant, trades=len(trades), win_rate=wr, pnl=pnl, dd=dd))
        print(f"{variant:<16}: {len(trades):>6} trades | {wr:>5.1f}% WR | ${pnl:>12,.0f} PnL | ${dd:>10,.0f} maxDD")

    print("\n" + "=" * 90)
    print("SUMMARY vs BASELINE")
    print("=" * 90)
    base = next((r for r in results if r["variant"] == "BASELINE"), None)
    if base:
        for r in results:
            if r["variant"] == "BASELINE":
                continue
            print(f"{r['variant']:<16}: WR {r['win_rate']-base['win_rate']:+.1f}pts | "
                  f"Trades {r['trades']-base['trades']:+d} ({(r['trades']/base['trades']-1)*100:+.1f}%) | "
                  f"PnL ${r['pnl']-base['pnl']:+,.0f}")

    pd.DataFrame(results).to_csv("vwap_variants_results.csv", index=False)
    print("\nFull results written to vwap_variants_results.csv")


if __name__ == "__main__":
    main()

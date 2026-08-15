"""
find_and_test_real_regimes.py

Scans the real 5-year BTC history for the genuinely cleanest real
90-day uptrend, downtrend, and sideways/choppy stretches -- then runs
the already-validated BTC_BOT_V3 backtest engine against each real
window.

"Cleanest" trend is scored by path-directness: net return divided by
the sum of all the absolute 15m-candle returns along the way, PLUS a
check on total price range -- this correctly distinguishes genuine
tight-range chop from a "V-shaped" trend-then-reversal that nets flat
but still traveled a long distance.

USAGE:
    python find_and_test_real_regimes.py
"""

import sys
import pandas as pd
import numpy as np

sys.path.insert(0, ".")
from backtest_btc_bot_v3 import run_backtest

PARQUET_PATH = r"D:\data\btc\BTCUSDT_1m_ohlcv_oi_delta_cvd_5y.parquet"
WINDOW_DAYS = 90
CANDLES_PER_WINDOW = WINDOW_DAYS * 24 * 4


def load_and_resample_15m(parquet_path: str) -> pd.DataFrame:
    print(f"Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    agg = df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "taker_buy_vol": "sum",
    }).dropna()
    agg = agg.rename(columns={"taker_buy_vol": "taker_buy_base"})
    return agg.reset_index()


def score_windows(bars: pd.DataFrame, window_size: int, step: int):
    closes = bars["close"].values
    n = len(closes)
    results = []
    for start in range(0, n - window_size, step):
        end = start + window_size
        window_closes = closes[start:end]
        net_return = (window_closes[-1] - window_closes[0]) / window_closes[0]
        step_returns = np.diff(window_closes) / window_closes[:-1]
        total_abs_movement = np.sum(np.abs(step_returns))
        directness = net_return / total_abs_movement if total_abs_movement > 0 else 0
        price_range = (window_closes.max() - window_closes.min()) / window_closes[0]
        results.append((start, net_return, directness, price_range))
    return results


def main():
    bars = load_and_resample_15m(PARQUET_PATH)
    print(f"Resampled to {len(bars):,} x 15m candles, {bars['timestamp'].iloc[0]} to {bars['timestamp'].iloc[-1]}")

    step = CANDLES_PER_WINDOW // 4
    print(f"Scanning {WINDOW_DAYS}-day windows (step={step} candles) for trend cleanliness...")
    scored = score_windows(bars, CANDLES_PER_WINDOW, step)
    print(f"Scored {len(scored)} windows\n")

    best_up = max(scored, key=lambda x: x[1] * max(x[2], 0))
    best_down = min(scored, key=lambda x: x[1] * max(-x[2], 0) if x[2] < 0 else 0)
    best_side = min(scored, key=lambda x: abs(x[1]) + abs(x[2]) + x[3])

    windows = {"uptrend": best_up, "downtrend": best_down, "sideways": best_side}

    print("=" * 100)
    print("SELECTED REAL HISTORICAL WINDOWS")
    print("=" * 100)
    real_bars = {}
    for regime, (start, net_ret, directness, price_range) in windows.items():
        end = start + CANDLES_PER_WINDOW
        window_bars = bars.iloc[start:end].reset_index(drop=True)
        real_bars[regime] = window_bars
        print(f"{regime:<10} {window_bars['timestamp'].iloc[0]} to {window_bars['timestamp'].iloc[-1]} "
              f"| net return: {net_ret*100:+.1f}% | directness: {directness:+.2f} | range: {price_range*100:.1f}%")

    print("\n" + "=" * 100)
    print("BACKTEST RESULTS ON REAL HISTORICAL REGIME WINDOWS")
    print("=" * 100)
    print(f"{'Regime':<12} {'Trades':>8} {'Win %':>8} {'Total PnL':>14} {'Max DD':>12} {'BUY':>8} {'SELL':>8}")
    print("-" * 90)

    for regime, window_bars in real_bars.items():
        trades, state = run_backtest(window_bars)
        if trades.empty:
            print(f"{regime:<12} {'0 trades':>8}")
            continue
        wins = trades[trades["pnl"] > 0]
        win_rate = len(wins) / len(trades) * 100
        total_pnl = trades["pnl"].sum()
        trades_sorted = trades.sort_values("exit_time").reset_index(drop=True)
        cum = trades_sorted["pnl"].cumsum()
        max_dd = (cum - cum.cummax()).min()
        n_buy = (trades["side"] == "BUY").sum()
        n_sell = (trades["side"] == "SELL").sum()
        print(f"{regime:<12} {len(trades):>8} {win_rate:>7.1f}% ${total_pnl:>12,.2f} ${max_dd:>10,.2f} {n_buy:>8} {n_sell:>8}")
        trades.to_csv(f"real_regime_{regime}_trades.csv", index=False)

    print("\nPer-regime trade logs written to real_regime_<name>_trades.csv")


if __name__ == "__main__":
    main()

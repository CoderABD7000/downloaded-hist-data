"""
yearly_breakdown.py

Runs the full BTC_BOT_V3 backtest across the entire available history
and breaks the results down year by year -- trade count, win rate,
total PnL, and max drawdown within each calendar year.

Requires backtest_btc_bot_v3.py in the same folder.

USAGE:
    python yearly_breakdown.py
"""

import sys
import pandas as pd

sys.path.insert(0, ".")
from backtest_btc_bot_v3 import load_and_resample_15m, run_backtest

PARQUET_PATH = r"D:\data\btc\BTCUSDT_1m_ohlcv_oi_delta_cvd_5y.parquet"


def main():
    bars = load_and_resample_15m(PARQUET_PATH, days=None)
    print(f"\nRunning full backtest across {len(bars):,} candles...")
    trades, state = run_backtest(bars)

    if trades.empty:
        print("No trades generated.")
        return

    trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    trades["year"] = trades["exit_time"].dt.year

    print("\n" + "=" * 90)
    print("YEAR-BY-YEAR PERFORMANCE")
    print("=" * 90)
    print(f"{'Year':<8} {'Trades':>8} {'Win %':>8} {'Total PnL':>14} {'Max DD (within year)':>22} {'Avg PnL/trade':>16}")
    print("-" * 90)

    for year in sorted(trades["year"].unique()):
        yr = trades[trades["year"] == year].sort_values("exit_time").reset_index(drop=True)
        wins = yr[yr["pnl"] > 0]
        win_rate = len(wins) / len(yr) * 100
        total_pnl = yr["pnl"].sum()
        avg_pnl = yr["pnl"].mean()
        cum = yr["pnl"].cumsum()
        max_dd = (cum - cum.cummax()).min()
        print(f"{year:<8} {len(yr):>8} {win_rate:>7.1f}% ${total_pnl:>12,.2f} ${max_dd:>20,.2f} ${avg_pnl:>14,.2f}")

    print("-" * 90)
    total_wins = trades[trades["pnl"] > 0]
    print(f"{'ALL':<8} {len(trades):>8} {len(total_wins)/len(trades)*100:>7.1f}% ${trades['pnl'].sum():>12,.2f}")

    trades.to_csv("full_trades_by_year.csv", index=False)
    print("\nFull trade log (with year column) written to full_trades_by_year.csv")


if __name__ == "__main__":
    main()

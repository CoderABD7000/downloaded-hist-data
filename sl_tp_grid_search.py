"""
sl_tp_grid_search.py

For GHULAM-OFBOT's 110 REAL trade entries, tests a grid of SL/TP
combinations to find which one is hit first (walking forward through
1m candles using High/Low, same verified logic as the excursion
analysis). Reports win rate AND total PnL for every combination --
maximizing win rate alone is NOT the same as maximizing profitability
(a tiny target + huge stop inflates win rate while often losing money
overall), so both are shown together.

USAGE:
    python sl_tp_grid_search.py
"""

import pandas as pd
import numpy as np

IN_PATH = r"D:\data\btc\BTCUSDT_1m_with_GHULAM_trades.xlsx"
OUT_PATH = r"D:\data\btc\GHULAM_sl_tp_grid_search.xlsx"

SL_GRID = [100, 150, 200, 250, 300, 400, 500, 600]
TP_GRID = [100, 150, 200, 250, 300, 400, 500, 600]
MAX_LOOKFORWARD_MIN = 2880

TRADES = [
    ("2026-08-08 14:13:00", "LONG", 65033.55),
    ("2026-08-08 21:54:00", "LONG", 65027.55),
    ("2026-08-09 06:10:00", "LONG", 64789.91),
    ("2026-08-09 09:03:00", "SHORT", 64787.59),
    ("2026-08-09 15:42:00", "LONG", 65020.5),
    ("2026-08-10 04:14:00", "SHORT", 65002.35),
    ("2026-08-10 04:40:00", "LONG", 65051.26),
    ("2026-08-10 07:12:00", "LONG", 65270.2),
    ("2026-08-10 13:48:00", "LONG", 64659.18),
    ("2026-08-10 15:23:00", "LONG", 64500.45),
    ("2026-08-10 18:19:00", "SHORT", 64096.18),
    ("2026-08-10 20:36:00", "LONG", 64092.77),
    ("2026-08-10 22:44:00", "SHORT", 63994.2),
    ("2026-08-11 01:06:00", "LONG", 63992.25),
    ("2026-08-11 15:44:00", "LONG", 63571.86),
    ("2026-08-12 13:03:00", "LONG", 64102.77),
    ("2026-08-12 14:51:00", "SHORT", 63505.05),
    ("2026-08-13 11:38:00", "SHORT", 63343.98),
    ("2026-08-13 13:56:00", "LONG", 63753.4),
    ("2026-08-13 17:12:00", "SHORT", 63096.33),
    ("2026-08-14 11:55:00", "SHORT", 62883.67),
    ("2026-08-14 13:54:00", "LONG", 62652.78),
    ("2026-08-15 00:21:00", "SHORT", 62973.65),
    ("2026-08-15 00:30:00", "LONG", 63017.95),
    ("2026-08-15 00:47:00", "SHORT", 62985.95),
    ("2026-08-15 00:52:00", "LONG", 63013.15),
    ("2026-08-15 01:25:00", "SHORT", 62972.55),
    ("2026-08-15 05:47:00", "LONG", 63047.31),
    ("2026-08-16 17:51:00", "LONG", 63140.18),
    ("2026-08-17 17:36:00", "LONG", 64128.87),
    ("2026-08-18 14:40:00", "LONG", 64947.94),
    ("2026-08-19 01:01:00", "LONG", 64468.84),
    ("2026-08-19 14:00:00", "LONG", 65007.35),
    ("2026-08-19 14:10:00", "LONG", 65361.42),
    ("2026-08-19 16:38:00", "LONG", 68817.81),
    ("2026-08-19 17:35:00", "LONG", 68023.25),
    ("2026-08-19 18:18:00", "SHORT", 68164.71),
    ("2026-08-19 18:37:00", "LONG", 68287.1),
    ("2026-08-19 19:08:00", "SHORT", 68352.28),
    ("2026-08-19 19:34:00", "LONG", 68313.61),
    ("2026-08-19 20:04:00", "SHORT", 68373.87),
    ("2026-08-19 21:44:00", "LONG", 69595.87),
    ("2026-08-19 22:17:00", "LONG", 69334.71),
    ("2026-08-19 23:00:00", "SHORT", 69208.11),
    ("2026-08-19 23:35:00", "LONG", 69320.01),
    ("2026-08-20 01:00:00", "LONG", 69399.43),
    ("2026-08-20 01:44:00", "SHORT", 69555.64),
    ("2026-08-20 03:00:00", "LONG", 69309.31),
    ("2026-08-20 04:30:00", "SHORT", 69277.69),
    ("2026-08-20 04:40:00", "LONG", 69346.82),
    ("2026-08-20 08:13:00", "LONG", 71058.16),
    ("2026-08-20 08:35:00", "SHORT", 71150.22),
    ("2026-08-20 09:04:00", "LONG", 71697.49),
    ("2026-08-20 09:34:00", "SHORT", 71865.07),
    ("2026-08-20 09:37:00", "LONG", 71988.34),
    ("2026-08-20 10:39:00", "LONG", 71890.53),
    ("2026-08-20 10:52:00", "SHORT", 71810.68),
    ("2026-08-20 11:13:00", "LONG", 71924.43),
    ("2026-08-20 11:43:00", "SHORT", 72147.02),
    ("2026-08-20 12:01:00", "LONG", 71930.83),
    ("2026-08-20 12:38:00", "SHORT", 71714.0),
    ("2026-08-20 13:18:00", "SHORT", 71925.56),
    ("2026-08-20 20:13:00", "LONG", 72739.8),
    ("2026-08-21 00:12:00", "SHORT", 73227.0),
    ("2026-08-21 00:45:00", "LONG", 73749.0),
    ("2026-08-21 00:54:00", "LONG", 73750.0),
    ("2026-08-21 01:19:00", "LONG", 74614.97),
    ("2026-08-21 01:27:00", "SHORT", 74488.15),
    ("2026-08-21 01:35:00", "SHORT", 74703.01),
    ("2026-08-21 02:24:00", "LONG", 74799.01),
    ("2026-08-21 03:09:00", "LONG", 74347.92),
    ("2026-08-21 04:08:00", "SHORT", 74703.21),
    ("2026-08-21 05:38:00", "LONG", 75289.5),
    ("2026-08-21 05:48:00", "SHORT", 75125.12),
    ("2026-08-21 06:18:00", "LONG", 75420.33),
    ("2026-08-21 06:31:00", "SHORT", 75374.77),
    ("2026-08-21 07:03:00", "LONG", 75808.81),
    ("2026-08-21 07:50:00", "SHORT", 76332.68),
    ("2026-08-21 07:58:00", "LONG", 76392.63),
    ("2026-08-21 08:17:00", "LONG", 76995.35),
    ("2026-08-21 08:29:00", "LONG", 77105.37),
    ("2026-08-21 09:04:00", "LONG", 77923.53),
    ("2026-08-21 09:29:00", "SHORT", 77707.81),
    ("2026-08-21 10:51:00", "LONG", 77786.2),
    ("2026-08-21 11:29:00", "LONG", 77445.34),
    ("2026-08-21 12:23:00", "LONG", 76879.02),
    ("2026-08-21 12:47:00", "LONG", 77355.87),
    ("2026-08-21 14:55:00", "SHORT", 77231.4),
    ("2026-08-21 16:18:00", "LONG", 77186.18),
    ("2026-08-21 16:34:00", "SHORT", 77273.99),
    ("2026-08-21 18:02:00", "LONG", 77454.24),
    ("2026-08-21 19:12:00", "SHORT", 76989.25),
    ("2026-08-21 20:46:00", "LONG", 77447.34),
    ("2026-08-21 20:51:00", "SHORT", 77397.27),
    ("2026-08-21 21:44:00", "LONG", 78216.79),
    ("2026-08-21 22:09:00", "SHORT", 78558.64),
    ("2026-08-21 22:54:00", "SHORT", 78505.25),
    ("2026-08-21 23:14:00", "LONG", 78539.95),
    ("2026-08-21 23:48:00", "LONG", 78292.91),
    ("2026-08-22 00:24:00", "SHORT", 78009.44),
    ("2026-08-22 01:55:00", "SHORT", 77799.59),
    ("2026-08-22 02:47:00", "LONG", 78379.32),
    ("2026-08-22 03:27:00", "LONG", 78736.19),
    ("2026-08-22 05:21:00", "LONG", 77068.36),
    ("2026-08-22 05:38:00", "LONG", 77251.8),
    ("2026-08-22 06:16:00", "LONG", 77252.4),
    ("2026-08-22 08:17:00", "SHORT", 77290.19),
    ("2026-08-22 09:08:00", "LONG", 77142.98),
    ("2026-08-22 10:02:00", "LONG", 76999.35),
    ("2026-08-22 11:15:00", "LONG", 76979.64),
]


def find_outcome(bars_high, bars_low, entry_idx, side, entry_price, sl, tp, n):
    end_idx = min(entry_idx + MAX_LOOKFORWARD_MIN, n)
    for i in range(entry_idx, end_idx):
        h, l = bars_high[i], bars_low[i]
        if side == "LONG":
            sl_price = entry_price - sl
            tp_price = entry_price + tp
            sl_hit = l <= sl_price
            tp_hit = h >= tp_price
        else:
            sl_price = entry_price + sl
            tp_price = entry_price - tp
            sl_hit = h >= sl_price
            tp_hit = l <= tp_price

        if sl_hit:
            return "LOSS"
        if tp_hit:
            return "WIN"
    return "NEITHER"


def main():
    print(f"Loading {IN_PATH} ...")
    df = pd.read_excel(IN_PATH)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    time_to_idx = {t: i for i, t in enumerate(df["Time"])}
    highs = df["High"].values
    lows = df["Low"].values

    trade_indices = []
    for entry_time_str, side, entry_price in TRADES:
        entry_time = pd.Timestamp(entry_time_str).round("min")
        if entry_time in time_to_idx:
            trade_indices.append((time_to_idx[entry_time], side, entry_price))
        else:
            print(f"  SKIP: {entry_time} not found")

    print(f"{len(trade_indices)} trades loaded. Running {len(SL_GRID)}x{len(TP_GRID)} grid search...")

    results = []
    for sl in SL_GRID:
        for tp in TP_GRID:
            wins = 0
            losses = 0
            neither = 0
            total_pnl = 0.0
            for idx, side, entry_price in trade_indices:
                outcome = find_outcome(highs, lows, idx, side, entry_price, sl, tp, n)
                if outcome == "WIN":
                    wins += 1
                    total_pnl += tp
                elif outcome == "LOSS":
                    losses += 1
                    total_pnl -= sl
                else:
                    neither += 1
            resolved = wins + losses
            win_rate = wins / resolved * 100 if resolved > 0 else 0
            rr = tp / sl
            breakeven = 1 / (1 + rr) * 100
            results.append(dict(sl=sl, tp=tp, rr=round(rr, 2), wins=wins, losses=losses,
                                 neither=neither, resolved=resolved, win_rate=round(win_rate, 1),
                                 breakeven_wr=round(breakeven, 1), edge=round(win_rate - breakeven, 1),
                                 total_pnl=round(total_pnl, 1)))

    res_df = pd.DataFrame(results)

    print()
    print("=" * 110)
    print("SORTED BY WIN RATE (best first) -- CAUTION: high win rate alone does not mean profitable, check PnL column")
    print("=" * 110)
    by_wr = res_df.sort_values("win_rate", ascending=False).head(15)
    header = "{:>6} {:>6} {:>6} {:>7} {:>11} {:>7} {:>6} {:>7} {:>10}".format(
        "SL", "TP", "R:R", "Win%", "Breakeven%", "Edge", "Wins", "Losses", "TotalPnL")
    print(header)
    for _, r in by_wr.iterrows():
        line = "{:>6.0f} {:>6.0f} {:>6.2f} {:>6.1f}% {:>10.1f}% {:>+6.1f} {:>6.0f} {:>7.0f} {:>+10.1f}".format(
            r['sl'], r['tp'], r['rr'], r['win_rate'], r['breakeven_wr'], r['edge'],
            r['wins'], r['losses'], r['total_pnl'])
        print(line)

    print()
    print("=" * 110)
    print("SORTED BY TOTAL PNL (best first) -- the metric that actually matters for profitability")
    print("=" * 110)
    by_pnl = res_df.sort_values("total_pnl", ascending=False).head(15)
    print(header)
    for _, r in by_pnl.iterrows():
        line = "{:>6.0f} {:>6.0f} {:>6.2f} {:>6.1f}% {:>10.1f}% {:>+6.1f} {:>6.0f} {:>7.0f} {:>+10.1f}".format(
            r['sl'], r['tp'], r['rr'], r['win_rate'], r['breakeven_wr'], r['edge'],
            r['wins'], r['losses'], r['total_pnl'])
        print(line)

    res_df.to_excel(OUT_PATH, index=False)
    print(f"\nFull {len(res_df)}-combination grid written to {OUT_PATH}")


if __name__ == "__main__":
    main()

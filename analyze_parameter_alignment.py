"""
analyze_parameter_alignment.py

For each GHULAM-OFBOT trade, finds the nearest COMPLETED 5-minute
CMA-Signal candle at or before the trade's entry time (no lookahead --
only information genuinely available before the trade was taken), and
checks whether each parameter (Trend, Momentum, Volatility, VWAP,
Volume, Structure) was supporting or contradicting the trade's actual
direction.

Cross-tabulates this against the 3 excursion categories already found:
  STRAIGHT_AGAINST -- went straight against us to -500 (26 trades)
  CLEAN_FAVORABLE  -- moved cleanly in our favor (6 trades)
  MIXED            -- everything else (78 trades)

USAGE:
    python analyze_parameter_alignment.py
"""

import pandas as pd
import numpy as np

CMA_PATH = r"D:\data\btc\BTCUSDT_5m_CMA_signals.parquet"
OUT_PATH = r"D:\data\btc\GHULAM_parameter_alignment.xlsx"

MIN_TREND_SCORE = 40.0
MIN_MOM_SCORE = 40.0
SUPPORT_THRESHOLD = 50.0

TRADES = [
    ("2026-08-08 14:13:00", "LONG", 65033.55, "MIXED"),
    ("2026-08-08 21:54:00", "LONG", 65027.55, "MIXED"),
    ("2026-08-09 06:10:00", "LONG", 64789.91, "MIXED"),
    ("2026-08-09 09:03:00", "SHORT", 64787.59, "STRAIGHT_AGAINST"),
    ("2026-08-09 15:42:00", "LONG", 65020.5, "MIXED"),
    ("2026-08-10 04:14:00", "SHORT", 65002.35, "MIXED"),
    ("2026-08-10 04:40:00", "LONG", 65051.26, "MIXED"),
    ("2026-08-10 07:12:00", "LONG", 65270.2, "MIXED"),
    ("2026-08-10 13:48:00", "LONG", 64659.18, "MIXED"),
    ("2026-08-10 15:23:00", "LONG", 64500.45, "STRAIGHT_AGAINST"),
    ("2026-08-10 18:19:00", "SHORT", 64096.18, "MIXED"),
    ("2026-08-10 20:36:00", "LONG", 64092.77, "MIXED"),
    ("2026-08-10 22:44:00", "SHORT", 63994.2, "MIXED"),
    ("2026-08-11 01:06:00", "LONG", 63992.25, "MIXED"),
    ("2026-08-11 15:44:00", "LONG", 63571.86, "MIXED"),
    ("2026-08-12 13:03:00", "LONG", 64102.77, "STRAIGHT_AGAINST"),
    ("2026-08-12 14:51:00", "SHORT", 63505.05, "MIXED"),
    ("2026-08-13 11:38:00", "SHORT", 63343.98, "STRAIGHT_AGAINST"),
    ("2026-08-13 13:56:00", "LONG", 63753.4, "MIXED"),
    ("2026-08-13 17:12:00", "SHORT", 63096.33, "STRAIGHT_AGAINST"),
    ("2026-08-14 11:55:00", "SHORT", 62883.67, "MIXED"),
    ("2026-08-14 13:54:00", "LONG", 62652.78, "MIXED"),
    ("2026-08-15 00:21:00", "SHORT", 62973.65, "MIXED"),
    ("2026-08-15 00:30:00", "LONG", 63017.95, "MIXED"),
    ("2026-08-15 00:47:00", "SHORT", 62985.95, "MIXED"),
    ("2026-08-15 00:52:00", "LONG", 63013.15, "MIXED"),
    ("2026-08-15 01:25:00", "SHORT", 62972.55, "MIXED"),
    ("2026-08-15 05:47:00", "LONG", 63047.31, "MIXED"),
    ("2026-08-16 17:51:00", "LONG", 63140.18, "MIXED"),
    ("2026-08-17 17:36:00", "LONG", 64128.87, "MIXED"),
    ("2026-08-18 14:40:00", "LONG", 64947.94, "STRAIGHT_AGAINST"),
    ("2026-08-19 01:01:00", "LONG", 64468.84, "MIXED"),
    ("2026-08-19 14:00:00", "LONG", 65007.35, "CLEAN_FAVORABLE"),
    ("2026-08-19 14:10:00", "LONG", 65361.42, "CLEAN_FAVORABLE"),
    ("2026-08-19 16:38:00", "LONG", 68817.81, "STRAIGHT_AGAINST"),
    ("2026-08-19 17:35:00", "LONG", 68023.25, "CLEAN_FAVORABLE"),
    ("2026-08-19 18:18:00", "SHORT", 68164.71, "STRAIGHT_AGAINST"),
    ("2026-08-19 18:37:00", "LONG", 68287.1, "MIXED"),
    ("2026-08-19 19:08:00", "SHORT", 68352.28, "MIXED"),
    ("2026-08-19 19:34:00", "LONG", 68313.61, "MIXED"),
    ("2026-08-19 20:04:00", "SHORT", 68373.87, "STRAIGHT_AGAINST"),
    ("2026-08-19 21:44:00", "LONG", 69595.87, "MIXED"),
    ("2026-08-19 22:17:00", "LONG", 69334.71, "MIXED"),
    ("2026-08-19 23:00:00", "SHORT", 69208.11, "STRAIGHT_AGAINST"),
    ("2026-08-19 23:35:00", "LONG", 69320.01, "MIXED"),
    ("2026-08-20 01:00:00", "LONG", 69399.43, "MIXED"),
    ("2026-08-20 01:44:00", "SHORT", 69555.64, "MIXED"),
    ("2026-08-20 03:00:00", "LONG", 69309.31, "MIXED"),
    ("2026-08-20 04:30:00", "SHORT", 69277.69, "STRAIGHT_AGAINST"),
    ("2026-08-20 04:40:00", "LONG", 69346.82, "CLEAN_FAVORABLE"),
    ("2026-08-20 08:13:00", "LONG", 71058.16, "CLEAN_FAVORABLE"),
    ("2026-08-20 08:35:00", "SHORT", 71150.22, "STRAIGHT_AGAINST"),
    ("2026-08-20 09:04:00", "LONG", 71697.49, "MIXED"),
    ("2026-08-20 09:34:00", "SHORT", 71865.07, "MIXED"),
    ("2026-08-20 09:37:00", "LONG", 71988.34, "MIXED"),
    ("2026-08-20 10:39:00", "LONG", 71890.53, "MIXED"),
    ("2026-08-20 10:52:00", "SHORT", 71810.68, "STRAIGHT_AGAINST"),
    ("2026-08-20 11:13:00", "LONG", 71924.43, "MIXED"),
    ("2026-08-20 11:43:00", "SHORT", 72147.02, "MIXED"),
    ("2026-08-20 12:01:00", "LONG", 71930.83, "MIXED"),
    ("2026-08-20 12:38:00", "SHORT", 71714.0, "MIXED"),
    ("2026-08-20 13:18:00", "SHORT", 71925.56, "MIXED"),
    ("2026-08-20 20:13:00", "LONG", 72739.8, "MIXED"),
    ("2026-08-21 00:12:00", "SHORT", 73227.0, "STRAIGHT_AGAINST"),
    ("2026-08-21 00:45:00", "LONG", 73749.0, "MIXED"),
    ("2026-08-21 00:54:00", "LONG", 73750.0, "MIXED"),
    ("2026-08-21 01:19:00", "LONG", 74614.97, "MIXED"),
    ("2026-08-21 01:27:00", "SHORT", 74488.15, "STRAIGHT_AGAINST"),
    ("2026-08-21 01:35:00", "SHORT", 74703.01, "MIXED"),
    ("2026-08-21 02:24:00", "LONG", 74799.01, "MIXED"),
    ("2026-08-21 03:09:00", "LONG", 74347.92, "CLEAN_FAVORABLE"),
    ("2026-08-21 04:08:00", "SHORT", 74703.21, "STRAIGHT_AGAINST"),
    ("2026-08-21 05:38:00", "LONG", 75289.5, "MIXED"),
    ("2026-08-21 05:48:00", "SHORT", 75125.12, "STRAIGHT_AGAINST"),
    ("2026-08-21 06:18:00", "LONG", 75420.33, "MIXED"),
    ("2026-08-21 06:31:00", "SHORT", 75374.77, "STRAIGHT_AGAINST"),
    ("2026-08-21 07:03:00", "LONG", 75808.81, "MIXED"),
    ("2026-08-21 07:50:00", "SHORT", 76332.68, "MIXED"),
    ("2026-08-21 07:58:00", "LONG", 76392.63, "MIXED"),
    ("2026-08-21 08:17:00", "LONG", 76995.35, "MIXED"),
    ("2026-08-21 08:29:00", "LONG", 77105.37, "MIXED"),
    ("2026-08-21 09:04:00", "LONG", 77923.53, "MIXED"),
    ("2026-08-21 09:29:00", "SHORT", 77707.81, "STRAIGHT_AGAINST"),
    ("2026-08-21 10:51:00", "LONG", 77786.2, "MIXED"),
    ("2026-08-21 11:29:00", "LONG", 77445.34, "STRAIGHT_AGAINST"),
    ("2026-08-21 12:23:00", "LONG", 76879.02, "MIXED"),
    ("2026-08-21 12:47:00", "LONG", 77355.87, "STRAIGHT_AGAINST"),
    ("2026-08-21 14:55:00", "SHORT", 77231.4, "STRAIGHT_AGAINST"),
    ("2026-08-21 16:18:00", "LONG", 77186.18, "MIXED"),
    ("2026-08-21 16:34:00", "SHORT", 77273.99, "MIXED"),
    ("2026-08-21 18:02:00", "LONG", 77454.24, "STRAIGHT_AGAINST"),
    ("2026-08-21 19:12:00", "SHORT", 76989.25, "MIXED"),
    ("2026-08-21 20:46:00", "LONG", 77447.34, "MIXED"),
    ("2026-08-21 20:51:00", "SHORT", 77397.27, "STRAIGHT_AGAINST"),
    ("2026-08-21 21:44:00", "LONG", 78216.79, "MIXED"),
    ("2026-08-21 22:09:00", "SHORT", 78558.64, "MIXED"),
    ("2026-08-21 22:54:00", "SHORT", 78505.25, "MIXED"),
    ("2026-08-21 23:14:00", "LONG", 78539.95, "MIXED"),
    ("2026-08-21 23:48:00", "LONG", 78292.91, "MIXED"),
    ("2026-08-22 00:24:00", "SHORT", 78009.44, "MIXED"),
    ("2026-08-22 01:55:00", "SHORT", 77799.59, "STRAIGHT_AGAINST"),
    ("2026-08-22 02:47:00", "LONG", 78379.32, "MIXED"),
    ("2026-08-22 03:27:00", "LONG", 78736.19, "STRAIGHT_AGAINST"),
    ("2026-08-22 05:21:00", "LONG", 77068.36, "MIXED"),
    ("2026-08-22 05:38:00", "LONG", 77251.8, "MIXED"),
    ("2026-08-22 06:16:00", "LONG", 77252.4, "MIXED"),
    ("2026-08-22 08:17:00", "SHORT", 77290.19, "MIXED"),
    ("2026-08-22 09:08:00", "LONG", 77142.98, "MIXED"),
    ("2026-08-22 10:02:00", "LONG", 76999.35, "MIXED"),
    ("2026-08-22 11:15:00", "LONG", 76979.64, "MIXED"),
]


def main():
    print(f"Loading {CMA_PATH} ...")
    cma = pd.read_parquet(CMA_PATH)
    cma["timestamp"] = pd.to_datetime(cma["timestamp"], utc=True).dt.tz_localize(None)
    cma = cma.sort_values("timestamp").reset_index(drop=True)
    print(f"{len(cma):,} x 5m CMA candles, {cma['timestamp'].iloc[0]} to {cma['timestamp'].iloc[-1]}")

    trades = pd.DataFrame(TRADES, columns=["entry_time", "side", "entry_price", "category"])
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])

    results = []
    for _, t in trades.iterrows():
        prior = cma[cma["timestamp"] <= t["entry_time"]]
        if len(prior) == 0:
            print(f"  SKIP: no CMA data before {t['entry_time']}")
            continue
        row = prior.iloc[-1]

        side = t["side"]
        if side == "LONG":
            trend_score = row["trend_long_score"]
            mom_score = row["momentum_long_score"]
            vwap_score = row["vwap_long_score"]
            struct_score = row["structure_long_score"]
            total_score = row["long_score"]
            mtf_ok = row["mtf_long_ok"]
        else:
            trend_score = row["trend_short_score"]
            mom_score = row["momentum_short_score"]
            vwap_score = row["vwap_short_score"]
            struct_score = row["structure_short_score"]
            total_score = row["short_score"]
            mtf_ok = row["mtf_short_ok"]

        results.append(dict(
            entry_time=t["entry_time"], side=side, category=t["category"],
            cma_candle_time=row["timestamp"],
            minutes_before_entry=(t["entry_time"] - row["timestamp"]).total_seconds() / 60,
            trend_score=trend_score, trend_supports=trend_score >= MIN_TREND_SCORE,
            momentum_score=mom_score, momentum_supports=mom_score >= MIN_MOM_SCORE,
            volatility_score=row["volatility_score"], vol_regime=row["vol_regime"],
            vwap_score=vwap_score, vwap_supports=vwap_score >= SUPPORT_THRESHOLD,
            volume_score=row["volume_score"], volume_supports=row["volume_score"] >= SUPPORT_THRESHOLD,
            structure_score=struct_score, structure_supports=struct_score >= SUPPORT_THRESHOLD,
            total_score=total_score, mtf_supports=mtf_ok,
        ))

    res = pd.DataFrame(results)
    print(f"\nAnalyzed {len(res)} trades\n")

    support_cols = ["trend_supports", "momentum_supports", "vwap_supports",
                     "volume_supports", "structure_supports", "mtf_supports"]

    print("=" * 90)
    print("SUPPORT RATE (%) BY PARAMETER, SPLIT BY EXCURSION CATEGORY")
    print("=" * 90)
    summary = res.groupby("category")[support_cols].mean() * 100
    summary = summary.round(1)
    print(summary.to_string())
    print()
    print("(Row = trade category, columns = % of trades in that category where this ")
    print(" parameter was SUPPORTING the trade's actual direction at entry)")

    print()
    print("=" * 90)
    print("AVERAGE TOTAL SCORE BY CATEGORY")
    print("=" * 90)
    print(res.groupby("category")["total_score"].agg(["mean", "median", "count"]).round(1).to_string())

    res.to_excel(OUT_PATH, index=False)
    print(f"\nFull per-trade parameter data written to {OUT_PATH}")


if __name__ == "__main__":
    main()

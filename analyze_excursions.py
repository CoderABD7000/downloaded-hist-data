"""
analyze_excursions.py

For each GHULAM-OFBOT trade, walks forward through 1-minute candles
computing running Maximum Favorable Excursion (MFE) and Maximum
Adverse Excursion (MAE) -- using each candle's High/Low (not just
Close) for accurate excursion tracking, since price can move further
within a candle than the close shows.

Answers:
  Q1: For trades that eventually reached -500 adverse, how long
      (minutes) did price move favorably BEFORE that reversal began?
  Q2: How many trades moved "almost straight" in our favor, tolerating
      dips no worse than -100, before reaching real profit?
  Q3: How many trades went straight against us, reaching -500 adverse
      with little/no favorable move first?

USAGE:
    python analyze_excursions.py
"""

import pandas as pd
import numpy as np

IN_PATH = r"D:\data\btc\BTCUSDT_1m_with_GHULAM_trades.xlsx"
OUT_PATH = r"D:\data\btc\GHULAM_excursion_analysis.xlsx"

ADVERSE_THRESHOLD = 500.0
CLEAN_TOLERANCE = 100.0
MAX_LOOKFORWARD_MIN = 2880

TRADES = [
    ("2026-08-08 14:13:01.952", "LONG", 65033.55),
    ("2026-08-08 21:54:08.470", "LONG", 65027.55),
    ("2026-08-09 06:10:06.860", "LONG", 64789.91),
    ("2026-08-09 09:02:50.581", "SHORT", 64787.59),
    ("2026-08-09 15:42:02.130", "LONG", 65020.5),
    ("2026-08-10 04:14:09.527", "SHORT", 65002.35),
    ("2026-08-10 04:39:51.711", "LONG", 65051.26),
    ("2026-08-10 07:11:49.013", "LONG", 65270.2),
    ("2026-08-10 13:47:40.281", "LONG", 64659.18),
    ("2026-08-10 15:23:01.571", "LONG", 64500.45),
    ("2026-08-10 18:19:21.739", "SHORT", 64096.18),
    ("2026-08-10 20:35:49.185", "LONG", 64092.77),
    ("2026-08-10 22:44:27.283", "SHORT", 63994.2),
    ("2026-08-11 01:06:20.312", "LONG", 63992.25),
    ("2026-08-11 15:44:08.864", "LONG", 63571.86),
    ("2026-08-12 13:02:52.505", "LONG", 64102.77),
    ("2026-08-12 14:51:24.153", "SHORT", 63505.05),
    ("2026-08-13 11:37:31.157", "SHORT", 63343.98),
    ("2026-08-13 13:56:16.427", "LONG", 63753.4),
    ("2026-08-13 17:11:56.289", "SHORT", 63096.33),
    ("2026-08-14 11:55:12.463", "SHORT", 62883.67),
    ("2026-08-14 13:54:19.179", "LONG", 62652.78),
    ("2026-08-15 00:21:00.257", "SHORT", 62973.65),
    ("2026-08-15 00:29:39.400", "LONG", 63017.95),
    ("2026-08-15 00:47:20.568", "SHORT", 62985.95),
    ("2026-08-15 00:51:55.456", "LONG", 63013.15),
    ("2026-08-15 01:24:48.904", "SHORT", 62972.55),
    ("2026-08-15 05:47:14.446", "LONG", 63047.31),
    ("2026-08-16 17:51:12.125", "LONG", 63140.18),
    ("2026-08-17 17:36:04.318", "LONG", 64128.87),
    ("2026-08-18 14:40:24.847", "LONG", 64947.94),
    ("2026-08-19 01:00:48.309", "LONG", 64468.84),
    ("2026-08-19 13:59:59.082", "LONG", 65007.35),
    ("2026-08-19 14:09:48.419", "LONG", 65361.42),
    ("2026-08-19 16:38:02.630", "LONG", 68817.81),
    ("2026-08-19 17:35:28.953", "LONG", 68023.25),
    ("2026-08-19 18:18:16.507", "SHORT", 68164.71),
    ("2026-08-19 18:37:23.365", "LONG", 68287.1),
    ("2026-08-19 19:08:21.422", "SHORT", 68352.28),
    ("2026-08-19 19:34:25.149", "LONG", 68313.61),
    ("2026-08-19 20:03:45.066", "SHORT", 68373.87),
    ("2026-08-19 21:43:38.412", "LONG", 69595.87),
    ("2026-08-19 22:17:14.280", "LONG", 69334.71),
    ("2026-08-19 23:00:12.201", "SHORT", 69208.11),
    ("2026-08-19 23:34:32.208", "LONG", 69320.01),
    ("2026-08-20 01:00:25.190", "LONG", 69399.43),
    ("2026-08-20 01:44:12.048", "SHORT", 69555.64),
    ("2026-08-20 03:00:18.220", "LONG", 69309.31),
    ("2026-08-20 04:30:13.359", "SHORT", 69277.69),
    ("2026-08-20 04:39:51.019", "LONG", 69346.82),
    ("2026-08-20 08:13:08.579", "LONG", 71058.16),
    ("2026-08-20 08:35:26.169", "SHORT", 71150.22),
    ("2026-08-20 09:03:58.965", "LONG", 71697.49),
    ("2026-08-20 09:34:09.586", "SHORT", 71865.07),
    ("2026-08-20 09:37:15.426", "LONG", 71988.34),
    ("2026-08-20 10:39:02.677", "LONG", 71890.53),
    ("2026-08-20 10:51:43.819", "SHORT", 71810.68),
    ("2026-08-20 11:13:09.939", "LONG", 71924.43),
    ("2026-08-20 11:43:04.574", "SHORT", 72147.02),
    ("2026-08-20 12:00:53.833", "LONG", 71930.83),
    ("2026-08-20 12:38:05.805", "SHORT", 71714.0),
    ("2026-08-20 13:17:36.186", "SHORT", 71925.56),
    ("2026-08-20 20:12:48.474", "LONG", 72739.8),
    ("2026-08-21 00:12:20.417", "SHORT", 73227.0),
    ("2026-08-21 00:44:34.108", "LONG", 73749.0),
    ("2026-08-21 00:53:59.189", "LONG", 73750.0),
    ("2026-08-21 01:18:49.692", "LONG", 74614.97),
    ("2026-08-21 01:26:45.033", "SHORT", 74488.15),
    ("2026-08-21 01:35:22.797", "SHORT", 74703.01),
    ("2026-08-21 02:24:00.172", "LONG", 74799.01),
    ("2026-08-21 03:08:40.080", "LONG", 74347.92),
    ("2026-08-21 04:08:22.503", "SHORT", 74703.21),
    ("2026-08-21 05:37:30.209", "LONG", 75289.5),
    ("2026-08-21 05:47:30.807", "SHORT", 75125.12),
    ("2026-08-21 06:17:31.552", "LONG", 75420.33),
    ("2026-08-21 06:31:29.611", "SHORT", 75374.77),
    ("2026-08-21 07:03:16.993", "LONG", 75808.81),
    ("2026-08-21 07:49:31.459", "SHORT", 76332.68),
    ("2026-08-21 07:58:17.235", "LONG", 76392.63),
    ("2026-08-21 08:17:04.513", "LONG", 76995.35),
    ("2026-08-21 08:29:09.170", "LONG", 77105.37),
    ("2026-08-21 09:03:57.223", "LONG", 77923.53),
    ("2026-08-21 09:28:47.168", "SHORT", 77707.81),
    ("2026-08-21 10:50:51.494", "LONG", 77786.2),
    ("2026-08-21 11:28:57.121", "LONG", 77445.34),
    ("2026-08-21 12:23:20.804", "LONG", 76879.02),
    ("2026-08-21 12:46:33.548", "LONG", 77355.87),
    ("2026-08-21 14:55:04.328", "SHORT", 77231.4),
    ("2026-08-21 16:18:13.104", "LONG", 77186.18),
    ("2026-08-21 16:34:25.681", "SHORT", 77273.99),
    ("2026-08-21 18:02:28.399", "LONG", 77454.24),
    ("2026-08-21 19:11:53.779", "SHORT", 76989.25),
    ("2026-08-21 20:46:10.942", "LONG", 77447.34),
    ("2026-08-21 20:51:16.638", "SHORT", 77397.27),
    ("2026-08-21 21:44:13.535", "LONG", 78216.79),
    ("2026-08-21 22:09:05.542", "SHORT", 78558.64),
    ("2026-08-21 22:54:10.146", "SHORT", 78505.25),
    ("2026-08-21 23:14:13.184", "LONG", 78539.95),
    ("2026-08-21 23:47:54.687", "LONG", 78292.91),
    ("2026-08-22 00:23:44.727", "SHORT", 78009.44),
    ("2026-08-22 01:55:00.381", "SHORT", 77799.59),
    ("2026-08-22 02:47:20.795", "LONG", 78379.32),
    ("2026-08-22 03:27:20.562", "LONG", 78736.19),
    ("2026-08-22 05:21:25.376", "LONG", 77068.36),
    ("2026-08-22 05:37:49.835", "LONG", 77251.8),
    ("2026-08-22 06:16:01.158", "LONG", 77252.4),
    ("2026-08-22 08:17:27.317", "SHORT", 77290.19),
    ("2026-08-22 09:07:44.302", "LONG", 77142.98),
    ("2026-08-22 10:02:01.391", "LONG", 76999.35),
    ("2026-08-22 11:14:57.922", "LONG", 76979.64),
]


def analyze_trade(bars_high, bars_low, bars_close, entry_idx, side, entry_price, n):
    running_mfe = 0.0
    running_mae = 0.0
    hit_neg500 = False
    time_to_hit_neg500 = None
    peak_mfe_before_neg500 = 0.0
    time_of_peak_before_neg500 = 0

    end_idx = min(entry_idx + MAX_LOOKFORWARD_MIN, n)
    for offset, i in enumerate(range(entry_idx, end_idx)):
        h, l = bars_high[i], bars_low[i]
        if side == "LONG":
            fav = h - entry_price
            adv = l - entry_price
        else:
            fav = entry_price - l
            adv = entry_price - h

        if fav > running_mfe:
            running_mfe = fav
            if not hit_neg500:
                peak_mfe_before_neg500 = running_mfe
                time_of_peak_before_neg500 = offset
        if adv < running_mae:
            running_mae = adv

        if running_mae <= -ADVERSE_THRESHOLD and not hit_neg500:
            hit_neg500 = True
            time_to_hit_neg500 = offset

        if running_mfe >= ADVERSE_THRESHOLD and running_mae <= -ADVERSE_THRESHOLD:
            break

    return dict(
        final_mfe=running_mfe, final_mae=running_mae,
        hit_neg500=hit_neg500, time_to_hit_neg500=time_to_hit_neg500,
        peak_mfe_before_neg500=peak_mfe_before_neg500,
        time_of_peak_before_neg500=time_of_peak_before_neg500,
    )


def main():
    print(f"Loading {IN_PATH} ...")
    df = pd.read_excel(IN_PATH)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    time_to_idx = {t: i for i, t in enumerate(df["Time"])}

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    results = []
    for entry_time_str, side, entry_price in TRADES:
        entry_time = pd.Timestamp(entry_time_str).round("min")
        if entry_time not in time_to_idx:
            print(f"  SKIP: {entry_time} not found in data")
            continue
        idx = time_to_idx[entry_time]
        r = analyze_trade(highs, lows, closes, idx, side, entry_price, n)
        r["entry_time"] = entry_time
        r["side"] = side
        r["entry_price"] = entry_price
        results.append(r)

    res_df = pd.DataFrame(results)
    print(f"\nAnalyzed {len(res_df)} trades\n")

    hit500 = res_df[res_df["hit_neg500"]]
    print("=" * 70)
    print(f"Q1: Trades that eventually reached -{ADVERSE_THRESHOLD} adverse: {len(hit500)} / {len(res_df)}")
    if len(hit500) > 0:
        print(f"    Time (minutes) from entry to the favorable PEAK before the reversal:")
        print(f"    Mean: {hit500['time_of_peak_before_neg500'].mean():.1f} min")
        print(f"    Median: {hit500['time_of_peak_before_neg500'].median():.1f} min")
        print(f"    Min/Max: {hit500['time_of_peak_before_neg500'].min()}/{hit500['time_of_peak_before_neg500'].max()} min")
        print(f"    Peak favorable value reached before reversal -- mean: ${hit500['peak_mfe_before_neg500'].mean():.1f}")

    clean_favorable = res_df[(res_df["final_mae"] > -CLEAN_TOLERANCE) & (res_df["final_mfe"] > 0)]
    print()
    print("=" * 70)
    print(f"Q2: Trades that moved 'almost straight' favorable (MAE never worse than -{CLEAN_TOLERANCE}): "
          f"{len(clean_favorable)} / {len(res_df)}")

    straight_against = res_df[res_df["hit_neg500"] & (res_df["peak_mfe_before_neg500"] < CLEAN_TOLERANCE)]
    print()
    print("=" * 70)
    print(f"Q3: Trades that went straight against us to -{ADVERSE_THRESHOLD} "
          f"(peak favorable move stayed under {CLEAN_TOLERANCE} first): {len(straight_against)} / {len(res_df)}")

    res_df.to_excel(OUT_PATH, index=False)
    print(f"\nFull per-trade results written to {OUT_PATH}")


if __name__ == "__main__":
    main()

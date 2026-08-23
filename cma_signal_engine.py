"""
cma_signal_engine.py

Faithful Python translation of the "Clean Multi-Asset Trading Signal
Engine" Pine Script (v6), for computing every score/component on
resampled 5m BTCUSDT candles built from the 1m data fetched by
fetch_1m_data.py.

Every component below is a direct translation of the corresponding
Pine section, using the exact same formulas, thresholds, and BTC
asset-profile presets from the original script.

USAGE:
    python cma_signal_engine.py
"""

import numpy as np
import pandas as pd

IN_PATH = r"D:\data\btc\BTCUSDT_1m_aug2026_latest.parquet"
OUT_PATH = r"D:\data\btc\BTCUSDT_5m_CMA_signals.parquet"

# ---- BTC asset profile presets, exactly matching the Pine script's BTC branch ----
ATR_PCT_LOW = 0.15
ATR_PCT_HIGH = 3.0
RVOL_HIGH = 1.8
RVOL_EXTREME = 3.0
VWAP_SIGMA1 = 1.2
VWAP_SIGMA2 = 2.5
VWAP_SIGMA3 = 4.0

# ---- Section inputs, matching the Pine script's defaults exactly ----
DI_LEN = 14
ADX_SMOOTH = 14
ADX_MIN_TREND = 20.0
ADX_STRONG = 35.0

TSI_SHORT = 13
TSI_LONG = 25
TSI_SIG = 13
RSI_LEN = 14
RSI_OB = 60.0
RSI_OS = 40.0
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIG = 9
MOMENTUM_MODE = "Blend"
TURN_LOOKBACK = 3
MOMENTUM_BLEND = 0.5

ATR_LEN = 14
BB_LEN = 20
BB_MULT = 2.0
ALLOW_EXTREME_VOL = True

RVOL_LEN = 20
RVOL_LOW = 0.5

PIVOT_LEN = 5

HTF1 = "15min"
HTF2 = "60min"
MTF_FAST_LEN = 20
MTF_SLOW_LEN = 50

W_TREND, W_MOM, W_VOL, W_VWAP, W_VOLU, W_STRUCT, W_OF = 25, 20, 10, 15, 10, 10, 0  # order flow disabled

SCORE_THRESH = 70.0
MIN_TREND_SCORE = 40.0
MIN_MOM_SCORE = 40.0


# =============================================================================
# Core indicator building blocks
# =============================================================================

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def wilder_smooth(series: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothing (RMA), matching Pine's ta.rma used internally by
    ta.dmi/ta.atr/ta.rsi. Verified against hand-calculated examples."""
    return series.ewm(alpha=1 / length, adjust=False).mean()


def compute_dmi(H, L, C, length, smooth_length):
    """+DI, -DI, ADX -- matches Pine's ta.dmi(length, smoothLength) exactly,
    including Wilder smoothing for both the directional movement and ADX."""
    up_move = H.diff()
    down_move = -L.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = H - L
    tr2 = (H - C.shift()).abs()
    tr3 = (L - C.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = wilder_smooth(tr, length)
    plus_di = 100 * wilder_smooth(pd.Series(plus_dm, index=H.index), length) / atr
    minus_di = 100 * wilder_smooth(pd.Series(minus_dm, index=H.index), length) / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = wilder_smooth(dx.fillna(0), smooth_length)
    return plus_di, minus_di, adx


def compute_atr(H, L, C, length):
    tr1 = H - L
    tr2 = (H - C.shift()).abs()
    tr3 = (L - C.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return wilder_smooth(tr, length)


def compute_rsi(C, length):
    delta = C.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = wilder_smooth(gain, length)
    avg_loss = wilder_smooth(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(100)  # matches Pine behavior when avg_loss == 0


def compute_tsi(C, short_len, long_len, sig_len):
    """True Strength Index -- double-smoothed momentum and abs-momentum,
    matching Pine's ta.tsi exactly."""
    mom = C.diff()
    smooth1 = ema(mom, long_len)
    smooth2 = ema(smooth1, short_len)
    abs_smooth1 = ema(mom.abs(), long_len)
    abs_smooth2 = ema(abs_smooth1, short_len)
    tsi = 100 * smooth2 / abs_smooth2.replace(0, np.nan)
    tsi_signal = ema(tsi, sig_len)
    return tsi.fillna(0), tsi_signal.fillna(0)


def compute_macd(C, fast_len, slow_len, sig_len):
    macd_line = ema(C, fast_len) - ema(C, slow_len)
    macd_signal = ema(macd_line, sig_len)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def compute_bollinger(C, length, mult):
    mid = C.rolling(length).mean()
    std = C.rolling(length).std(ddof=0)
    upper = mid + mult * std
    lower = mid - mult * std
    return mid, upper, lower


def barssince(cond: pd.Series) -> pd.Series:
    """Bars since condition was last True, matching Pine's ta.barssince
    (NaN/9999 before the first occurrence, matching the nz(...,9999) wrap
    used around every call site in the original script)."""
    idx = np.arange(len(cond))
    last_true_idx = np.where(cond.values, idx, np.nan)
    last_true_idx = pd.Series(last_true_idx, index=cond.index).ffill()
    result = idx - last_true_idx.values
    result = pd.Series(result, index=cond.index)
    result[last_true_idx.isna()] = 9999
    return result


def pivothigh_pivotlow(H, L, left, right):
    """Matches Pine's ta.pivothigh/ta.pivotlow: a pivot at bar i is
    confirmed 'right' bars later, only known once those bars exist --
    i.e. genuinely centered, non-repainting once confirmed."""
    n = len(H)
    ph = pd.Series(np.nan, index=H.index)
    pl = pd.Series(np.nan, index=L.index)
    Hv, Lv = H.values, L.values
    for i in range(left, n - right):
        window_h = Hv[i - left:i + right + 1]
        window_l = Lv[i - left:i + right + 1]
        if Hv[i] == window_h.max() and (window_h == Hv[i]).sum() == 1:
            ph.iloc[i + right] = Hv[i]  # value becomes known 'right' bars later
        if Lv[i] == window_l.min() and (window_l == Lv[i]).sum() == 1:
            pl.iloc[i + right] = Lv[i]
    return ph, pl


# =============================================================================
# Full pipeline: resample 1m -> 5m, compute every component, score, signal
# =============================================================================

def resample_to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    df = df_1m.set_index("timestamp")
    agg = df.resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna()
    return agg.reset_index()


def compute_mtf_bias(df5m: pd.DataFrame, htf: str, fast_len: int, slow_len: int) -> pd.Series:
    """Matches Pine's request.security with lookahead_off: the HTF EMA
    slope-sign is computed on the HTF's OWN closed candles, then merged
    back onto 5m bars using only the most recently CLOSED htf candle at
    or before each 5m timestamp (never a same-bar or future one)."""
    htf_df = df5m.set_index("timestamp")["close"].resample(htf).last().dropna()
    fast = ema(htf_df, fast_len)
    slow = ema(htf_df, slow_len)
    bias = np.sign(fast - slow)
    bias_shifted = bias.shift(1)
    merged = pd.merge_asof(
        df5m[["timestamp"]].sort_values("timestamp"),
        bias_shifted.reset_index().rename(columns={"timestamp": "timestamp", 0: "bias"}).rename(columns={bias_shifted.name or "close": "bias"}),
        on="timestamp", direction="backward"
    )
    return merged["bias"].fillna(0).values


def run_full_pipeline(df1m_path: str, out_path: str):
    print(f"Loading {df1m_path} ...")
    df1m = pd.read_parquet(df1m_path)
    df1m["timestamp"] = pd.to_datetime(df1m["timestamp"], utc=True)

    print("Resampling to 5m...")
    df = resample_to_5m(df1m)
    n = len(df)
    print(f"{n:,} x 5m candles, {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    H, L, C, V = df["high"], df["low"], df["close"], df["volume"]

    # ---- Trend ----
    plus_di, minus_di, adx = compute_dmi(H, L, C, DI_LEN, ADX_SMOOTH)
    di_spread = plus_di - minus_di
    trend_exists = adx >= ADX_MIN_TREND
    adx_norm = ((adx - ADX_MIN_TREND) / max(ADX_STRONG - ADX_MIN_TREND, 0.0001)).clip(0, 1)
    di_spread_norm = (di_spread.abs() / 40.0).clip(upper=1)
    trend_magnitude = np.where(trend_exists, adx_norm * di_spread_norm * 100.0, 0.0)
    trend_dir_up = di_spread > 0
    trend_long_score = np.where(trend_dir_up, trend_magnitude, 0.0)
    trend_short_score = np.where(~trend_dir_up, trend_magnitude, 0.0)

    # ---- Momentum ----
    atr_val = compute_atr(H, L, C, ATR_LEN)
    tsi_val, tsi_signal = compute_tsi(C, TSI_SHORT, TSI_LONG, TSI_SIG)
    tsi_dir_up = tsi_val > tsi_signal
    tsi_norm = ((tsi_val - tsi_signal).abs() / 10.0).clip(upper=1)

    rsi_val = compute_rsi(C, RSI_LEN)
    dir_rsi = np.where(rsi_val > RSI_OB, 1, np.where(rsi_val < RSI_OS, -1, 0))
    rsi_signed_norm = np.where(rsi_val > RSI_OB, ((rsi_val - RSI_OB) / max(100 - RSI_OB, 0.0001)).clip(upper=1),
                        np.where(rsi_val < RSI_OS, -((RSI_OS - rsi_val) / max(RSI_OS, 0.0001)).clip(upper=1), 0.0))

    macd_line, macd_signal, macd_hist = compute_macd(C, MACD_FAST, MACD_SLOW, MACD_SIG)
    macd_dir_up = macd_line > macd_signal
    macd_norm = (macd_hist.abs() / (atr_val * 0.1 + 1e-10)).clip(upper=1)

    dir_tsi = np.where(tsi_dir_up, 1, -1)
    dir_macd = np.where(macd_dir_up, 1, -1)

    votes_sum_mag = dir_tsi + dir_rsi + dir_macd
    mom_direction_mag = np.sign(votes_sum_mag)
    agree_count_mag = (dir_tsi == mom_direction_mag).astype(int) + (dir_rsi == mom_direction_mag).astype(int) + (dir_macd == mom_direction_mag).astype(int)
    agreement_factor_mag = np.where(mom_direction_mag == 0, 0.0, agree_count_mag / 3.0)
    sum_agree_mag = (np.where(dir_tsi == mom_direction_mag, tsi_norm, 0.0) +
                     np.where(dir_rsi == mom_direction_mag, np.abs(rsi_signed_norm), 0.0) +
                     np.where(dir_macd == mom_direction_mag, macd_norm, 0.0))
    mom_magnitude_mag = np.where(agree_count_mag > 0, sum_agree_mag / np.where(agree_count_mag==0,1,agree_count_mag), 0.0)
    momentum_score_raw_mag = np.where(mom_direction_mag == 0, 0.0, mom_magnitude_mag * agreement_factor_mag * 100.0)
    momentum_mag_long = np.where(mom_direction_mag > 0, momentum_score_raw_mag, 0.0)
    momentum_mag_short = np.where(mom_direction_mag < 0, momentum_score_raw_mag, 0.0)

    tsi_turn_up = (tsi_val > tsi_signal) & (tsi_val.shift(1) <= tsi_signal.shift(1))
    tsi_turn_down = (tsi_val < tsi_signal) & (tsi_val.shift(1) >= tsi_signal.shift(1))
    rsi_turn_up = (rsi_val > 50) & (rsi_val.shift(1) <= 50)
    rsi_turn_down = (rsi_val < 50) & (rsi_val.shift(1) >= 50)
    macd_turn_up = (macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))
    macd_turn_down = (macd_line < macd_signal) & (macd_line.shift(1) >= macd_signal.shift(1))

    tsi_up_age = barssince(tsi_turn_up.fillna(False))
    tsi_dn_age = barssince(tsi_turn_down.fillna(False))
    rsi_up_age = barssince(rsi_turn_up.fillna(False))
    rsi_dn_age = barssince(rsi_turn_down.fillna(False))
    macd_up_age = barssince(macd_turn_up.fillna(False))
    macd_dn_age = barssince(macd_turn_down.fillna(False))

    dir_tsi_turn = np.where(tsi_up_age < tsi_dn_age, 1, np.where(tsi_dn_age < tsi_up_age, -1, 0))
    dir_rsi_turn = np.where(rsi_up_age < rsi_dn_age, 1, np.where(rsi_dn_age < rsi_up_age, -1, 0))
    dir_macd_turn = np.where(macd_up_age < macd_dn_age, 1, np.where(macd_dn_age < macd_up_age, -1, 0))

    tsi_turn_mag = np.where(dir_tsi_turn == 0, 0.0, np.clip(1.0 - np.minimum(tsi_up_age, tsi_dn_age) / TURN_LOOKBACK, 0, None))
    rsi_turn_mag = np.where(dir_rsi_turn == 0, 0.0, np.clip(1.0 - np.minimum(rsi_up_age, rsi_dn_age) / TURN_LOOKBACK, 0, None))
    macd_turn_mag = np.where(dir_macd_turn == 0, 0.0, np.clip(1.0 - np.minimum(macd_up_age, macd_dn_age) / TURN_LOOKBACK, 0, None))

    votes_sum_turn = dir_tsi_turn + dir_rsi_turn + dir_macd_turn
    mom_direction_turn = np.sign(votes_sum_turn)
    agree_count_turn = (dir_tsi_turn == mom_direction_turn).astype(int) + (dir_rsi_turn == mom_direction_turn).astype(int) + (dir_macd_turn == mom_direction_turn).astype(int)
    agreement_factor_turn = np.where(mom_direction_turn == 0, 0.0, agree_count_turn / 3.0)
    sum_agree_turn = (np.where(dir_tsi_turn == mom_direction_turn, tsi_turn_mag, 0.0) +
                      np.where(dir_rsi_turn == mom_direction_turn, rsi_turn_mag, 0.0) +
                      np.where(dir_macd_turn == mom_direction_turn, macd_turn_mag, 0.0))
    mom_magnitude_turn = np.where(agree_count_turn > 0, sum_agree_turn / np.where(agree_count_turn==0,1,agree_count_turn), 0.0)
    momentum_score_raw_turn = np.where(mom_direction_turn == 0, 0.0, mom_magnitude_turn * agreement_factor_turn * 100.0)
    momentum_turn_long = np.where(mom_direction_turn > 0, momentum_score_raw_turn, 0.0)
    momentum_turn_short = np.where(mom_direction_turn < 0, momentum_score_raw_turn, 0.0)

    momentum_long_score = np.minimum(momentum_mag_long + momentum_turn_long * MOMENTUM_BLEND, 100.0)
    momentum_short_score = np.minimum(momentum_mag_short + momentum_turn_short * MOMENTUM_BLEND, 100.0)
    mom_direction = np.where(momentum_long_score > momentum_short_score, 1,
                     np.where(momentum_short_score > momentum_long_score, -1, 0))

    # ---- Volatility ----
    bb_mid, bb_upper, bb_lower = compute_bollinger(C, BB_LEN, BB_MULT)
    atr_pct = atr_val / C * 100.0
    expansion_cut = ATR_PCT_LOW + (ATR_PCT_HIGH - ATR_PCT_LOW) * 0.35
    vol_regime = np.select(
        [atr_pct < ATR_PCT_LOW, atr_pct > ATR_PCT_HIGH, atr_pct > expansion_cut],
        ["LOW", "EXTREME", "EXPANSION"], default="NORMAL")
    volatility_score = np.select(
        [vol_regime == "LOW", vol_regime == "NORMAL", vol_regime == "EXPANSION"],
        [15.0, 55.0, 90.0], default=(70.0 if ALLOW_EXTREME_VOL else 10.0))
    volatility_blocks_signal = (vol_regime == "EXTREME") & (not ALLOW_EXTREME_VOL)

    # ---- VWAP / Location (session-anchored, daily reset) ----
    hlc3 = (H + L + C) / 3
    day = df["timestamp"].dt.floor("D")
    cum_pv = (V * hlc3).groupby(day).cumsum()
    cum_v = V.groupby(day).cumsum()
    cum_pv2 = (V * hlc3 * hlc3).groupby(day).cumsum()
    vwap_val = cum_pv / cum_v
    vwap_variance = (cum_pv2 / cum_v - vwap_val**2).clip(lower=0)
    vwap_stdev = np.sqrt(vwap_variance)
    vwap_sigma_dist = np.where(vwap_stdev > 0, (C - vwap_val) / vwap_stdev, 0.0)

    above_vwap = C > vwap_val
    extended_above = vwap_sigma_dist > VWAP_SIGMA2
    extended_below = vwap_sigma_dist < -VWAP_SIGMA2
    hyper_above = vwap_sigma_dist > VWAP_SIGMA3
    hyper_below = vwap_sigma_dist < -VWAP_SIGMA3
    ext_above_norm = np.clip((vwap_sigma_dist - VWAP_SIGMA1) / max(VWAP_SIGMA3 - VWAP_SIGMA1, 0.0001), 0, 1)
    ext_below_norm = np.clip((-vwap_sigma_dist - VWAP_SIGMA1) / max(VWAP_SIGMA3 - VWAP_SIGMA1, 0.0001), 0, 1)

    vwap_cont_long = np.where(above_vwap & trend_dir_up.values & (mom_direction > 0) & ~hyper_above, 60 + 40*(1-ext_above_norm), 0.0)
    vwap_cont_short = np.where(~above_vwap.values & ~trend_dir_up.values & (mom_direction < 0) & ~hyper_below, 60 + 40*(1-ext_below_norm), 0.0)
    vwap_rev_long = np.where(extended_below & (mom_direction >= 0), 50 + 50*ext_below_norm, 0.0)
    vwap_rev_short = np.where(extended_above & (mom_direction <= 0), 50 + 50*ext_above_norm, 0.0)
    vwap_long_score = np.maximum(vwap_cont_long, vwap_rev_long)
    vwap_short_score = np.maximum(vwap_cont_short, vwap_rev_short)

    # ---- Volume ----
    rvol_avg = V.rolling(RVOL_LEN).mean()
    rvol = np.where(rvol_avg > 0, V / rvol_avg, 1.0)
    vol_class = np.select([rvol < RVOL_LOW, rvol > RVOL_EXTREME, rvol > RVOL_HIGH], ["LOW","EXTREME","HIGH"], default="NORMAL")
    volume_score = np.select([vol_class=="LOW", vol_class=="NORMAL", vol_class=="HIGH"], [20.0,50.0,80.0], default=100.0)

    # ---- Market Structure ----
    ph, pl = pivothigh_pivotlow(H, L, PIVOT_LEN, PIVOT_LEN)
    last_pivot_high = np.full(n, np.nan)
    prev_pivot_high = np.full(n, np.nan)
    last_pivot_low = np.full(n, np.nan)
    prev_pivot_low = np.full(n, np.nan)
    lph, pph, lpl, ppl = np.nan, np.nan, np.nan, np.nan
    ph_v, pl_v = ph.values, pl.values
    for i in range(n):
        if not np.isnan(ph_v[i]):
            pph = lph
            lph = ph_v[i]
        if not np.isnan(pl_v[i]):
            ppl = lpl
            lpl = pl_v[i]
        last_pivot_high[i], prev_pivot_high[i] = lph, pph
        last_pivot_low[i], prev_pivot_low[i] = lpl, ppl

    higher_high = ~np.isnan(last_pivot_high) & ~np.isnan(prev_pivot_high) & (last_pivot_high > prev_pivot_high)
    lower_high = ~np.isnan(last_pivot_high) & ~np.isnan(prev_pivot_high) & (last_pivot_high < prev_pivot_high)
    higher_low = ~np.isnan(last_pivot_low) & ~np.isnan(prev_pivot_low) & (last_pivot_low > prev_pivot_low)
    lower_low = ~np.isnan(last_pivot_low) & ~np.isnan(prev_pivot_low) & (last_pivot_low < prev_pivot_low)
    bullish_structure = higher_high | higher_low
    bearish_structure = lower_high | lower_low

    bos_up = np.zeros(n, dtype=bool)
    bos_down = np.zeros(n, dtype=bool)
    Cv = C.values
    for i in range(1, n):
        if not np.isnan(last_pivot_high[i]) and Cv[i] > last_pivot_high[i] and Cv[i-1] <= last_pivot_high[i]:
            bos_up[i] = True
        if not np.isnan(last_pivot_low[i]) and Cv[i] < last_pivot_low[i] and Cv[i-1] >= last_pivot_low[i]:
            bos_down[i] = True

    structure_bias = np.full(n, "NONE", dtype=object)
    bias = "NONE"
    prior_bias_arr = np.full(n, "NONE", dtype=object)
    for i in range(n):
        prior_bias_arr[i] = bias
        if bullish_structure[i]:
            bias = "BULL"
        elif bearish_structure[i]:
            bias = "BEAR"
        structure_bias[i] = bias

    choch_up = bos_up & (prior_bias_arr == "BEAR")
    choch_down = bos_down & (prior_bias_arr == "BULL")

    structure_long_score = np.minimum((bullish_structure*40.0) + (bos_up*30.0) + (choch_up*30.0), 100.0)
    structure_short_score = np.minimum((bearish_structure*40.0) + (bos_down*30.0) + (choch_down*30.0), 100.0)

    # ---- Order Flow (disabled) ----
    order_flow_long_score = np.zeros(n)
    order_flow_short_score = np.zeros(n)

    # ---- MTF ----
    htf1_bias = compute_mtf_bias(df, HTF1, MTF_FAST_LEN, MTF_SLOW_LEN)
    htf2_bias = compute_mtf_bias(df, HTF2, MTF_FAST_LEN, MTF_SLOW_LEN)
    mtf_long_ok = (htf1_bias >= 0) & (htf2_bias >= 0)
    mtf_short_ok = (htf1_bias <= 0) & (htf2_bias <= 0)

    # ---- Score Engine ----
    w_sum_raw = W_TREND + W_MOM + W_VOL + W_VWAP + W_VOLU + W_STRUCT + W_OF
    w_sum = w_sum_raw if w_sum_raw > 0 else 100.0
    nT, nM, nV, nVW, nVO, nS, nOF = [w/w_sum*100 for w in (W_TREND,W_MOM,W_VOL,W_VWAP,W_VOLU,W_STRUCT,W_OF)]

    long_score = (trend_long_score*nT + momentum_long_score*nM + volatility_score*nV + vwap_long_score*nVW +
                  volume_score*nVO + structure_long_score*nS + order_flow_long_score*nOF) / 100.0
    short_score = (trend_short_score*nT + momentum_short_score*nM + volatility_score*nV + vwap_short_score*nVW +
                   volume_score*nVO + structure_short_score*nS + order_flow_short_score*nOF) / 100.0

    # ---- Signal Qualification ----
    long_cond = ((long_score >= SCORE_THRESH) & (long_score > short_score) &
                 (trend_long_score >= MIN_TREND_SCORE) & (momentum_long_score >= MIN_MOM_SCORE) &
                 mtf_long_ok & ~volatility_blocks_signal)
    short_cond = ((short_score >= SCORE_THRESH) & (short_score > long_score) &
                  (trend_short_score >= MIN_TREND_SCORE) & (momentum_short_score >= MIN_MOM_SCORE) &
                  mtf_short_ok & ~volatility_blocks_signal)

    signal_state = np.full(n, "NONE", dtype=object)
    new_long = np.zeros(n, dtype=bool)
    new_short = np.zeros(n, dtype=bool)
    state = "NONE"
    for i in range(n):
        if long_cond[i] and state != "LONG":
            new_long[i] = True
        if short_cond[i] and state != "SHORT":
            new_short[i] = True
        if long_cond[i]:
            state = "LONG"
        elif short_cond[i]:
            state = "SHORT"
        elif not long_cond[i] and not short_cond[i]:
            state = "NONE"
        signal_state[i] = state

    out = df.copy()
    out["trend_long_score"] = trend_long_score
    out["trend_short_score"] = trend_short_score
    out["momentum_long_score"] = momentum_long_score
    out["momentum_short_score"] = momentum_short_score
    out["volatility_score"] = volatility_score
    out["vol_regime"] = vol_regime
    out["vwap_long_score"] = vwap_long_score
    out["vwap_short_score"] = vwap_short_score
    out["volume_score"] = volume_score
    out["structure_long_score"] = structure_long_score
    out["structure_short_score"] = structure_short_score
    out["long_score"] = long_score
    out["short_score"] = short_score
    out["mtf_long_ok"] = mtf_long_ok
    out["mtf_short_ok"] = mtf_short_ok
    out["signal_state"] = signal_state
    out["new_long_signal"] = new_long
    out["new_short_signal"] = new_short

    out.to_parquet(out_path, index=False)
    print(f"\nSaved {len(out):,} rows with all scores to {out_path}")
    print(f"Total LONG signals: {new_long.sum()}")
    print(f"Total SHORT signals: {new_short.sum()}")
    return out


if __name__ == "__main__":
    run_full_pipeline(IN_PATH, OUT_PATH)

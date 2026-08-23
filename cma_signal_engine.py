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
    dir_rsi = np.where(rsi_val > RSI_OB,

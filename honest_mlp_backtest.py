"""
honest_mlp_backtest.py

Reproduces zaid-24's MLP classifier approach (same features, same
architecture, same percentile-based labeling scheme) but FIXES the
critical flaws we found in the original:

  1. A real chronological train/test split (train on the first 80% of
     time, test ONLY on the final 20% -- the model never sees test-period
     data during training or feature/threshold selection).
  2. "Win rate" is computed from the MODEL'S OWN PREDICTIONS on the held-out
     test set, checked against what price ACTUALLY did afterward -- not
     from the look-ahead labels themselves.
  3. Sharpe/Sortino/drawdown are computed from the actual simulated trade
     returns on the test set, not random noise or hardcoded placeholders.

This is the fair version of their test: does their feature set + MLP
architecture have any real, honest edge?

USAGE:
    pip install pandas numpy scikit-learn imbalanced-learn pyarrow
    python honest_mlp_backtest.py
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from collections import Counter

try:
    from imblearn.under_sampling import RandomUnderSampler
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

DEFAULT_PARQUET = r"D:\data\btc\BTCUSDT_1m_ohlcv_oi_delta_cvd_5y.parquet"
TRAIN_FRACTION = 0.8  # chronological -- first 80% train, last 20% test


def load_and_resample_15m(parquet_path: str) -> pd.DataFrame:
    print(f"Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    agg = df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna()
    agg = agg.reset_index()
    print(f"Resampled to {len(agg):,} x 15m candles, {agg['timestamp'].iloc[0]} to {agg['timestamp'].iloc[-1]}")
    return agg


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d["close"]

    d["20_MA"] = close.rolling(20).mean()
    d["20_std"] = close.rolling(20).std()
    d["upper_band"] = d["20_MA"] + 2 * d["20_std"]
    d["lower_band"] = d["20_MA"] - 2 * d["20_std"]
    d["UBB"] = close - d["upper_band"]
    d["LBB"] = close - d["lower_band"]
    d["BBW"] = (d["upper_band"] - d["lower_band"]) / d["20_MA"]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    prior_close = close.shift(1)
    bp = close - pd.concat([d["low"], prior_close], axis=1).min(axis=1)
    tr = pd.concat([d["high"], prior_close], axis=1).max(axis=1) - pd.concat([d["low"], prior_close], axis=1).min(axis=1)
    avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
    avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
    avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
    d["ULTOSC"] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7

    d["Close_Price_Pct_Variation"] = close.pct_change()
    roll_mean = close.rolling(30).mean()
    roll_std = close.rolling(30).std()
    d["Z_Score"] = (close - roll_mean) / roll_std.replace(0, np.nan)

    for span in [1, 20, 50, 100]:
        d[f"ema_{span}"] = close.ewm(span=span, min_periods=span).mean()
    d["crossover_1_20"] = d["ema_1"] - d["ema_20"]
    d["crossover_20_50"] = d["ema_20"] - d["ema_50"]
    d["crossover_50_100"] = d["ema_50"] - d["ema_100"]
    d["crossover_1_50"] = d["ema_1"] - d["ema_50"]

    ema12 = close.ewm(span=12, min_periods=12).mean()
    ema26 = close.ewm(span=26, min_periods=26).mean()
    d["macd_line"] = ema12 - ema26
    d["signal_line"] = d["macd_line"].ewm(span=9, min_periods=9).mean()
    d["macd_histogram"] = d["macd_line"] - d["signal_line"]

    return d


FEATURE_COLS = ['close', '20_MA', '20_std', 'upper_band', 'lower_band', 'UBB', 'LBB', 'BBW',
                 'RSI', 'ULTOSC', 'Close_Price_Pct_Variation', 'Z_Score', 'ema_1', 'ema_20',
                 'ema_50', 'ema_100', 'crossover_1_20', 'crossover_20_50', 'crossover_50_100',
                 'crossover_1_50', 'macd_line', 'signal_line', 'macd_histogram']


def compute_thresholds(pct_change: pd.Series):
    a = np.percentile(pct_change.dropna(), 85)
    b = np.percentile(pct_change.dropna(), 99.7)
    return a, b


def labeling_algorithm(close: pd.Series, forW: int, a: float, b: float, f: float = 0.005) -> np.ndarray:
    n = len(close)
    labels = np.zeros(n, dtype=int)
    close_vals = close.values
    for i in range(n - forW):
        R = ((1 - f) * close_vals[i + forW] - (1 + f) * close_vals[i]) / close_vals[i]
        if a < abs(R) < b:
            labels[i] = 1 if R > 0 else -1
    return labels


def run(parquet_path: str, forW: int = 3):
    bars = load_and_resample_15m(parquet_path)
    feats = compute_features(bars)
    feats = feats.dropna().reset_index(drop=True)
    print(f"{len(feats):,} rows remain after indicator warm-up (dropna)")

    split_idx = int(len(feats) * TRAIN_FRACTION)
    train_df = feats.iloc[:split_idx].reset_index(drop=True)
    test_df = feats.iloc[split_idx:].reset_index(drop=True)
    print(f"Train: {train_df['timestamp'].iloc[0]} to {train_df['timestamp'].iloc[-1]} ({len(train_df):,} rows)")
    print(f"Test:  {test_df['timestamp'].iloc[0]} to {test_df['timestamp'].iloc[-1]} ({len(test_df):,} rows) -- MODEL NEVER SEES THIS DURING TRAINING")

    pct_change_train = (train_df["close"] - train_df["close"].shift(1)) / train_df["close"].shift(1)
    a, b = compute_thresholds(pct_change_train)
    train_labels = labeling_algorithm(train_df["close"], forW, a, b)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_labels

    if HAS_IMBLEARN:
        counter = Counter(y_train)
        print(f"Training label distribution before balancing: {dict(counter)}")
        try:
            sampler = RandomUnderSampler(random_state=42)
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            print(f"After balancing: {dict(Counter(y_train))}")
        except Exception as e:
            print(f"Skipping undersampling ({e}), training on raw class distribution")
    else:
        print("imbalanced-learn not installed -- training on raw (unbalanced) classes")

    print("Training MLP (same architecture as original: 128-64-32, relu, adam)...")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
                         random_state=42, max_iter=300)
    mlp.fit(X_train, y_train)

    X_test = test_df[FEATURE_COLS].values
    predictions = mlp.predict(X_test)
    pred_counts = Counter(predictions)
    print(f"\nTest-set prediction distribution: {dict(pred_counts)}")

    close_test = test_df["close"].values
    trade_returns = []
    for i in range(len(predictions) - forW):
        pred = predictions[i]
        if pred == 0:
            continue
        actual_move = (close_test[i + forW] - close_test[i]) / close_test[i]
        trade_return = actual_move if pred == 1 else -actual_move
        trade_returns.append(trade_return)

    trade_returns = np.array(trade_returns)
    print("\n" + "=" * 70)
    print("HONEST OUT-OF-SAMPLE RESULTS (test period the model never saw)")
    print("=" * 70)
    if len(trade_returns) == 0:
        print("Model never predicted Buy or Sell on the test set -- no trades to evaluate.")
        return

    wins = trade_returns[trade_returns > 0]
    real_win_rate = len(wins) / len(trade_returns) * 100
    print(f"Total signals (Buy/Sell, excl. Hold): {len(trade_returns)}")
    print(f"REAL win rate (predicted direction matched actual future move): {real_win_rate:.2f}%")
    print(f"Mean return per trade: {trade_returns.mean()*100:.4f}%")
    print(f"Total compounded return if trading every signal (no fees): {(np.prod(1+trade_returns)-1)*100:.2f}%")

    if trade_returns.std() > 0:
        sharpe = trade_returns.mean() / trade_returns.std() * np.sqrt(252 * 96 / forW)
        print(f"Sharpe-like ratio (rough annualization): {sharpe:.3f}")

    equity = np.cumprod(1 + trade_returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    print(f"Max drawdown: {drawdown.min()*100:.2f}%")

    bh_return = (close_test[-1] - close_test[0]) / close_test[0] * 100
    print(f"\nFor comparison -- simple buy & hold BTC over the SAME test window: {bh_return:+.2f}%")

    print("\n" + "-" * 70)
    print("IMPORTANT: the compounded-return and Sharpe figures above include")
    print("NO trading fees or slippage. With a win rate anywhere near 50%,")
    print(f"real transaction costs on {len(trade_returns)} trades would likely erase")
    print("most or all of any apparent edge. A win rate meaningfully above")
    print("50% (e.g. >54-55%, net of the true cost per trade) is the real")
    print("bar for 'this might be worth pursuing further' -- not the raw")
    print("compounded return number, which is easy to overstate this way.")
    print("-" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET)
    parser.add_argument("--forward-window", type=int, default=3, help="Bars ahead used for labeling (forW)")
    args = parser.parse_args()
    run(args.parquet, forW=args.forward_window)

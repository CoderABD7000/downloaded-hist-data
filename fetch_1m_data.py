"""
fetch_1m_data.py

Downloads 1-minute BTCUSDT klines from Binance's public REST API
(no API key needed for market data) from Aug 1, 2026 to the latest
available candle, and saves as a single parquet file.

USAGE:
    python fetch_1m_data.py
"""

import time
from datetime import datetime, timezone

import pandas as pd
import requests

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
START = datetime(2026, 8, 1, tzinfo=timezone.utc)
OUT_PATH = r"D:\data\btc\BTCUSDT_1m_aug2026_latest.parquet"
BASE_URL = "https://api.binance.com/api/v3/klines"


def fetch_all():
    start_ms = int(START.timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    all_rows = []
    cur = start_ms

    while cur < now_ms:
        params = dict(symbol=SYMBOL, interval=INTERVAL, startTime=cur, limit=1000)
        r = requests.get(BASE_URL, params=params, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        all_rows.extend(batch)
        last_open_time = batch[-1][0]
        cur = last_open_time + 60_000  # advance 1 minute past the last candle
        print(f"Fetched {len(all_rows):,} candles so far... "
              f"(latest: {datetime.fromtimestamp(last_open_time/1000, tz=timezone.utc)})")
        time.sleep(0.15)  # stay well under Binance's rate limit

    return all_rows


def main():
    print(f"Fetching {SYMBOL} {INTERVAL} candles from {START} to now...")
    rows = fetch_all()

    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume",
              "taker_buy_base", "taker_buy_quote", "trades"]]

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(df):,} rows to {OUT_PATH}")
    print(f"Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")


if __name__ == "__main__":
    main()

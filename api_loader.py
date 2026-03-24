"""
api_loader.py
-------------
Fetches BTCUSDT perpetual futures klines from Binance public API.
No API key required — public endpoint.

Replaces data_loader.py for the api-klines version of the pipeline.
Saves directly to CSV — no DuckDB needed.

Outputs:
  data/df_klines_1m.csv    1-minute candles (raw, ~1.5M rows for 3 years)
  data/df_klines_15m.csv   15-minute candles (pre-aggregated for backtest)

Usage:
  python api_loader.py
"""

import requests
import pandas as pd
import time
from datetime import datetime, timezone

# ── CONFIG ────────────────────────────────────────────────────────────────
SYMBOL     = "BTCUSDT"
INTERVAL   = "1m"
START_DATE = "2023-01-01"
END_DATE   = "2026-01-31"

BASE_URL   = "https://fapi.binance.com/fapi/v1/klines"
LIMIT      = 1000          # max candles per request (Binance limit)
SLEEP_MS   = 0.1           # sleep between requests to avoid rate limit

OUTPUT_1M  = "data/df_klines_1m.csv"
OUTPUT_15M = "data/df_klines_15m.csv"


# ── HELPERS ───────────────────────────────────────────────────────────────
def to_ms(date_str):
    """Convert 'YYYY-MM-DD' string to millisecond timestamp."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def from_ms(ms):
    """Convert millisecond timestamp to datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


# ── FETCH KLINES ──────────────────────────────────────────────────────────
def fetch_klines(symbol, interval, start_ms, end_ms):
    """
    Fetches all klines between start_ms and end_ms.
    Handles pagination automatically — Binance returns max 1000 per request.

    Returns: DataFrame with columns:
        datetime, open, high, low, close, volume, date
    """
    all_candles = []
    current_ms  = start_ms
    total       = 0

    print(f"Fetching {symbol} {interval} candles...")
    print(f"  From: {from_ms(start_ms).strftime('%Y-%m-%d')}")
    print(f"  To:   {from_ms(end_ms).strftime('%Y-%m-%d')}")
    print()

    while current_ms < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": current_ms,
            "endTime":   end_ms,
            "limit":     LIMIT,
        }

        try:
            response = requests.get(BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            candles = response.json()
        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e} — retrying in 5s...")
            time.sleep(5)
            continue

        if not candles:
            break

        all_candles.extend(candles)
        total += len(candles)

        # Next request starts from the last candle's close time + 1ms
        current_ms = candles[-1][6] + 1

        # Progress update every 50k candles
        if total % 50_000 < LIMIT:
            pct = (current_ms - start_ms) / (end_ms - start_ms) * 100
            print(f"  {total:,} candles fetched... ({pct:.1f}%)")

        time.sleep(SLEEP_MS)

    print(f"  Done — {total:,} candles total.")
    return all_candles


# ── PARSE TO DATAFRAME ────────────────────────────────────────────────────
def parse_klines(raw_candles):
    """
    Parse raw Binance kline response into a clean DataFrame.

    Binance kline format:
    [open_time, open, high, low, close, volume, close_time,
     quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
    """
    df = pd.DataFrame(raw_candles, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    # Keep only what we need
    df = df[["open_time", "open", "high", "low", "close", "volume", "taker_buy_base"]].copy()

    # Types
    df["datetime"]       = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["datetime"]       = df["datetime"].dt.tz_localize(None)  # remove tz for CSV
    df["date"]           = df["datetime"].dt.date
    df["open"]           = df["open"].astype(float)
    df["high"]           = df["high"].astype(float)
    df["low"]            = df["low"].astype(float)
    df["close"]          = df["close"].astype(float)
    df["volume"]         = df["volume"].astype(float)
    df["buy_volume"]     = df["taker_buy_base"].astype(float)
    df["sell_volume"]    = df["volume"] - df["buy_volume"]

    df = df[["datetime", "date", "open", "high", "low", "close",
             "volume", "buy_volume", "sell_volume"]]
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


# ── BUILD 15-MIN CANDLES ──────────────────────────────────────────────────
def build_15min(df_1m):
    """
    Aggregate 1-min candles into 15-min candles.
    Used by backtest.py for the intraday POC touch detection.
    """
    df = df_1m.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")

    df_15m = df.resample("15min").agg(
        open        = ("open",       "first"),
        high        = ("high",       "max"),
        low         = ("low",        "min"),
        close       = ("close",      "last"),
        volume      = ("volume",     "sum"),
        buy_volume  = ("buy_volume", "sum"),
        sell_volume = ("sell_volume","sum"),
    ).dropna().reset_index()

    df_15m["date"] = df_15m["datetime"].dt.date
    return df_15m


# ── MAIN ──────────────────────────────────────────────────────────────────
def run_api_loader():
    start_ms = to_ms(START_DATE)
    end_ms   = to_ms(END_DATE)

    # Fetch raw 1-min candles
    raw = fetch_klines(SYMBOL, INTERVAL, start_ms, end_ms)

    # Parse
    print("\nParsing candles...")
    df_1m = parse_klines(raw)
    print(f"  {len(df_1m):,} candles parsed")
    print(f"  Date range: {df_1m['date'].min()} → {df_1m['date'].max()}")
    print(f"  Unique days: {df_1m['date'].nunique()}")

    # Build 15-min
    print("\nBuilding 15-min candles...")
    df_15m = build_15min(df_1m)
    print(f"  {len(df_15m):,} 15-min candles")

    # Save
    print("\nSaving...")
    df_1m.to_csv(OUTPUT_1M,  index=False)
    df_15m.to_csv(OUTPUT_15M, index=False)
    print(f"  ✓ {OUTPUT_1M}")
    print(f"  ✓ {OUTPUT_15M}")
    print(f"\nAPI loader complete!")

    return df_1m, df_15m


if __name__ == "__main__":
    run_api_loader()
#!/usr/bin/env python3
"""
Build/update z_eoddata.parquet -- a rolling 47-trading-day window of raw
daily OHLCV (NYSE + NASDAQ combined), sourced from EOD/*.csv (free source).

This is the daily-timeframe companion to 2_append_weekly.py. It always
rebuilds from whatever CSVs are on disk (EOD/ is never pruned), but only
reads the last SCAN_LOOKBACK_DAYS worth of files so the read stays cheap
even as the EOD/ archive grows indefinitely.

Flow:
1. Scan EOD/ for CSVs newer than SCAN_LOOKBACK_DAYS ago
2. Combine NYSE + NASDAQ, standardize to T/date/o/h/l/c/v
3. Keep only the most recent ROLLING_DAYS unique trading dates
4. Overwrite z_eoddata.parquet
"""

import os

import pandas as pd

from _common import load_eod_csvs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EOD_DIR = os.path.join(BASE_DIR, "EOD")
OUTPUT_PARQUET = os.path.join(BASE_DIR, "z_eoddata.parquet")

ROLLING_DAYS = 47
SCAN_LOOKBACK_DAYS = 90

print("=" * 70)
print("BUILD z_eoddata.parquet (rolling daily EOD)")
print("=" * 70)

scan_min_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=SCAN_LOOKBACK_DAYS)
print(f"\n1. Scanning {EOD_DIR} for CSVs newer than {scan_min_date.date()}...")
df = load_eod_csvs(EOD_DIR, min_date=scan_min_date)

if df.empty:
    print("   No EOD CSVs found. Nothing to do.")
    raise SystemExit(0)

print(f"   Loaded {len(df):,} rows across {df['T'].nunique():,} tickers")

unique_dates = sorted(df["date"].unique())
print(
    f"\n2. {len(unique_dates)} unique trading date(s) available: "
    f"{pd.Timestamp(unique_dates[0]).date()} to {pd.Timestamp(unique_dates[-1]).date()}"
)

keep_dates = set(unique_dates[-ROLLING_DAYS:])
df = df[df["date"].isin(keep_dates)].sort_values(["T", "date"]).reset_index(drop=True)
print(f"   Trimmed to most recent {len(keep_dates)} date(s)")

df.to_parquet(OUTPUT_PARQUET, compression="snappy", index=False)
size_mb = os.path.getsize(OUTPUT_PARQUET) / (1024 * 1024)

print(f"\n3. Saved {OUTPUT_PARQUET}")
print(f"   Rows: {len(df):,}  Tickers: {df['T'].nunique():,}  Size: {size_mb:.2f} MB")
print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print("\n" + "=" * 70)

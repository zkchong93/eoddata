#!/usr/bin/env python3
"""
Build/update z_eowdata.parquet -- a rolling 47-week window of weekly OHLCV
(NYSE + NASDAQ combined), resampled from daily bars. Mirrors the 47-day
window used by z_eoddata.parquet, just at weekly granularity, so the file
stays well under GitHub's per-file size limit.

Two modes, chosen automatically:

BOOTSTRAP (z_eowdata.parquet does not exist yet):
    Seeds history from the old paid-source z_eod.parquet (WINDOW_START to
    just before NEW_SOURCE_START), then splices in the new free-source
    EOD/*.csv files from NEW_SOURCE_START onward. z_eod.parquet is only
    needed for this one run -- once z_eowdata.parquet exists, delete the
    big seed file yourself; it is never read again.

INCREMENTAL (z_eowdata.parquet already exists):
    Drops the most recent week from the existing file (it may have been a
    partial week last run) and recomputes it -- plus any newly-completed
    week(s) -- from EOD/*.csv alone. No seed file needed.

Either way the result is trimmed to the most recent ROLLING_WEEKS weeks.
"""

import os

import pandas as pd

from _common import load_eod_csvs, to_weekly

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EOD_DIR = os.path.join(BASE_DIR, "EOD")
SEED_PARQUET = os.path.join(BASE_DIR, "z_eod.parquet")
OUTPUT_PARQUET = os.path.join(BASE_DIR, "z_eowdata.parquet")

NEW_SOURCE_START = pd.Timestamp("2026-07-06")  # first date trusted from the free source
WINDOW_START = pd.Timestamp("2025-10-13")      # bootstrap seed lower bound (~47 weeks back)
ROLLING_WEEKS = 47

print("=" * 70)
print("BUILD z_eowdata.parquet (rolling weekly EOD)")
print("=" * 70)

if not os.path.exists(OUTPUT_PARQUET):
    print(f"\n1. No existing {OUTPUT_PARQUET} -- BOOTSTRAP build.")

    if not os.path.exists(SEED_PARQUET):
        raise SystemExit(f"   Seed file not found: {SEED_PARQUET}. Cannot bootstrap.")

    seed = pd.read_parquet(SEED_PARQUET)
    seed["date"] = pd.to_datetime(seed["date"])
    seed = seed[["T", "date", "o", "h", "l", "c", "v"]]
    seed = seed[(seed["date"] >= WINDOW_START) & (seed["date"] < NEW_SOURCE_START)]
    print(f"   Seed rows [{WINDOW_START.date()}, {NEW_SOURCE_START.date()}): {len(seed):,}")

    new_daily = load_eod_csvs(EOD_DIR, min_date=NEW_SOURCE_START)
    print(f"   New-source rows (>= {NEW_SOURCE_START.date()}): {len(new_daily):,}")

    combined = pd.concat([seed, new_daily], ignore_index=True)
    combined = combined.drop_duplicates(subset=["T", "date"], keep="last")

    weekly_full = to_weekly(combined)

else:
    print(f"\n1. Existing {OUTPUT_PARQUET} found -- INCREMENTAL update.")

    existing = pd.read_parquet(OUTPUT_PARQUET)
    existing["date"] = pd.to_datetime(existing["date"])
    existing["week"] = existing["date"].dt.to_period("W-FRI")

    cutoff_week = existing["week"].max()
    cutoff_start = cutoff_week.start_time
    print(f"   Recomputing week of {cutoff_start.date()} onward")

    base = existing[existing["week"] != cutoff_week].drop(columns=["week"])

    fresh_daily = load_eod_csvs(EOD_DIR, min_date=cutoff_start)
    if fresh_daily.empty:
        print("   No daily data available for that window. Nothing to update.")
        raise SystemExit(0)

    fresh_weekly = to_weekly(fresh_daily).drop(columns=["week"])
    weekly_full = pd.concat([base, fresh_weekly], ignore_index=True)

print("\n2. Trimming to rolling window...")
weekly_full["week"] = weekly_full["date"].dt.to_period("W-FRI")
weekly_full = weekly_full.drop_duplicates(subset=["T", "week"], keep="last")

weeks_sorted = sorted(weekly_full["week"].unique())
keep_weeks = set(weeks_sorted[-ROLLING_WEEKS:])
weekly_full = weekly_full[weekly_full["week"].isin(keep_weeks)]
weekly_full = (
    weekly_full.drop(columns=["week"])
    .sort_values(["T", "date"])
    .reset_index(drop=True)
)
print(f"   {len(weeks_sorted)} week(s) seen -> kept most recent {len(keep_weeks)}")

weekly_full.to_parquet(OUTPUT_PARQUET, compression="snappy", index=False)
size_mb = os.path.getsize(OUTPUT_PARQUET) / (1024 * 1024)

print(f"\n3. Saved {OUTPUT_PARQUET}")
print(
    f"   Rows: {len(weekly_full):,}  Tickers: {weekly_full['T'].nunique():,}  "
    f"Size: {size_mb:.2f} MB"
)
print(f"   Week range: {weekly_full['date'].min().date()} to {weekly_full['date'].max().date()}")
print("\n" + "=" * 70)

#!/usr/bin/env python3
"""
Build etp_exclusions.csv -- the list of symbols 3_screen.py must not screen,
because they are exchange-traded products rather than operating companies.

WHY THIS EXISTS
The EODData CSVs in EOD/ carry only Symbol,Date,OHLCV. There is no instrument
type anywhere in the pipeline's own data, so an ETF like PDBC or a leveraged
single-stock product like AMDL looks exactly like a common stock to the screen
and passes it on momentum that means something quite different.

SOURCE
Nasdaq's public consolidated symbol directory:
    https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt
Pipe-delimited, no API key, regenerated every trading morning, and it covers
NYSE + NYSE American + Cboe + Nasdaq (Listing Exchange N/A/P/Z/Q), not just
Nasdaq -- so it spans the same universe the EOD/ files do. The columns used
here are `Symbol`, `Security Name` and `ETF`.

WHAT COUNTS AS AN ETP
    ETF = Y                 the directory's own fund flag. Catches plain ETFs
                            (SMH, SOXX, ICLN), commodity/strategy funds (PDBC)
                            and the leveraged single-stock products (AMDL,
                            AMZD, TSLL, METU, ...), which are all structured as
                            ETFs and flagged accordingly.
    ETN by name             exchange-traded NOTES are debt, not funds, and the
                            directory flags them ETF=N. They are caught off
                            `Security Name` instead. The pattern is deliberately
                            narrow -- it must not sweep in corporate baby bonds
                            ("6.25% Notes due 2054"), preferreds, depositary
                            shares or corporate units, which are out of scope.

SNAPSHOT, NOT A LIVE LOOKUP
The result is committed to the repo and 3_screen.py reads only the committed
file. That keeps the screen reproducible (a rerun of an old commit screens the
same universe it did at the time) and means a nasdaqtrader.com outage cannot
silently change the results: if the fetch fails, this script leaves the last
good etp_exclusions.csv in place and exits 0, and the pipeline carries on with
yesterday's list.

Symbol form note: EODData writes preferreds/units with a dash (NEE-T) where
Nasdaq writes a dollar sign (NEE$T). No ETP uses either suffix, so it does not
affect this file -- 3_screen.py handles the mapping on its side when it joins.

Usage:
    python 0b_fetch_symbol_types.py
"""

import os
import re
import sys
import urllib.error
import urllib.request

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(BASE_DIR, "etp_exclusions.csv")

SOURCE_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
TIMEOUT_SEC = 60

# Narrow on purpose -- see "WHAT COUNTS AS AN ETP" above. "\bETN\b" catches the
# common "... ETN" / "... ETNs due 2032" naming; the other two catch issuers who
# spell it out. Nothing here matches a plain corporate note or a preferred.
ETN_NAME_RE = re.compile(
    r"\bETNs?\b|Exchange[- ]Traded Notes?\b|(?:Index|Equity|Currency)[- ]Linked Notes?\b",
    re.IGNORECASE,
)

print("=" * 70)
print("BUILD etp_exclusions.csv (instrument-type exclusion list)")
print("=" * 70)

print(f"\n1. Fetching {SOURCE_URL}")
try:
    req = urllib.request.Request(
        SOURCE_URL, headers={"User-Agent": "eoddata-pipeline/1.0"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
except (urllib.error.URLError, OSError, TimeoutError) as exc:
    # Never fail the pipeline over this. The committed snapshot is still valid;
    # it is just a day stale, and a stale ETP list is far better than either a
    # broken run or an unfiltered one.
    print(f"   FETCH FAILED: {exc}")
    if os.path.exists(OUTPUT_CSV):
        print(f"   Keeping existing {os.path.basename(OUTPUT_CSV)} (stale but usable).")
        raise SystemExit(0)
    print(f"   No existing {os.path.basename(OUTPUT_CSV)} either -- 3_screen.py will")
    print("   run WITHOUT an instrument-type filter and say so loudly.")
    raise SystemExit(0)

lines = raw.splitlines()
# The file ends with a "File Creation Time: ..." trailer that is not a symbol.
creation = next((ln for ln in reversed(lines) if ln.startswith("File Creation")), "")
data_lines = [ln for ln in lines if not ln.startswith("File Creation")]
print(f"   {len(raw):,} bytes   {creation.split('|')[0] if creation else '(no trailer)'}")

from io import StringIO  # noqa: E402  (only needed once the fetch succeeded)

df = pd.read_csv(StringIO("\n".join(data_lines)), sep="|", dtype=str)
df = df[df["Symbol"].notna()]
print(f"\n2. Parsed {len(df):,} symbols")

name = df["Security Name"].fillna("")
is_etf = df["ETF"].fillna("N").str.upper().eq("Y")
is_etn = name.str.contains(ETN_NAME_RE)

df["Reason"] = ""
df.loc[is_etn, "Reason"] = "etn_name"
df.loc[is_etf, "Reason"] = "etf_flag"  # flag wins if a name somehow matches both

etp = df[is_etf | is_etn].copy()
print(f"   ETF flag = Y : {int(is_etf.sum()):,}")
print(f"   ETN by name  : {int(is_etn.sum()):,}  (+{int((is_etn & ~is_etf).sum()):,} not already flagged ETF)")
print(f"   -> {len(etp):,} excluded symbols")

out = (
    etp[["Symbol", "Reason", "Security Name"]]
    .rename(columns={"Symbol": "Ticker", "Security Name": "Security_Name"})
    .sort_values("Ticker")
)
out.to_csv(OUTPUT_CSV, index=False)
size_kb = os.path.getsize(OUTPUT_CSV) / 1024

print(f"\n3. Saved {OUTPUT_CSV}")
print(f"   Rows: {len(out):,}  Size: {size_kb:.1f} KB")
print("\n" + "=" * 70)

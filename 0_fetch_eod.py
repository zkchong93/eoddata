#!/usr/bin/env python3
"""
Fetch daily EOD bars from the EODData REST API into EOD/{EXCHANGE}_{YYYYMMDD}.csv.

This replaces the manual "download the daily CSV from eoddata.com" step. It
writes byte-compatible files in the exact format the site serves --
`Symbol,Date,Open,High,Low,Close,Volume` with Date as `09-Sep-2026` -- so
1_append_daily.py / 2_append_weekly.py / _common.py need no changes at all.

Endpoint:
    GET /Quote/List/{exchangeCode}?ApiKey=...&DateStamp=yyyy-MM-dd
    -> one row per symbol that traded on that date.

NOTE: do NOT use /Symbol/List/{exchange} for this. That endpoint returns each
symbol's *last known* bar, so `dateStamp` varies per row (a thinly-traded name
can be weeks stale) -- it cannot produce a clean single-day snapshot.

The API key is read from the EODDATA_API_KEY environment variable, falling back
to a local `eoddata_api_key.txt` (gitignored). It is deliberately never
hardcoded: this directory is a public git repo.

Usage:
    python 0_fetch_eod.py                      # fetch whatever is missing in the last 7 weekdays
    python 0_fetch_eod.py --days 10            # re-check the last 10 calendar days
    python 0_fetch_eod.py --date 2026-09-09    # one specific date
    python 0_fetch_eod.py --from 2026-08-01 --to 2026-08-31
    python 0_fetch_eod.py --force              # re-download even if the CSV exists
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EOD_DIR = os.path.join(BASE_DIR, "EOD")
KEY_FILE = os.path.join(BASE_DIR, "eoddata_api_key.txt")

API_ROOT = "https://api.eoddata.com"
EXCHANGES = ["NYSE", "NASDAQ"]

# The API rate-limits aggressively and exposes no quota headers, so calls are
# spaced out and 429s are retried with exponential backoff.
MIN_INTERVAL_SEC = 20
MAX_ATTEMPTS = 5
BACKOFF_START_SEC = 30

# Default mode re-checks this many weekdays back and fetches any (exchange,
# date) file that is missing. Scanning a window rather than just "newest file
# + 1" makes the job self-healing: if one exchange fails on a given day, the
# next run notices that file is still absent and retries it, instead of
# advancing past the hole and leaving it there permanently.
DEFAULT_LOOKBACK_WEEKDAYS = 7

FNAME_RE = re.compile(r"^(NYSE|NASDAQ)_(\d{8})\.csv$")

_last_request_at = 0.0


class FetchError(Exception):
    """Unrecoverable failure for one exchange/date. Recorded and reported at
    the end rather than aborting -- so files already fetched this run survive."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_api_key():
    key = os.environ.get("EODDATA_API_KEY", "").strip()
    if key:
        return key
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE) as f:
            key = f.read().strip()
        if key:
            return key
    sys.exit(
        "ERROR: no API key found.\n"
        "  Set the EODDATA_API_KEY environment variable, or put the key in\n"
        f"  {KEY_FILE}\n"
        "  (In GitHub Actions this comes from the EODDATA_API_KEY repo secret.)"
    )


def fetch_quotes(exchange, day, api_key):
    """Return the list of quote dicts for one exchange/date, or [] if the API
    has nothing for that day (weekend, holiday, not yet published)."""
    global _last_request_at

    qs = urllib.parse.urlencode({"ApiKey": api_key, "DateStamp": day.isoformat()})
    url = f"{API_ROOT}/Quote/List/{exchange}?{qs}"
    backoff = BACKOFF_START_SEC

    for attempt in range(1, MAX_ATTEMPTS + 1):
        wait = MIN_INTERVAL_SEC - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)

        try:
            _last_request_at = time.monotonic()
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))

        except urllib.error.HTTPError as e:
            # 404 = the API simply has nothing for that date (weekend, market
            # holiday, or today's file not published yet). That is a normal,
            # expected outcome -- never a failure, or every holiday would
            # break the scheduled run.
            if e.code == 404:
                return []

            # 429 = rate limited, 5xx = transient server trouble. Both retry.
            if e.code == 429 or 500 <= e.code < 600:
                if attempt == MAX_ATTEMPTS:
                    raise FetchError(
                        f"HTTP {e.code} after {MAX_ATTEMPTS} attempts (rate limited)"
                    )
                print(
                    f"\n      HTTP {e.code} -- retry {attempt}/{MAX_ATTEMPTS - 1} "
                    f"in {backoff}s ... ",
                    end="",
                    flush=True,
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            raise FetchError(f"HTTP {e.code} {e.reason}")

        except urllib.error.URLError as e:
            if attempt == MAX_ATTEMPTS:
                raise FetchError(f"network error: {e.reason}")
            print(f"\n      network error ({e.reason}) -- retry in {backoff}s ... ",
                  end="", flush=True)
            time.sleep(backoff)
            backoff *= 2

    return []


def fmt_price(x):
    """Match the site's CSV style: fixed 4dp with trailing zeros stripped
    (145.2300 -> 145.23, 0.0060 -> 0.006)."""
    if x is None:
        return ""
    s = f"{float(x):.4f}".rstrip("0").rstrip(".")
    return s or "0"


def write_csv(path, rows, day):
    """Write rows in the exact column order/format of the manually-downloaded
    files, sorted by symbol the way the site serves them."""
    date_str = day.strftime("%d-%b-%Y")
    rows = sorted(rows, key=lambda r: (r.get("symbolCode") or ""))

    lines = ["Symbol,Date,Open,High,Low,Close,Volume"]
    for r in rows:
        lines.append(
            ",".join(
                [
                    str(r.get("symbolCode") or ""),
                    date_str,
                    fmt_price(r.get("open")),
                    fmt_price(r.get("high")),
                    fmt_price(r.get("low")),
                    fmt_price(r.get("close")),
                    str(int(r.get("volume") or 0)),
                ]
            )
        )

    # newline="" + explicit \n keeps CRLF out of the file on Windows, so a
    # locally-fetched file is byte-identical to one fetched by the CI runner.
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def latest_csv_date():
    """Newest date already present in EOD/, or None if the archive is empty."""
    dates = []
    if os.path.isdir(EOD_DIR):
        for fname in os.listdir(EOD_DIR):
            m = FNAME_RE.match(fname)
            if m:
                dates.append(datetime.strptime(m.group(2), "%Y%m%d").date())
    return max(dates) if dates else None


def last_weekdays(n, end=None):
    """The n most recent weekdays up to and including `end` (default today)."""
    cur = end or date.today()
    days = []
    while len(days) < n:
        if cur.weekday() < 5:
            days.append(cur)
        cur -= timedelta(days=1)
    return sorted(days)


def weekdays(start, end):
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", help="fetch a single date (yyyy-mm-dd)")
    p.add_argument("--from", dest="from_date", help="range start (yyyy-mm-dd)")
    p.add_argument("--to", dest="to_date", help="range end (yyyy-mm-dd)")
    p.add_argument("--days", type=int, help="re-check the last N calendar days")
    p.add_argument("--force", action="store_true", help="re-download dates whose CSV already exists")
    return p.parse_args()


def resolve_dates(args):
    today = date.today()

    if args.date:
        return [datetime.strptime(args.date, "%Y-%m-%d").date()]

    if args.from_date or args.to_date:
        start = datetime.strptime(args.from_date, "%Y-%m-%d").date() if args.from_date else today - timedelta(days=7)
        end = datetime.strptime(args.to_date, "%Y-%m-%d").date() if args.to_date else today
        return weekdays(start, end)

    if args.days:
        return weekdays(today - timedelta(days=args.days), today)

    # Default: re-check a short trailing window and fetch whatever is missing.
    # Dates whose CSV already exists are skipped without an API call, so the
    # steady-state cost is just the new day -- but a previously-failed
    # exchange/date inside the window is automatically retried.
    return last_weekdays(DEFAULT_LOOKBACK_WEEKDAYS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    api_key = get_api_key()
    os.makedirs(EOD_DIR, exist_ok=True)

    print("=" * 70)
    print("FETCH EOD CSVs from EODData API")
    print("=" * 70)

    newest = latest_csv_date()
    print(f"\n1. Newest CSV on disk: {newest if newest else '(none)'}")

    targets = resolve_dates(args)
    if not targets:
        print("\n   Nothing to fetch -- already up to date.")
        print("\n" + "=" * 70)
        return

    print(f"\n2. {len(targets)} candidate trading day(s): {targets[0]} to {targets[-1]}")
    print("-" * 70)

    written, skipped, empty = 0, 0, 0
    failures = []

    for day in targets:
        compact = day.strftime("%Y%m%d")

        for exchange in EXCHANGES:
            path = os.path.join(EOD_DIR, f"{exchange}_{compact}.csv")

            if os.path.exists(path) and not args.force:
                skipped += 1
                continue

            print(f"   {day}  {exchange:<6} fetching ... ", end="", flush=True)

            try:
                quotes = fetch_quotes(exchange, day, api_key)
            except FetchError as e:
                print(f"FAILED -- {e}")
                failures.append((day, exchange, str(e)))
                continue

            # Defensive: only keep rows actually stamped with the requested
            # date, so a future API change can never smuggle in stale bars.
            rows = [q for q in quotes if (q.get("dateStamp") or "")[:10] == day.isoformat()]

            if not rows:
                print("no data (weekend / holiday / not yet published)")
                empty += 1
                continue

            write_csv(path, rows, day)
            print(f"OK  {len(rows):,} symbols  ->  EOD/{exchange}_{compact}.csv")
            written += 1

    print("-" * 70)
    print(
        f"\n3. Done. {written} file(s) written, {skipped} already present, "
        f"{empty} with no data, {len(failures)} failed."
    )

    if failures:
        print("\n   FAILURES (will be retried automatically on the next run):")
        for day, exchange, err in failures:
            print(f"     {day} {exchange}: {err}")

    print("\n" + "=" * 70)

    # Non-zero exit surfaces the problem in CI, but only after everything that
    # could be fetched has been written to disk.
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()

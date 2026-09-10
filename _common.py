"""Shared helpers for the eoddata pipeline (free-source EOD CSVs -> parquet)."""

import os
import re

import pandas as pd

FNAME_RE = re.compile(r"^(NYSE|NASDAQ)_(\d{8})\.csv$")

CSV_COLUMN_MAP = {
    "Symbol": "T",
    "Open": "o",
    "High": "h",
    "Low": "l",
    "Close": "c",
    "Volume": "v",
}


def load_eod_csvs(eod_dir, min_date=None):
    """Read every EOD/{NYSE,NASDAQ}_YYYYMMDD.csv file whose filename date is
    >= min_date (if given), combine NYSE + NASDAQ, and return one
    T/date/o/h/l/c/v dataframe deduplicated on (T, date)."""
    frames = []
    for fname in sorted(os.listdir(eod_dir)):
        m = FNAME_RE.match(fname)
        if not m:
            continue
        file_date = pd.to_datetime(m.group(2), format="%Y%m%d")
        if min_date is not None and file_date < min_date:
            continue
        # keep_default_na=False: the ticker "NA" is a real NASDAQ symbol, and
        # pandas' default NA strings would silently turn it into NaN (one
        # nulled row per file). na_values=[""] keeps genuinely blank numeric
        # cells parsing as NaN.
        df = pd.read_csv(
            os.path.join(eod_dir, fname), keep_default_na=False, na_values=[""]
        )
        df = df.rename(columns=CSV_COLUMN_MAP)
        df["date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y")
        frames.append(df[["T", "date", "o", "h", "l", "c", "v"]])

    if not frames:
        return pd.DataFrame(columns=["T", "date", "o", "h", "l", "c", "v"])

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["T", "date"], keep="last")
    return out


def to_weekly(df):
    """Resample a T/date/o/h/l/c/v daily dataframe into weekly (Mon-Fri)
    OHLCV bars keyed by (T, week). `date` on the result is the actual last
    trading date seen in that week (not necessarily a Friday, e.g. holidays).
    Keeps a `week` (pandas Period, W-FRI) column for downstream trimming."""
    d = df.sort_values(["T", "date"]).copy()
    d["week"] = d["date"].dt.to_period("W-FRI")
    weekly = (
        d.groupby(["T", "week"], sort=False)
        .agg(
            date=("date", "max"),
            o=("o", "first"),
            h=("h", "max"),
            l=("l", "min"),
            c=("c", "last"),
            v=("v", "sum"),
        )
        .reset_index()
    )
    return weekly

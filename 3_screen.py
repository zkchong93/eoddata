#!/usr/bin/env python3
"""
Compute indicators and screen z_eoddata.parquet (daily) and z_eowdata.parquet
(weekly) independently, writing result_daily.csv / result_weekly.csv.

Indicators (per ticker, latest bar):
    SMA4, SMA12, SMA46          simple moving average of close
    RSI6, RSI12, RSI24          Wilder-smoothed RSI (first avg = simple mean
                                 of the first `period` gains/losses, then
                                 Wilder smoothing thereafter)
    RSI_sync_days               how many consecutive bars back from the latest
                                 one the RSI6 > RSI12 > RSI24 stack has held.
                                 0 = not stacked on the latest bar. Bars where
                                 RSI24 has not seeded yet count as "not held",
                                 so the reading is capped by the window.
    Gradient_SMA4/46            1-bar LOG return of the SMA itself, in %:
                                 ln(SMA_t / SMA_t-1) * 100
                                 (log, not simple %, so a move is comparable
                                 in magnitude regardless of price level)
    Deviation_SMA4/46           abs % distance of the bar's extreme from the
                                 SMA, in the direction of the current trend:
                                 close > SMA  -> abs((high - SMA) / SMA) * 100
                                 close <= SMA -> abs((low  - SMA) / SMA) * 100
    Bars_since_SMA4             bars since the last bar that straddled SMA4
                                 (low <= SMA4 <= high). 0 = the latest bar is
                                 touching it. NaN = no touch anywhere in the
                                 available window.
    ATR10, ATR10_pct            Wilder ATR over 10 periods (NOT the usual 14),
                                 seeded the same way as the RSI above (first
                                 ATR = simple mean of the first 10 true
                                 ranges). ATR10_pct = ATR10 / close * 100.
    RVOL10_max_10                max relative-volume reading over the last
                                 RVOL_LOOKBACK periods, where each period's
                                 RVOL = volume_t / mean(volume of the
                                 preceding RVOL_PERIOD periods). Needs
                                 RVOL_PERIOD + RVOL_LOOKBACK periods of
                                 history (10 + 10 = 20) to evaluate.
    CLV                         close location value of the latest bar:
                                 ((C - L) - (H - C)) / (H - L). +1 = closed on
                                 the high, -1 = closed on the low. NaN when
                                 H == L (no range to locate the close in).
    CLV_rvol                    CLV * the latest bar's RVOL -- close location
                                 weighted by how much volume showed up for it.
    Star_candle                 latest bar opened below all three SMAs and
                                 closed above all three, on RVOL > RVOL_MIN.
    Star_days_ago               bars since the most recent star candle in the
                                 window. NOTE: a star candle needs SMA46, so
                                 it can only be evaluated on the last
                                 (bars - 45) bars -- with the 47-bar rolling
                                 parquet that is a 2-bar search depth, so this
                                 is effectively {0, 1, NaN}. See MIN_BARS.
    WMA5, WMA10, WMA20          linearly weighted moving average of close
                                 (weights 1..n, newest bar weighted n).
    Gradient_WMA20              1-bar LOG return of WMA20, in %, same
                                 convention as Gradient_SMA4/46.
    WMA_sync                    WMA5 > WMA10 > WMA20 on the latest bar.
    WMA_fan_spread              (WMA5 - WMA20) / WMA20 * 100 -- how far the
                                 fast/slow WMAs have fanned apart, in %.
    VWAP_window                 volume-weighted average price over the
                                 entire available rolling window for that
                                 ticker, using typical price (H+L+C)/3 as the
                                 price input. Named "_window" because the
                                 lookback is NOT fixed: it is however many
                                 bars the parquet currently holds for that
                                 ticker -- at most ROLLING_DAYS = 47 bars
                                 (daily) / ROLLING_WEEKS = 47 weeks (weekly),
                                 and fewer for a recently-listed name. It
                                 rolls forward every run, so it is not an
                                 anchored VWAP and two runs are not directly
                                 comparable.

Screen (all conditions concurrent; volume/dvol thresholds differ daily vs weekly):
    close  >= 5                          no penny stocks
    volume >  2,000,000 (daily) / 8,000,000 (weekly)
    dvol   > 20,000,000 (daily) / 50,000,000 (weekly)   (close * volume)
    RSI6 > RSI12 > RSI24
    SMA4 > SMA12
    RVOL10 > 1.3 at least once in the last 10 periods

Instrument-type filter (applied AFTER the screen, deliberately kept separate):
    ETFs, ETNs and leveraged single-stock products are dropped, per
    etp_exclusions.csv as built by 0b_fetch_symbol_types.py. These are not
    operating companies -- momentum on a 2x single-stock ETF is a restatement
    of its underlying's momentum, and a broad ETF's is a restatement of the
    index -- so they crowd the list without adding a name to look at.
    It runs as its own step so the screen's own pass count stays visible and
    comparable across runs: a change there means a screen condition moved,
    whereas a change here only means the exclusion list moved.

Output: result_daily.csv / result_weekly.csv, overwritten each run. Every
indicator computed above is written out; the previous run's file is read first
so the run can print a before/after pass count and name the tickers that
entered or left. The screen pass count should only move for real market
reasons -- if it jumps on a run that changed this file, a screen condition was
altered by accident.
"""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RVOL_PERIOD = 10    # periods averaged to get "typical" volume
RVOL_LOOKBACK = 10  # how many recent periods to check for RVOL > RVOL_MIN
RVOL_MIN = 1.3

ATR_PERIOD = 10

# Needed history: SMA46 *and* SMA46 one bar back (for gradient) = 47;
# RVOL10 over the last 10 periods needs RVOL_PERIOD + RVOL_LOOKBACK = 20.
# Everything added since fits inside that 47: ATR10 needs 11 bars, RSI24
# needs 25, WMA20 + its 1-bar gradient needs 21, and a star candle needs 46
# (SMA46 on the bar itself). So MIN_BARS stays at 47 -- raising it would
# shrink the universe for no gain. The one thing 47 does squeeze is
# Star_days_ago's search depth (47 - 46 + 1 = 2 bars); see its note above.
MIN_BARS = max(47, RVOL_PERIOD + RVOL_LOOKBACK)


def calculate_rsi(close, period):
    """Wilder RSI, matching NYSE/2_filter_indicator.py's convention.

    Returns the FULL rsi array, not just the latest reading -- callers that
    only want the latest bar take [-1] themselves, and callers that need to
    walk backwards (RSI_sync_days) get the series. The array is one shorter
    than `close` because it is indexed off the diff: rsi[i] belongs to
    close[i + 1]. All three periods therefore share one index space, which is
    what lets RSI_sync_days compare them element-wise.

    Bars before the seed point (index period - 1) are NaN rather than the 0
    they used to compute out as -- RSI genuinely is not defined there, and
    leaving zeros in would let a backwards scan walk into warm-up garbage.
    Seeded and smoothed values are untouched.
    """
    if len(close) <= period:
        return np.full(max(len(close) - 1, 0), np.nan)

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros(len(gains))
    avg_loss = np.zeros(len(losses))
    avg_gain[period - 1] = gains[:period].mean()
    avg_loss[period - 1] = losses[:period].mean()

    for i in range(period, len(gains)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

    # Blank the warm-up *after* the recursion above has run off it.
    avg_gain[: period - 1] = np.nan
    avg_loss[: period - 1] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(high, low, close, period):
    """Wilder ATR. Same seeding convention as calculate_rsi(): the first ATR
    is the simple mean of the first `period` true ranges, Wilder smoothing
    after that.

    True range needs a previous close, so the array is indexed exactly like
    calculate_rsi()'s: atr[i] belongs to close[i + 1], length len(close) - 1.
    Needs period + 1 bars to produce anything.
    """
    if len(close) <= period:
        return np.full(max(len(close) - 1, 0), np.nan)

    prev_close = close[:-1]
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)),
    )

    atr = np.full(len(tr), np.nan)
    atr[period - 1] = tr[:period].mean()
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def calculate_wma(values, period):
    """Linearly weighted moving average, weights 1..period with the newest
    bar weighted `period`. Returned aligned to `values` (NaN for the first
    period - 1 bars) so it can be indexed alongside close/high/low."""
    out = np.full(len(values), np.nan)
    if len(values) < period:
        return out

    weights = np.arange(1, period + 1, dtype=float)
    # np.convolve reverses its second argument, so pass the weights reversed
    # to end up with the newest bar carrying the largest weight.
    out[period - 1:] = np.convolve(values, weights[::-1], mode="valid") / weights.sum()
    return out


def bars_since(flags):
    """Bars between the latest bar and the most recent True in `flags`.
    0 = the latest bar itself. NaN = never True in the available window."""
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return np.nan
    return float(len(flags) - 1 - idx[-1])


def log_gradient(curr, prev):
    if not prev or pd.isna(curr) or pd.isna(prev):
        return np.nan
    return np.log(curr / prev) * 100


def compute_ticker_metrics(df_ticker):
    """Return a dict of latest-bar metrics for one ticker, or None if there
    isn't enough history yet."""
    if len(df_ticker) < MIN_BARS:
        return None

    opens = df_ticker["o"].values
    closes = df_ticker["c"].values
    highs = df_ticker["h"].values
    lows = df_ticker["l"].values
    volumes = df_ticker["v"].values

    sma4_series = pd.Series(closes).rolling(4).mean()
    sma12_series = pd.Series(closes).rolling(12).mean()
    sma46_series = pd.Series(closes).rolling(46).mean()

    sma4, sma4_prev = sma4_series.iloc[-1], sma4_series.iloc[-2]
    sma12 = sma12_series.iloc[-1]
    sma46, sma46_prev = sma46_series.iloc[-1], sma46_series.iloc[-2]

    latest_close = closes[-1]
    latest_high = highs[-1]
    latest_low = lows[-1]
    latest_volume = volumes[-1]

    gradient_sma4 = log_gradient(sma4, sma4_prev)
    gradient_sma46 = log_gradient(sma46, sma46_prev)

    def deviation(sma):
        if pd.isna(sma) or sma == 0:
            return np.nan
        ref = latest_high if latest_close > sma else latest_low
        return abs((ref - sma) / sma) * 100

    # RVOL10: volume(t) / mean(volume of the preceding RVOL_PERIOD periods),
    # take the max reading over the last RVOL_LOOKBACK periods.
    vol_s = pd.Series(volumes)
    avg_vol = vol_s.rolling(RVOL_PERIOD).mean().shift(1)
    rvol = vol_s / avg_vol
    rvol_arr = rvol.values
    rvol10_max_recent = rvol.iloc[-RVOL_LOOKBACK:].max()
    rvol_latest = rvol_arr[-1]

    # RSI series share one index space (rsi[i] <-> close[i + 1]), so the
    # stack can be tested bar by bar. NaN warm-up bars compare False and stop
    # the backwards walk, which is the honest answer -- RSI24 isn't defined
    # there.
    rsi6_series = calculate_rsi(closes, 6)
    rsi12_series = calculate_rsi(closes, 12)
    rsi24_series = calculate_rsi(closes, 24)
    with np.errstate(invalid="ignore"):
        rsi_stacked = (rsi6_series > rsi12_series) & (rsi12_series > rsi24_series)
    # Count consecutive True's back from the latest bar.
    rsi_sync_days = int(np.argmin(rsi_stacked[::-1])) if not rsi_stacked.all() else int(len(rsi_stacked))

    atr_series = calculate_atr(highs, lows, closes, ATR_PERIOD)
    atr10 = atr_series[-1]
    atr10_pct = atr10 / latest_close * 100 if latest_close else np.nan

    # Bars since price last straddled SMA4. NaN SMA4 (warm-up) compares
    # False, so those bars simply never count as a touch.
    sma4_arr = sma4_series.values
    with np.errstate(invalid="ignore"):
        touched_sma4 = (lows <= sma4_arr) & (sma4_arr <= highs)
    bars_since_sma4 = bars_since(touched_sma4)

    # Close location value of the latest bar.
    bar_range = latest_high - latest_low
    clv = (
        ((latest_close - latest_low) - (latest_high - latest_close)) / bar_range
        if bar_range
        else np.nan
    )
    clv_rvol = clv * rvol_latest if not pd.isna(clv) else np.nan

    # Star candle: opened under the whole SMA stack, closed over all of it,
    # on above-average volume. Computed as a series so Star_days_ago can look
    # back -- but SMA46 only exists on the last (bars - 45) bars, so that
    # lookback is 2 bars deep on a 47-bar window.
    sma_min = np.minimum(np.minimum(sma4_arr, sma12_series.values), sma46_series.values)
    sma_max = np.maximum(np.maximum(sma4_arr, sma12_series.values), sma46_series.values)
    with np.errstate(invalid="ignore"):
        star_series = (opens < sma_min) & (closes > sma_max) & (rvol_arr > RVOL_MIN)
    star_days_ago = bars_since(star_series)

    wma5_series = calculate_wma(closes, 5)
    wma10_series = calculate_wma(closes, 10)
    wma20_series = calculate_wma(closes, 20)
    wma5, wma10, wma20 = wma5_series[-1], wma10_series[-1], wma20_series[-1]

    # VWAP over the entire available window (all bars for this ticker),
    # typical price = (H+L+C)/3. The window is whatever the parquet holds --
    # up to 47 bars / 47 weeks, fewer for a new listing -- hence the
    # "_window" suffix on the output column; it is not an anchored VWAP.
    typical = (highs + lows + closes) / 3
    vol_sum = volumes.sum()
    vwap_window = (typical * volumes).sum() / vol_sum if vol_sum else np.nan

    return {
        "T": df_ticker["T"].iloc[0],
        "Close": latest_close,
        "Volume": latest_volume,
        "DVOL": latest_close * latest_volume,
        "SMA4": sma4,
        "SMA12": sma12,
        "SMA46": sma46,
        "Gradient_SMA4": gradient_sma4,
        "Gradient_SMA46": gradient_sma46,
        "Deviation_SMA4": deviation(sma4),
        "Deviation_SMA46": deviation(sma46),
        "Bars_since_SMA4": bars_since_sma4,
        "RSI6": rsi6_series[-1],
        "RSI12": rsi12_series[-1],
        "RSI24": rsi24_series[-1],
        "RSI_sync_days": rsi_sync_days,
        "ATR10": atr10,
        "ATR10_pct": atr10_pct,
        "RVOL10_max_10": rvol10_max_recent,
        "CLV": clv,
        "CLV_rvol": clv_rvol,
        "Star_candle": bool(star_series[-1]),
        "Star_days_ago": star_days_ago,
        "WMA5": wma5,
        "WMA10": wma10,
        "WMA20": wma20,
        "Gradient_WMA20": log_gradient(wma20, wma20_series[-2]),
        "WMA_sync": bool(wma5 > wma10 > wma20),
        "WMA_fan_spread": (wma5 - wma20) / wma20 * 100 if wma20 else np.nan,
        "VWAP_window": vwap_window,
    }


# Written in this order. Ticker/Close/VWAP_window lead so the file still reads
# the way it used to at a glance.
OUTPUT_COLUMNS = [
    "Ticker",
    "Close",
    "VWAP_window",
    "Volume",
    "DVOL",
    "SMA4",
    "SMA12",
    "SMA46",
    "Gradient_SMA4",
    "Gradient_SMA46",
    "Deviation_SMA4",
    "Deviation_SMA46",
    "Bars_since_SMA4",
    "RSI6",
    "RSI12",
    "RSI24",
    "RSI_sync_days",
    "ATR10",
    "ATR10_pct",
    "RVOL10_max_10",
    "CLV",
    "CLV_rvol",
    "Star_candle",
    "Star_days_ago",
    "WMA5",
    "WMA10",
    "WMA20",
    "Gradient_WMA20",
    "WMA_sync",
    "WMA_fan_spread",
]

# Keep the CSV readable and small. VWAP_window stays at 3dp, as it always was.
ROUNDING = {"VWAP_window": 3}
DEFAULT_ROUND = 4


ETP_EXCLUSIONS_CSV = os.path.join(BASE_DIR, "etp_exclusions.csv")


def load_etp_exclusions():
    """Tickers to drop as ETFs/ETNs, from the committed snapshot that
    0b_fetch_symbol_types.py maintains. Returns None (not an empty set) when
    the file is absent, so the caller can tell "nothing to exclude" apart from
    "the filter isn't running" and shout about the latter."""
    if not os.path.exists(ETP_EXCLUSIONS_CSV):
        return None
    try:
        ex = pd.read_csv(ETP_EXCLUSIONS_CSV)
    except Exception as exc:
        print(f"   (could not read {os.path.basename(ETP_EXCLUSIONS_CSV)}: {exc})")
        return None
    if "Ticker" not in ex.columns:
        return None
    return set(ex["Ticker"].astype(str))


def is_etp(ticker, exclusions):
    """EODData writes preferreds/units with a dash (NEE-T) where Nasdaq writes
    a dollar sign (NEE$T), so try that form too. Anything not in the list is
    kept -- an unmatched symbol must never be dropped on a failed join."""
    return ticker in exclusions or ticker.replace("-", "$") in exclusions


def previous_pass_list(output_csv):
    """Tickers in the last run's output, for the before/after print. Returns
    None if there is no previous file (first run, or a fresh clone)."""
    if not os.path.exists(output_csv):
        return None
    try:
        prev = pd.read_csv(output_csv)
    except Exception as exc:  # a truncated/half-written file must not kill the run
        print(f"   (could not read previous {os.path.basename(output_csv)}: {exc})")
        return None
    if "Ticker" not in prev.columns:
        return None
    return sorted(prev["Ticker"].astype(str))


def screen(parquet_path, output_csv, label, vol_min, dvol_min):
    print("=" * 70)
    print(f"SCREEN {label} ({os.path.basename(parquet_path)})")
    print("=" * 70)

    before = previous_pass_list(output_csv)

    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["T", "date"]).reset_index(drop=True)
    print(f"\n1. Loaded {len(df):,} rows, {df['T'].nunique():,} tickers")

    print("\n2. Computing indicators (latest bar per ticker)...")
    results = []
    for ticker, df_ticker in df.groupby("T", sort=False):
        metrics = compute_ticker_metrics(df_ticker)
        if metrics is not None:
            results.append(metrics)

    df_metrics = pd.DataFrame(results)
    print(f"   {len(df_metrics):,} / {df['T'].nunique():,} tickers had enough history")

    print(f"\n3. Applying screen (volume > {vol_min:,}, dvol > {dvol_min:,})...")
    passed = df_metrics[
        (df_metrics["Close"] >= 5)
        & (df_metrics["Volume"] > vol_min)
        & (df_metrics["DVOL"] > dvol_min)
        & (df_metrics["RSI6"] > df_metrics["RSI12"])
        & (df_metrics["RSI12"] > df_metrics["RSI24"])
        & (df_metrics["SMA4"] > df_metrics["SMA12"])
        & (df_metrics["RVOL10_max_10"] > RVOL_MIN)
    ].copy()
    screen_pass_count = len(passed)
    print(f"   {screen_pass_count:,} ticker(s) pass all filters")

    print("\n3b. Instrument-type filter (ETFs / ETNs)...")
    exclusions = load_etp_exclusions()
    if exclusions is None:
        removed = []
        print(f"   !! {os.path.basename(ETP_EXCLUSIONS_CSV)} missing -- FILTER NOT APPLIED.")
        print("   !! Run 0b_fetch_symbol_types.py. ETPs will be in this output.")
    else:
        mask_etp = passed["T"].astype(str).map(lambda t: is_etp(t, exclusions))
        removed = sorted(passed.loc[mask_etp, "T"].astype(str))
        passed = passed[~mask_etp].copy()
        print(f"   {len(exclusions):,} known ETP symbols loaded")
        print(f"   removed {len(removed)}: {', '.join(removed) if removed else '-'}")
        print(f"   {len(passed):,} ticker(s) remain")

    out = passed.rename(columns={"T": "Ticker"}).sort_values("Ticker")
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(ROUNDING.get(col, DEFAULT_ROUND))
    out = out[OUTPUT_COLUMNS]
    out.to_csv(output_csv, index=False)
    print(f"\n4. Saved {output_csv} ({len(out)} rows, {len(OUTPUT_COLUMNS)} columns)")

    # Before/after pass count. The screen conditions are supposed to be frozen,
    # so a swing in "screen conditions" right after this file was edited means
    # a condition moved by accident. The instrument filter is reported on its
    # own line so its effect never hides inside that number.
    after = sorted(out["Ticker"].astype(str))
    print(f"\n5. Pass count {label}:")
    print(f"   screen conditions : {screen_pass_count}")
    print(f"   instrument filter : -{len(removed)} ETP")
    print(f"   written           : {len(after)}")
    if before is None:
        print(f"   vs previous file  : (no previous {os.path.basename(output_csv)})")
    else:
        entered = sorted(set(after) - set(before))
        left = sorted(set(before) - set(after))
        print(
            f"   vs previous file  : {len(before)} -> {len(after)}   "
            f"delta: {len(after) - len(before):+d}"
        )
        print(f"   entered ({len(entered)}): {', '.join(entered) if entered else '-'}")
        print(f"   left    ({len(left)}): {', '.join(left) if left else '-'}")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    screen(
        os.path.join(BASE_DIR, "z_eoddata.parquet"),
        os.path.join(BASE_DIR, "result_daily.csv"),
        "DAILY",
        vol_min=2_000_000,
        dvol_min=20_000_000,
    )
    screen(
        os.path.join(BASE_DIR, "z_eowdata.parquet"),
        os.path.join(BASE_DIR, "result_weekly.csv"),
        "WEEKLY",
        vol_min=8_000_000,
        dvol_min=50_000_000,
    )

#!/usr/bin/env python3
"""
Compute indicators and screen z_eoddata.parquet (daily) and z_eowdata.parquet
(weekly) independently, writing result_daily.csv / result_weekly.csv.

Indicators (per ticker, latest bar):
    SMA4, SMA12, SMA46          simple moving average of close
    RSI6, RSI12, RSI24          Wilder-smoothed RSI (first avg = simple mean
                                 of the first `period` gains/losses, then
                                 Wilder smoothing thereafter)
    Gradient_SMA4/46            1-bar LOG return of the SMA itself, in %:
                                 ln(SMA_t / SMA_t-1) * 100
                                 (log, not simple %, so a move is comparable
                                 in magnitude regardless of price level)
    Deviation_SMA4/46           abs % distance of the bar's extreme from the
                                 SMA, in the direction of the current trend:
                                 close > SMA  -> abs((high - SMA) / SMA) * 100
                                 close <= SMA -> abs((low  - SMA) / SMA) * 100
    RVOL10_max_10                max relative-volume reading over the last
                                 RVOL_LOOKBACK periods, where each period's
                                 RVOL = volume_t / mean(volume of the
                                 preceding RVOL_PERIOD periods). Needs
                                 RVOL_PERIOD + RVOL_LOOKBACK periods of
                                 history (10 + 10 = 20) to evaluate.
    VWAP                        volume-weighted average price over the
                                 entire available rolling window for that
                                 ticker (up to 47 bars), using typical price
                                 (H+L+C)/3 as the price input

Screen (all conditions concurrent; volume/dvol thresholds differ daily vs weekly):
    close  >= 5                          no penny stocks
    volume >  2,000,000 (daily) / 8,000,000 (weekly)
    dvol   > 20,000,000 (daily) / 50,000,000 (weekly)   (close * volume)
    RSI6 > RSI12 > RSI24
    SMA4 > SMA12
    RVOL10 > 1.3 at least once in the last 10 periods

Output: result_daily.csv / result_weekly.csv, overwritten each run.
Columns: Ticker, Close, VWAP (VWAP rounded to 3dp)
"""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RVOL_PERIOD = 10    # periods averaged to get "typical" volume
RVOL_LOOKBACK = 10  # how many recent periods to check for RVOL > RVOL_MIN
RVOL_MIN = 1.3

# Needed history: SMA46 *and* SMA46 one bar back (for gradient) = 47;
# RVOL10 over the last 10 periods needs RVOL_PERIOD + RVOL_LOOKBACK = 20.
MIN_BARS = max(47, RVOL_PERIOD + RVOL_LOOKBACK)


def calculate_rsi(close, period):
    """Wilder RSI, matching NYSE/2_filter_indicator.py's convention."""
    if len(close) <= period:
        return np.nan

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

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi[-1]


def compute_ticker_metrics(df_ticker):
    """Return a dict of latest-bar metrics for one ticker, or None if there
    isn't enough history yet."""
    if len(df_ticker) < MIN_BARS:
        return None

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

    def log_gradient(curr, prev):
        if not prev or pd.isna(curr) or pd.isna(prev):
            return np.nan
        return np.log(curr / prev) * 100

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
    rvol10_max_recent = rvol.iloc[-RVOL_LOOKBACK:].max()

    # VWAP over the entire available window (all bars for this ticker),
    # typical price = (H+L+C)/3.
    typical = (highs + lows + closes) / 3
    vol_sum = volumes.sum()
    vwap = (typical * volumes).sum() / vol_sum if vol_sum else np.nan

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
        "RSI6": calculate_rsi(closes, 6),
        "RSI12": calculate_rsi(closes, 12),
        "RSI24": calculate_rsi(closes, 24),
        "RVOL10_max_10": rvol10_max_recent,
        "VWAP": vwap,
    }


def screen(parquet_path, output_csv, label, vol_min, dvol_min):
    print("=" * 70)
    print(f"SCREEN {label} ({os.path.basename(parquet_path)})")
    print("=" * 70)

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
    print(f"   {len(passed):,} ticker(s) pass all filters")

    out = (
        passed[["T", "Close", "VWAP"]]
        .rename(columns={"T": "Ticker"})
        .sort_values("Ticker")
    )
    out["VWAP"] = out["VWAP"].round(3)
    out.to_csv(output_csv, index=False)
    print(f"\n4. Saved {output_csv} ({len(out)} rows)")
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

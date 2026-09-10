# Anchored VWAP — options and size estimates

Investigation only. Nothing here is implemented. (CHANGE 6.)

**Goal:** VWAP anchored to a named event date — a star-candle date, or a
post-earnings gap date — rather than to "whatever the parquet happens to hold",
which is what `VWAP_window` is today.

---

## Why it is blocked right now

`1_append_daily.py` trims `z_eoddata.parquet` to `ROLLING_DAYS = 47` unique
trading dates; `2_append_weekly.py` trims `z_eowdata.parquet` to
`ROLLING_WEEKS = 47`. So the reachable anchor depth is:

| store | window | calendar reach | resolution |
|---|---|---|---|
| `z_eoddata.parquet` | 47 trading days | ~9.4 weeks | daily |
| `z_eowdata.parquet` | 47 weeks | ~10.8 months | weekly |

A post-earnings anchor is typically 1–2 quarters back. Daily can't reach it.
Weekly can, but a weekly bar's typical price smears the gap day across the
whole week, which defeats the point of anchoring at the gap.

## Measured baseline (2026-09-10)

| item | size |
|---|---|
| `z_eoddata.parquet` (47 d × 9,095 T, 388,199 rows, snappy) | 5.9 MB |
| `z_eowdata.parquet` (47 w × 14,676 T, 539,139 rows, snappy) | 10.6 MB |
| `EOD/` archive, 49 trading days (2026-07-01 → 2026-09-09) | 19.3 MB |
| one trading day of `EOD/` CSVs (NYSE + NASDAQ) | ~390 KB |
| `result_daily.csv` + `result_weekly.csv` after widening | 52 KB |

Git blob cost per pipeline run (gzip proxy, since the workflow rewrites all four
files every run):

| file | raw | ~blob |
|---|---|---|
| `z_eoddata.parquet` | 5.9 MB | 5.3 MB |
| `z_eowdata.parquet` | 10.6 MB | 9.8 MB |
| both result CSVs | 52 KB | 24 KB |

**~15.1 MB of near-incompressible history per run.** At 5 runs/week that is
~3.8 GB/year of pack growth. That, not the widened CSV, is the repo's real size
problem — see "Budget note" at the bottom.

Thin-store cost, measured by actually writing the subset with `zstd` and a
categorical ticker column:

| columns | bytes/row |
|---|---|
| `T, date, c, v` | 5.86 |
| `T, date, h, l, c, v` | 12.25 |

`h, l, c, v` is the row shape that matters: `VWAP_window` uses typical price
`(H+L+C)/3`, so an anchored VWAP has to carry H and L or it won't be
methodologically comparable to the column already in the output.

---

## The finding that changes the framing

**`EOD/` is never pruned.** `1_append_daily.py` only *reads* a trailing
`SCAN_LOOKBACK_DAYS = 90` window; it never deletes, and the workflow commits
`EOD/*.csv` every run. Full-resolution daily OHLCV back to **2026-07-01** is
already sitting in the repo.

So the 9-week ceiling is not a storage decision that needs undoing — it is just
the age of the free-source archive. It deepens by one trading day per day on its
own. Around **2026-11** it reaches one quarter; around **2027-02**, two.

That reframes the question from "how do we store more history" to "when do we
start reading the history we are already storing" — and, separately, "does
`EOD/` itself need a size plan" (~95 MB/year, so yes, eventually).

---

## Options

### A. Read `EOD/*.csv` directly for the anchored leg — no new store

Compute anchored VWAP from the raw CSVs instead of the parquet. `_common.py`'s
`load_eod_csvs(eod_dir, min_date=...)` already does exactly this read.

- **New storage: 0 MB.**
- Anchor reach: back to 2026-07-01 today, +1 trading day/day, no ceiling.
- Cost: parsing ~250 CSVs (~95 MB) per run once a year has accumulated. Slow but
  not prohibitive, and only for the anchor watchlist's tickers.
- Risk: makes `EOD/`'s own unbounded growth load-bearing. It is currently
  19.3 MB and grows ~95 MB/year; it will need pruning or an LFS/release-asset
  plan around year two, and that decision would then break anchoring.

### B. Second thin parquet, anchor watchlist only — *recommended*

`z_anchors.parquet`, columns `T, date, h, l, c, v`, restricted to a maintained
watchlist of names that currently have a live anchor. Measured, then projected:

| watchlist | 47 bars | 1 year (252 bars) | 2 years (504 bars) |
|---|---|---|---|
| 100 tickers | 73 KB | 0.4 MB | 0.8 MB |
| 300 tickers | 196 KB | 1.1 MB | 2.2 MB |

- Sub-1% of current per-run churn even at 300 names × 2 years.
- Repairable: bars are kept, so a revised print or a changed anchor date can be
  recomputed from scratch.
- Needs a watchlist to be maintained, and a name only gets history from the day
  it joins — unless seeded from `EOD/` (option A) at join time, which is easy
  while `EOD/` still covers the anchor.

### C. Second thin parquet, full universe

| columns | 1 year | 2 years | 5 years |
|---|---|---|---|
| `c, v` | 12.2 MB | 24.4 MB | 61.0 MB |
| `h, l, c, v` | 25.5 MB | 51.0 MB | 127.5 MB |

- The useful shape (`h, l, c, v`) crosses GitHub's 50 MB per-file **warning** at
  ~2 years and the 100 MB **hard** limit at ~4.
- Worse, it would add ~24 MB of fresh near-incompressible blob per run at the
  2-year mark, roughly **tripling** the pack growth that is already the problem.
- **Not recommended.** It buys universe-wide anchoring that nothing asked for.

### D. Store accumulators, not bars

Per `(ticker, anchor)` keep just two running sums — `Σ(typical_price × volume)`
and `Σ(volume)` — advanced by one bar each run. Anchored VWAP is their ratio.

- **O(1) per ticker-anchor, flat forever, independent of anchor age.** Full
  universe × 2 anchors ≈ 9,095 × 2 × ~40 B ≈ **0.7 MB, and it never grows.**
- Hard limitation: an anchor must be **nominated before or on the day it
  happens**. You cannot retro-anchor to a date you didn't call at the time, and
  you cannot repair the running sum if a bar is later revised — the underlying
  bars are gone.
- Star-candle anchors fit this cleanly: `3_screen.py` already detects
  `Star_candle` on the day it prints, so it can self-nominate.
- Post-earnings-gap anchors **do not** fit: the repo has no earnings calendar,
  so there is nothing to nominate from. That is a separate missing input, not a
  storage problem.

### E. Hybrid — D for the running value, B as the repairable backing store

Accumulators for the live read; the thin watchlist parquet behind them so any
anchor can be rebuilt, re-dated, or audited. ~1 MB total at 300 names/1 year,
plus 0.7 MB of accumulators. Strictly more capable than B or D alone; the cost
is two stores to keep consistent.

---

## Recommendation

1. **Now:** B (thin watchlist parquet), seeded from `EOD/` via A at the moment a
   ticker joins the watchlist. ~0.4–1.1 MB per year. Nothing else in the
   pipeline has to change.
2. **Before post-earnings anchoring is possible at all:** source an earnings
   calendar. This is the actual blocker for that anchor type — not bar depth.
   Neither EODData's `/Quote/List` nor the `EOD/` CSVs carry earnings dates.
3. **Defer** C outright. Revisit D/E only if the watchlist grows past a few
   hundred names.

## Budget note (CHANGE 4/1 fallout, recorded here since it is the same budget)

Widening the result CSVs took them from 4.7 KB to 52 KB — about **+19 KB of git
blob per run**, or 0.13% of the ~15.1 MB the two parquets already add every run.
It is not a size concern.

The parquets are. `.git` is 58 MB after 6 commits; at ~15.1 MB/run and 5
runs/week the pack passes GitHub's 1 GB "please slim down" threshold in roughly
**4–5 months** and 5 GB inside two years. Neither this change nor anchored VWAP
causes that, but any anchored-VWAP option should be chosen knowing that a
history rewrite or a move to release assets / LFS is coming regardless.

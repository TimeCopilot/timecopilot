#!/usr/bin/env python3
"""
processor.py

Input (JSONL, same folder):
  spreads_weekend_americanfootball_nfl.json

Output (single CSV, same folder):
  spread_mode_ts.csv

Goal
----
Create a forecastable pregame time series from spread snapshots:

  ds | unique_id | y

Definitions
-----------
- ds: snapshot timestamp (UTC)
- unique_id: contract identifier that includes:
    * event_id
    * matchup: away @ home
    * kickoff time (KO)
    * side (outcome_name)
    * fixed spread (L0, signed)
- y: de-vig, market-level implied probability that the side covers the fixed spread L0

Core idea
---------
Your feed encodes signed spreads per team (e.g., Bills +1.5 and Broncos -1.5).
Those are the SAME market line magnitude (1.5) but opposite sides.
So pairing must be done on abs(spread_point) == "line", not on spread_point itself.

Contract selection (your rule)
------------------------------
For each event-side, pick fixed L0 as the MODE of signed spread_point over the last 3 days
(relative to that side’s last snapshot). This window is selection-only.

Then build y(t) at that fixed L0 magnitude for all pregame snapshots.

Cutoff
------
We drop any snapshot after kickoff (ds > commence_time), per event.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# CONFIG (no args)
# =========================
INPUT_FILE = Path("spreads_weekend_americanfootball_nfl.json")
OUTPUT_FILE = Path("spread_mode_ts.csv")
MODE_LOOKBACK_DAYS = 3


# =========================
# ODDS HELPERS
# =========================
def american_to_implied_prob(a: float) -> float:
    """American odds -> implied probability (naive, includes vig)."""
    a = float(a)
    if a == 0:
        return np.nan
    if a > 0:
        return 100.0 / (a + 100.0)
    return abs(a) / (abs(a) + 100.0)


def mode_with_tiebreak(values: pd.Series, center: float) -> float:
    """
    Mode of numeric series. If tie, choose candidate closest to center (median).
    """
    vc = values.value_counts(dropna=True)
    if vc.empty:
        return np.nan
    max_count = vc.max()
    cands = vc[vc == max_count].index.to_numpy(dtype=float)
    if len(cands) == 1:
        return float(cands[0])
    cands = np.asarray(cands, dtype=float)
    return float(cands[np.argmin(np.abs(cands - center))])


# =========================
# LOAD JSONL
# =========================
if not INPUT_FILE.exists():
    raise SystemExit(f"Missing input file: {INPUT_FILE.resolve()}")

rows = []
with INPUT_FILE.open("r", encoding="utf-8") as f:
    for ln, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Bad JSON on line {ln}: {e}") from e

if not rows:
    raise RuntimeError("No JSON objects found in input file (empty JSONL).")

df = pd.DataFrame(rows)

print("\n=== RAW DATAFRAME (head) ===")
print(df.head())
print("\nColumns:")
print(df.columns.tolist())

required_cols = [
    "snapshot_utc",
    "event_id",
    "commence_time",
    "home_team",
    "away_team",
    "bookmaker_key",
    "outcome_name",
    "spread_point",
    "american_price",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")


# =========================
# CLEAN TYPES + PRE-GAME CUTOFF
# =========================
df["ds"] = pd.to_datetime(df["snapshot_utc"], utc=True, errors="coerce")
df["kickoff"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
df["spread_point"] = pd.to_numeric(df["spread_point"], errors="coerce")
df["american_price"] = pd.to_numeric(df["american_price"], errors="coerce")

if df["ds"].isna().any():
    raise RuntimeError("Some snapshot_utc failed to parse.")
if df["kickoff"].isna().any():
    raise RuntimeError("Some commence_time failed to parse.")

# Keep only pregame snapshots
df = df[df["ds"] <= df["kickoff"]].copy()

# Implied prob and market line magnitude
df["imp_prob"] = df["american_price"].map(american_to_implied_prob)
df["line"] = df["spread_point"].abs()

# Side identifier (event + team-side)
df["side_id"] = (
    df["event_id"].astype(str)
    + " | "
    + df["away_team"].astype(str)
    + " @ "
    + df["home_team"].astype(str)
    + " | "
    + df["outcome_name"].astype(str)
)


# =========================
# EVENT META (for reliable "who vs who" and kickoff)
# =========================
event_meta = (
    df.groupby("event_id", as_index=False)
      .agg(
          away_team=("away_team", "first"),
          home_team=("home_team", "first"),
          kickoff=("kickoff", "first"),
      )
)
event_meta["matchup"] = event_meta["away_team"] + " @ " + event_meta["home_team"]


# =========================
# STEP 1: SELECT FIXED L0 PER SIDE (MODE over last 3 days)
# =========================
l0_rows = []
for side_id, g in df.groupby("side_id"):
    last_ds = g["ds"].max()
    window_start = last_ds - pd.Timedelta(days=MODE_LOOKBACK_DAYS)
    gw = g[g["ds"] >= window_start]
    if gw.empty:
        continue

    center = float(gw["spread_point"].median())
    l0 = mode_with_tiebreak(gw["spread_point"], center=center)
    l0_rows.append({"side_id": side_id, "L0": l0})

l0_df = pd.DataFrame(l0_rows)
if l0_df.empty:
    raise RuntimeError("Failed to compute L0 for any side.")

df = df.merge(l0_df, on="side_id", how="left")


# =========================
# STEP 2: MARKET-LEVEL medians per snapshot+event+line+side
# =========================
# We take the median across books for robustness.
market_med = (
    df.groupby(["ds", "event_id", "line", "outcome_name"], as_index=False)["imp_prob"]
      .median()
      .rename(columns={"imp_prob": "p_med"})
)

# Bring matchup + kickoff info back in
market_med = market_med.merge(event_meta, on="event_id", how="left")

# Rebuild side_id so we can attach L0
market_med["side_id"] = (
    market_med["event_id"].astype(str)
    + " | "
    + market_med["matchup"].astype(str)
    + " | "
    + market_med["outcome_name"].astype(str)
)

# But our original side_id used away/home from df, not "matchup" string.
# Create the same format for a safe join:
market_med["side_id"] = (
    market_med["event_id"].astype(str)
    + " | "
    + market_med["away_team"].astype(str)
    + " @ "
    + market_med["home_team"].astype(str)
    + " | "
    + market_med["outcome_name"].astype(str)
)

market_med = market_med.merge(l0_df, on="side_id", how="left")

# Require both outcomes exist at the SAME (ds,event,line) to de-vig
counts = (
    market_med.groupby(["ds", "event_id", "line"], as_index=False)["outcome_name"]
              .nunique()
              .rename(columns={"outcome_name": "n_outcomes"})
)
market_med = market_med.merge(counts, on=["ds", "event_id", "line"], how="left")
market_med = market_med[market_med["n_outcomes"] == 2].copy()

if market_med.empty:
    raise RuntimeError(
        "No paired sides at (ds,event,abs(line)). Collector likely incomplete or sparse."
    )

# De-vig: normalize the two side medians so they sum to 1 per (ds,event,line)
den = (
    market_med.groupby(["ds", "event_id", "line"], as_index=False)["p_med"]
              .sum()
              .rename(columns={"p_med": "p_sum"})
)
market_med = market_med.merge(den, on=["ds", "event_id", "line"], how="left")
market_med["p_devig"] = market_med["p_med"] / market_med["p_sum"]


# =========================
# STEP 3: FILTER TO FIXED L0 (by magnitude) and build time series
# =========================
market_med["L0_line"] = market_med["L0"].abs()
fixed = market_med[market_med["line"] == market_med["L0_line"]].copy()

if fixed.empty:
    raise RuntimeError(
        "No rows match selected L0 magnitude.\n"
        "Try changing MODE_LOOKBACK_DAYS or check your data density."
    )

# unique_id embeds:
# - event_id
# - matchup (away @ home)
# - kickoff
# - side (team)
# - signed L0
fixed["unique_id"] = (
    fixed["event_id"].astype(str)
    + " | "
    + fixed["matchup"].astype(str)
    + " | KO="
    + fixed["kickoff"].dt.strftime("%Y-%m-%dT%H:%MZ")
    + " | "
    + fixed["outcome_name"].astype(str)
    + " | L0="
    + fixed["L0"].map(lambda x: f"{x:+g}")
)

ts_df = (
    fixed.groupby(["ds", "unique_id"], as_index=False)["p_devig"]
         .median()
         .rename(columns={"p_devig": "y"})
         .sort_values(["unique_id", "ds"])
         .reset_index(drop=True)
)

print("\n=== FINAL TIME SERIES (head) ===")
print(ts_df.head(20))


# =========================
# STEP 4: VERY VERBOSE SUMMARY
# =========================
def verbose_summary(ts: pd.DataFrame) -> None:
    for uid, g in ts.groupby("unique_id"):
        g = g.sort_values("ds")
        std = float(g["y"].std(ddof=1)) if len(g) > 1 else np.nan
        first_ds, last_ds = g["ds"].iloc[0], g["ds"].iloc[-1]
        first_y, last_y = float(g["y"].iloc[0]), float(g["y"].iloc[-1])

        # Parse human-friendly bits from uid
        # event_id | matchup | KO=... | team | L0=...
        parts = [p.strip() for p in uid.split("|")]
        matchup = parts[1] if len(parts) > 1 else ""
        ko = parts[2].replace("KO=", "").strip() if len(parts) > 2 else ""
        team = parts[3] if len(parts) > 3 else ""
        l0 = parts[4].replace("L0=", "").strip() if len(parts) > 4 else ""

        print("\n" + "=" * 90)
        print(f"{matchup} (KO {ko})")
        print(f"{team} mode spread is {l0}, std deviation for price at this spread is: {std:.6f}")
        print(f"first value was y={first_y:.6f} at {first_ds.isoformat()}")
        print(f"last  value was y={last_y:.6f} at {last_ds.isoformat()}")
        print(f"n_points: {len(g)}")


print("\n=== VERBOSE SUMMARIES ===")
verbose_summary(ts_df)


# =========================
# WRITE CSV
# =========================
ts_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved → {OUTPUT_FILE.resolve()}")


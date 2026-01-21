#!/usr/bin/env python3
"""
Download (snapshot) NFL Divisional Round weekend spreads from multiple books
using The Odds API v4, and append as JSONL.

Output: one JSON object per line (JSONL). Easy to grep / load later.

Setup:
  pip install requests
  export ODDS_API_KEY="YOUR_KEY"

Run:
  python nfl_weekend_spreads_jsonl.py
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import requests

BASE_URL = "https://api.the-odds-api.com/v4"
TZ_LOCAL = ZoneInfo("America/Mexico_City")  # your timezone


@dataclass
class SpreadRow:
    snapshot_utc: str
    snapshot_local: str

    sport_key: str
    event_id: str
    commence_time: str  # ISO

    home_team: str
    away_team: str

    bookmaker_key: str
    bookmaker_title: str
    bookmaker_last_update: str | None

    outcome_name: str
    spread_point: float | None
    american_price: int | None


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def iso_local_now() -> str:
    return datetime.now(TZ_LOCAL).replace(microsecond=0).isoformat()


def weekend_window_local(anchor_local: datetime) -> tuple[datetime, datetime]:
    """
    Returns (start, end) local datetimes covering THIS weekend:
      Saturday 00:00 local -> Monday 00:00 local
    If it's already weekend, it still returns the current weekend window.
    """
    # Monday=0 ... Sunday=6
    dow = anchor_local.weekday()
    # Find upcoming Saturday (or today if Saturday)
    days_until_sat = (5 - dow) % 7
    sat = anchor_local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        days=days_until_sat
    )
    mon = sat + timedelta(days=2)
    return sat, mon


def get_odds(api_key: str, sport_key: str) -> list[dict[str, Any]]:
    """
    v4 odds endpoint (featured markets like spreads/totals/h2h).
    """
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "spreads",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    url = f"{BASE_URL}/sports/{sport_key}/odds"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def parse_iso_as_utc(dt_str: str) -> datetime:
    # The Odds API returns ISO timestamps; treat "Z" as UTC.
    # Python fromisoformat doesn't like "Z" directly.
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    return datetime.fromisoformat(dt_str).astimezone(timezone.utc)


def in_window(
    commence_time_iso: str, start_local: datetime, end_local: datetime
) -> bool:
    commence_utc = parse_iso_as_utc(commence_time_iso)
    commence_local = commence_utc.astimezone(TZ_LOCAL)
    return start_local <= commence_local < end_local


def flatten_spreads(
    events: list[dict[str, Any]], snapshot_utc: str, snapshot_local: str
) -> list[SpreadRow]:
    rows: list[SpreadRow] = []

    for ev in events:
        sport_key = ev.get("sport_key", "")
        event_id = ev.get("id", "")
        commence_time = ev.get("commence_time", "")
        home_team = ev.get("home_team", "")
        away_team = ev.get("away_team", "")

        for bk in ev.get("bookmakers", []) or []:
            bk_key = bk.get("key", "")
            bk_title = bk.get("title", "")
            bk_last_update = bk.get("last_update")

            for mkt in bk.get("markets", []) or []:
                if mkt.get("key") != "spreads":
                    continue

                for out in mkt.get("outcomes", []) or []:
                    rows.append(
                        SpreadRow(
                            snapshot_utc=snapshot_utc,
                            snapshot_local=snapshot_local,
                            sport_key=sport_key,
                            event_id=event_id,
                            commence_time=commence_time,
                            home_team=home_team,
                            away_team=away_team,
                            bookmaker_key=bk_key,
                            bookmaker_title=bk_title,
                            bookmaker_last_update=bk_last_update,
                            outcome_name=out.get("name"),
                            spread_point=out.get("point"),
                            american_price=out.get("price"),
                        )
                    )
    return rows


def append_jsonl(path: str, rows: list[SpreadRow]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def main() -> None:
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise SystemExit("Missing ODDS_API_KEY env var.")

    sport_key = os.getenv("SPORT_KEY", "americanfootball_nfl")

    now_local = datetime.now(TZ_LOCAL)
    start_local, end_local = weekend_window_local(now_local)

    snapshot_utc = iso_utc_now()
    snapshot_local = iso_local_now()

    print(f"[{snapshot_local}] Weekend window (local): {start_local} -> {end_local}")
    print(f"[{snapshot_local}] Fetching spreads for: {sport_key}")

    events = get_odds(api_key=api_key, sport_key=sport_key)

    weekend_events = [
        ev
        for ev in events
        if ev.get("commence_time")
        and in_window(ev["commence_time"], start_local, end_local)
    ]
    print(
        f"[{snapshot_local}] Events returned: {len(events)} | "
        f"weekend-matching: {len(weekend_events)}"
    )

    rows = flatten_spreads(
        weekend_events, snapshot_utc=snapshot_utc, snapshot_local=snapshot_local
    )
    out_path = os.getenv("OUT_PATH", f"spreads_weekend_{sport_key}.jsonl")

    append_jsonl(out_path, rows)
    print(f"[{snapshot_local}] Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()

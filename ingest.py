"""
Data ingestion.

NETWORK NOTE: stats.nba.com is not on this container's allowlist, so the
game-log path here is written but UNTESTED against the live endpoint. It is
tested against fixtures in demo.py. To run it for real either execute locally or
ask a GDC org owner to add *.nba.com to the network settings.
api.opticodds.com IS allowlisted.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from . import rates as rates_mod

DATA = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Game logs
# ---------------------------------------------------------------------------
def fetch_player_logs(season: str = "2025-26",
                      season_type: str = "Regular Season",
                      measure_type: str = "Base",
                      retries: int = 4) -> pd.DataFrame:
    """
    Pull every player's game logs for a season in ONE request.

    Uses PlayerGameLogs (plural), not PlayerGameLog (singular). The singular
    endpoint is per-player and would mean ~570 requests against a host that
    rate-limits by silently hanging rather than returning 429 -- which is also
    why the retry loop below backs off rather than failing fast.
    """
    from nba_api.stats.endpoints import playergamelogs

    last_err = None
    for attempt in range(retries):
        try:
            resp = playergamelogs.PlayerGameLogs(
                season_nullable=season,
                season_type_nullable=season_type,
                measure_type_player_game_logs_nullable=measure_type,
                timeout=45,
            )
            df = resp.get_data_frames()[0]
            break
        except Exception as err:          # includes read timeouts
            last_err = err
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"PlayerGameLogs failed after {retries} tries: {last_err}")

    df = rates_mod.normalize_columns(df, "nba_api")
    df["game_date"] = pd.to_datetime(df["game_date"])
    keep = [
        "nba_player_id", "game_id", "game_date", "team", "minutes",
        "fga", "fgm", "pt3a", "pt3m", "fta", "ftm",
        "oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
    ]
    return df[[c for c in keep if c in df.columns]]


def incremental_logs(season: str, since: pd.Timestamp | None) -> pd.DataFrame:
    """
    Only the games not yet folded into state.

    The state accumulator is NOT idempotent -- re-feeding a game double-counts
    its evidence. Always gate on the watermark.
    """
    df = fetch_player_logs(season)
    if since is not None:
        df = df[df["game_date"] > pd.Timestamp(since)]
    return df.sort_values("game_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Odds
# ---------------------------------------------------------------------------
def fetch_lines(date: str, api_key: str | None = None,
                sportsbook: str = "pinnacle") -> pd.DataFrame:
    """
    Spreads and totals per team-game from OpticOdds.

    Returns one row per TEAM-game (two rows per game), with `spread` from that
    team's perspective, negative = favorite. That orientation matters: every
    downstream formula assumes it.

    Pinnacle by default -- lowest vig, so the implied numbers are closest to a
    true probability, which is what the efficiency decomposition needs.
    """
    import requests

    key = api_key or os.environ.get("OPTICODDS_API_KEY")
    if not key:
        raise RuntimeError("set OPTICODDS_API_KEY")

    resp = requests.get(
        "https://api.opticodds.com/api/v3/fixtures/odds",
        params={
            "sport": "basketball",
            "league": "nba",
            "start_date": date,
            "sportsbook": sportsbook,
            "market": ["point_spread", "total_points"],
            "key": key,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return _parse_optic(resp.json())


def _parse_optic(payload: dict) -> pd.DataFrame:
    """
    Flatten the OpticOdds response into (game_id, team, opp, is_home, spread,
    total).

    Kept separate from the HTTP call so it can be unit-tested on a saved
    payload without hitting the API.
    """
    rows = []
    for fx in payload.get("data", []):
        home = fx.get("home_team_display") or fx.get("home_competitors", [{}])[0].get("abbreviation")
        away = fx.get("away_team_display") or fx.get("away_competitors", [{}])[0].get("abbreviation")
        gid = fx.get("id")
        spreads, total = {}, None
        for odd in fx.get("odds", []):
            mk = (odd.get("market_id") or odd.get("market") or "").lower()
            if "spread" in mk:
                spreads[odd.get("selection")] = odd.get("points")
            elif "total" in mk and total is None:
                total = odd.get("points")
        if total is None or not spreads:
            continue
        for team, opp, is_home in ((home, away, 1), (away, home, 0)):
            sp = spreads.get(team)
            if sp is None:
                continue
            rows.append({
                "game_id": gid, "team": team, "opp": opp,
                "is_home": is_home, "spread": float(sp),
                "total": float(total), "game_date": fx.get("start_date"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Team abbreviation crosswalk
# ---------------------------------------------------------------------------
# RotoWire and ESPN disagree on 7 of 30 abbreviations. Silently inner-joining
# on `team` therefore drops those seven teams' pace priors and lines with no
# error -- roughly a quarter of the league, invisibly. Every join on team must
# go through normalize_team() first.
ROTOWIRE_TO_ESPN = {
    "BRO": "BKN", "CHR": "CHA", "NOR": "NO", "PHO": "PHX",
    "SAN": "SA", "UTA": "UTAH", "WAS": "WSH",
}
ESPN_TO_ROTOWIRE = {v: k for k, v in ROTOWIRE_TO_ESPN.items()}


def normalize_team(s: pd.Series, to: str = "rotowire") -> pd.Series:
    """Map a team abbreviation column into one naming convention."""
    s = s.astype("string").str.strip()
    if to == "rotowire":
        return s.replace(ESPN_TO_ROTOWIRE)
    if to == "espn":
        return s.replace(ROTOWIRE_TO_ESPN)
    raise ValueError(f"to must be 'rotowire' or 'espn', got {to!r}")


def check_team_join(left: pd.Series, right: pd.Series, label: str = "") -> None:
    """Raise if two team columns do not fully overlap. Call before any join."""
    l, r = set(left.dropna()), set(right.dropna())
    if l - r or r - l:
        raise ValueError(
            f"team abbreviation mismatch{' in ' + label if label else ''}: "
            f"left-only={sorted(l - r)} right-only={sorted(r - l)}. "
            "Run ingest.normalize_team() on both sides first."
        )


# ---------------------------------------------------------------------------
# Player crosswalk
# ---------------------------------------------------------------------------
def build_crosswalk(preseason: pd.DataFrame,
                    overrides: dict[int, int] | None = None) -> pd.DataFrame:
    """
    Map RotoWire newsid -> nba_player_id.

    Nothing in nba_api speaks newsid, so this is a hard dependency for every
    join in the pipeline. Name matching gets most of the way; the residual is
    always suffixes, accents, and Jr/II, and it has to be resolved by hand once
    and then stored. Ship the overrides dict in version control.

    Returns the matched frame plus an `unmatched` attribute listing the misses,
    which should be reviewed rather than silently dropped -- an unmatched
    starter is a silently missing projection.
    """
    from nba_api.stats.static import players as static_players

    nba = pd.DataFrame(static_players.get_players())
    nba["key"] = (
        nba["full_name"].str.lower()
        .str.replace(r"[^a-z ]", "", regex=True)
        .str.replace(r"\s+(jr|sr|ii|iii|iv)$", "", regex=True)
        .str.strip()
    )

    pre = preseason.copy()
    pre.columns = [c.strip() for c in pre.columns]
    pre["key"] = (
        (pre["firstname"].astype(str) + " " + pre["lastname"].astype(str))
        .str.lower()
        .str.replace(r"[^a-z ]", "", regex=True)
        .str.replace(r"\s+(jr|sr|ii|iii|iv)$", "", regex=True)
        .str.strip()
    )

    merged = pre.merge(
        nba[["key", "id"]].rename(columns={"id": "nba_player_id"}),
        on="key", how="left",
    )
    if overrides:
        for newsid, nba_id in overrides.items():
            merged.loc[merged["newsid"] == newsid, "nba_player_id"] = nba_id

    out = merged[["newsid", "firstname", "lastname", "team", "nba_player_id"]]
    out.attrs["unmatched"] = out[out["nba_player_id"].isna()]
    return out


# ---------------------------------------------------------------------------
# Local storage
# ---------------------------------------------------------------------------
def save(df: pd.DataFrame, name: str) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"{name}.parquet")

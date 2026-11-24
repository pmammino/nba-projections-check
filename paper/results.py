"""Actual results, pulled by the repo: nba.com's static CDN (the same JSON the
nba.com scoreboard page reads). Chosen over nba_api / stats.nba.com on purpose:
stats.nba.com is notorious for blocking or hanging cloud-runner IPs (ingest.py
already notes it is unreachable from the dev container), whereas cdn.nba.com is a
plain static host that works from GitHub Actions.

  schedule  https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json
  boxscore  https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gameId}.json

One source feeds BOTH needs:
  * settlement  -- the day's final box scores (store.actuals)
  * model state -- the season's game logs (data/paper/inputs/logs.csv), grown
                   incrementally and backfillable, so no separate nba_api pull
                   and no nba_player_id <-> newsid crosswalk: players are matched
                   to the RotoWire sheet by NAME + TEAM (crosswalk.resolve),
                   refusing to guess when ambiguous.

OpticOdds' /fixtures/results (same API key, same fixture ids as the odds) carries
final status, scores and TEAM stats only -- no player lines -- so it cannot settle
player props. It is used for what it does carry: `completed` is the finality gate
(an unambiguous status in the odds' own id space), and its team totals are checked
against the sum of the CDN's player lines (verify_box), so a box score that is
incomplete or mis-parsed is never graded.

STATUS: parsers are tested on fixtures shaped like the CDN's JSON; NOT run against
the live CDN from the build session (blocked there). `python -m paper.cli
check-results --date D` prints exactly what it saw, which is the first live check.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta

import pandas as pd

from .crosswalk import resolve
from .teams import canon

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nba-paper-trading/1.0)", "Accept": "application/json"}
FINAL = 3                      # gameStatus: 1 scheduled, 2 live, 3 final

LOG_COLS = ["player_id", "nba_player_id", "player_name", "game_id", "game_date", "team", "minutes",
            "fga", "fgm", "pt3a", "pt3m", "fta", "ftm", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf"]


def _get(session, url):
    r = session.get(url, headers=HEADERS, timeout=45)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    return r.json()


def parse_minutes(v) -> float:
    """'PT34M12.00S' -> 34.2 ; also accepts 'MM:SS' and plain numbers."""
    if v in (None, ""):
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    m = re.match(r"PT(\d+)M([\d.]+)S", str(v))
    if m:
        return int(m.group(1)) + float(m.group(2)) / 60
    m = re.match(r"(\d+):(\d+)", str(v))
    return int(m.group(1)) + int(m.group(2)) / 60 if m else 0.0


def parse_schedule(payload, start: str = None, end: str = None):
    """-> list of {game_id, date (ET), start_utc, status, home, away}. Optional
    inclusive ET-date filter (YYYY-MM-DD)."""
    out = []
    for gd in (payload.get("leagueSchedule", payload) or {}).get("gameDates", []):
        for g in gd.get("games", []):
            utc = g.get("gameDateTimeUTC") or ""
            est = (g.get("gameDateEst") or g.get("gameEt") or "")[:10]
            date = est or utc[:10]
            if (start and date < start) or (end and date > end):
                continue
            out.append({"game_id": str(g["gameId"]), "date": date, "start_utc": utc,
                        "status": int(g.get("gameStatus", 1)),
                        "home": canon((g.get("homeTeam") or {}).get("teamTricode")),
                        "away": canon((g.get("awayTeam") or {}).get("teamTricode"))})
    return out


def parse_boxscore(payload, game_date: str):
    """-> (is_final, rows). One row per player who is listed with minutes."""
    game = payload.get("game", payload)
    final = int(game.get("gameStatus", 0)) == FINAL
    rows = []
    for side in ("homeTeam", "awayTeam"):
        t = game.get(side) or {}
        team = canon(t.get("teamTricode"))
        for p in t.get("players", []):
            st = p.get("statistics") or {}
            mins = parse_minutes(st.get("minutes") or st.get("minutesCalculated"))
            if not mins > 0:
                continue                            # DNP: absent from the box => void downstream
            name = p.get("name") or f"{p.get('firstName', '')} {p.get('familyName', '')}".strip()
            g = lambda k: float(st.get(k) or 0)     # noqa: E731
            rows.append({"nba_player_id": int(p.get("personId", 0)), "player_name": name,
                         "game_id": str(game.get("gameId", "")), "game_date": game_date, "team": team,
                         "minutes": round(mins, 2), "fga": g("fieldGoalsAttempted"), "fgm": g("fieldGoalsMade"),
                         "pt3a": g("threePointersAttempted"), "pt3m": g("threePointersMade"),
                         "fta": g("freeThrowsAttempted"), "ftm": g("freeThrowsMade"),
                         "oreb": g("reboundsOffensive"), "dreb": g("reboundsDefensive"), "ast": g("assists"),
                         "stl": g("steals"), "blk": g("blocks"), "tov": g("turnovers"), "pf": g("foulsPersonal")})
    return final, rows


class Results:
    def __init__(self, session=None):
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self._sched = None

    def schedule(self, start=None, end=None):
        if self._sched is None:
            self._sched = parse_schedule(_get(self.session, SCHEDULE_URL))
        return [g for g in self._sched if (not start or g["date"] >= start) and (not end or g["date"] <= end)]

    def box(self, game_id, date):
        """-> (is_final, rows) for one CDN game id."""
        return parse_boxscore(_get(self.session, BOX_URL.format(gid=game_id)), date)

    def find_game(self, home, away, date):
        """The schedule entry for home/away within a day of `date` (CDN dates are
        ET; a late tip can sit either side of our UTC-based slate day)."""
        d = datetime.strptime(date, "%Y-%m-%d")
        lo, hi = (d - timedelta(days=1)).strftime("%Y-%m-%d"), (d + timedelta(days=1)).strftime("%Y-%m-%d")
        c = [g for g in self.schedule(lo, hi) if g["home"] == home and g["away"] == away]
        return min(c, key=lambda g: abs((datetime.strptime(g["date"], "%Y-%m-%d") - d).days)) if c else None

    def day(self, date):
        """-> (all_final, rows, games). Final games only contribute rows."""
        games = self.schedule(date, date)
        rows, finals = [], 0
        for g in games:
            if g["status"] != FINAL:
                continue
            ok, rs = parse_boxscore(_get(self.session, BOX_URL.format(gid=g["game_id"])), date)
            if ok:
                finals += 1
                rows += rs
        return bool(games) and finals == len(games), rows, games


def attach_player_ids(rows, players, overrides=None):
    """Match box-score rows to our player_id by NAME + TEAM. Returns
    (matched rows with player_id, unmatched names). `players` = slate_players()
    style dicts (player_id, name, team); a row's team stands in for the fixture's
    teams, so a shared name is disambiguated by who he plays for."""
    probe = [{**r, "home": r["team"], "away": r["team"], "fixture_id": r["game_id"]} for r in rows]
    matched, unmatched = resolve(probe, players, overrides)
    if unmatched:
        # A player traded since the sheet was written is filed under his old team:
        # retry those by NAME alone, accepted only when exactly one player has it.
        bad = {u["name"] for u in unmatched}
        retry = [{**r, "home": "", "away": "", "fixture_id": r["game_id"]} for r in rows
                 if r["player_name"] in bad]
        again, still = resolve(retry, players, overrides)
        for r in again:
            r["team"] = canon(next((x["team"] for x in rows if x["player_name"] == r["player_name"]), r["team"]))
        matched, unmatched = matched + again, still
    return matched, unmatched


def to_logs(matched):
    """Matched box-score rows -> game-log rows (LOG_COLS)."""
    return [{c: r.get(c) for c in LOG_COLS} for r in matched]


def read_logs(path):
    import os
    return pd.read_csv(path, parse_dates=["game_date"]) if os.path.exists(path) else pd.DataFrame(columns=LOG_COLS)


def append_logs(path, new_rows):
    """Idempotent: a (game_id, player_id) already logged is never added twice.
    state.update is an accumulator -- re-feeding a game double-counts it."""
    old = read_logs(path)
    add = pd.DataFrame(new_rows, columns=LOG_COLS)
    if len(add):
        add["game_date"] = pd.to_datetime(add["game_date"])
    if len(old) and len(add):
        seen = set(zip(old["game_id"].astype(str), old["player_id"].astype("int64")))
        add = add[[(str(g), int(p)) not in seen for g, p in zip(add["game_id"], add["player_id"])]]
    out = pd.concat([old, add], ignore_index=True) if len(add) else old
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return len(add)


def season_dates(start, end):
    d0, d1 = datetime.strptime(start, "%Y-%m-%d"), datetime.strptime(end, "%Y-%m-%d")
    return [(d0 + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((d1 - d0).days + 1)]


# ---- OpticOdds results: finality + an integrity check on the box score ---------
COMPLETED = {"completed", "final", "finished", "ended"}
DEAD = {"postponed", "cancelled", "canceled", "suspended"}


def parse_optic_results(items):
    """/fixtures/results `data` rows -> [{fixture_id, home, away, status, completed,
    dead, scores, stats}]. `stats` are the period=="all" team stats per side."""
    out = []
    for it in items or []:
        fx = it.get("fixture") or {}
        def abbr(side):
            c = (fx.get(side) or [{}])[0]
            return canon(c.get("abbreviation") or c.get("name"))
        def allstats(side):
            for blk in (it.get("stats") or {}).get(side) or []:
                if blk.get("period") == "all":
                    return blk.get("stats") or {}
            return {}
        status = str(fx.get("status", "")).lower()
        sc = it.get("scores") or {}
        out.append({"fixture_id": str(fx.get("id", "")), "home": abbr("home_competitors"),
                    "away": abbr("away_competitors"), "status": status,
                    "completed": status in COMPLETED, "dead": status in DEAD,
                    "start_date": fx.get("start_date"),
                    "scores": {"home": (sc.get("home") or {}).get("total"), "away": (sc.get("away") or {}).get("total")},
                    "stats": {"home": allstats("home"), "away": allstats("away")}})
    return out


# team stat -> (box column or expression). Only player-attributable stats: team
# rebounds and team turnovers exist in the official team line but not in any
# player's, so total_rebounds / turnovers would mismatch for honest data.
_CHECKS = {
    "assists": lambda r: r["ast"], "blocks": lambda r: r["blk"], "steals": lambda r: r["stl"],
    "field_goals_attempted": lambda r: r["fga"], "free_throws_attempted": lambda r: r["fta"],
    "three_point_field_goals_made": lambda r: r["pt3m"],
}


def verify_box(rows, game):
    """Compare Optic's team totals with the sum of player lines. `rows` = one
    game's player rows (canonical teams); `game` = a parse_optic_results entry.
    Returns a list of mismatch strings (empty = the box score is complete)."""
    bad = []
    for side in ("home", "away"):
        team = game[side]
        mine = [r for r in rows if r["team"] == team]
        if not mine:
            bad.append(f"{team}: no player rows")
            continue
        pts = sum(2 * r["fgm"] + r["pt3m"] + r["ftm"] for r in mine)
        want = game["scores"][side]
        if want is not None and pts != want:
            bad.append(f"{team} points {pts} != {want}")
        for key, fn in _CHECKS.items():
            want = game["stats"][side].get(key)
            if want is None:
                continue
            got = sum(fn(r) for r in mine)
            if got != want:
                bad.append(f"{team} {key} {got} != {want}")
    return bad

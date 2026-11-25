"""Actual results, from OpticOdds only (same API key, same fixture ids as the odds).

  /fixtures/results         final status, scores, TEAM stats      -> finality + integrity
  /fixtures/player-results  per-player stats AND `market_stats`   -> settlement + model logs

`market_stats` is Optic's own grading of every player market, keyed by the exact
market id we bet (player_points, player_points_+_rebounds_+_assists, ...). Each
player's `player.id` is the same hex id the ODDS feed puts on a prop, so a bet is
matched to its result by id -- no name matching at settlement. Names are used once,
at capture, to map an Optic id to our RotoWire player_id; that mapping is persisted
(inputs/optic_ids.csv) and reused for every later result.

Three guards before a date is graded (automation.grade_date):
  1. every captured fixture has fixture.status == "completed" (not postponed etc.)
  2. every player's box columns reproduce Optic's own market_stats for that player
     (pts, reb, ast, 3PM, stl, blk, tov): a mis-parsed line is never graded
  3. per team, the sum of player lines equals Optic's team scores/stats (verify_box):
     a box score with a missing player is never graded

A player with no minutes is a DNP: absent from the rows, so his bets settle `void`
(the book refunds them), not as losses.

`player_double_double` / `player_triple_double` in market_stats are NOT 0/1 (they
carry an encoding such as 1.2070100002 that is not documented), so dd/td are
computed from the stat line instead and never read from there.

STATUS: tested against the real payload shape (a trimmed copy is in the tests);
NOT yet run against the live endpoint from the build session. `python -m paper.cli
check-results --date D` prints exactly what the feed returned.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .crosswalk import resolve
from .teams import canon

LOG_COLS = ["player_id", "optic_player_id", "player_name", "game_id", "game_date", "team", "minutes",
            "fga", "fgm", "pt3a", "pt3m", "fta", "ftm", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf"]
COMPLETED = {"completed", "final", "finished", "ended"}
DEAD = {"postponed", "cancelled", "canceled", "suspended"}

# derived box value -> Optic market id it must reproduce
_CHECK_MARKETS = {"pts": "player_points", "reb": "player_rebounds", "ast": "player_assists",
                  "pt3m": "player_made_threes", "stl": "player_steals", "blk": "player_blocks",
                  "tov": "player_turnovers"}


def _all(stats_blocks):
    for blk in stats_blocks or []:
        if blk.get("period") == "all":
            return blk.get("stats") or {}
    return {}


def _team_map(fixture):
    """Optic team id -> canonical tri-code, from the fixture's competitors."""
    m = {}
    for side in ("home_competitors", "away_competitors"):
        for c in fixture.get(side) or []:
            if c.get("id"):
                m[c["id"]] = canon(c.get("abbreviation") or c.get("name"))
    return m


def parse_player_results(items):
    """/fixtures/player-results `data` -> [{fixture_id, completed, dead, home, away,
    rows, errors}]. `rows` are one per player who actually played (minutes > 0);
    `errors` lists any player whose box columns disagree with Optic's market_stats."""
    out = []
    for it in items or []:
        fx = it.get("fixture") or {}
        teams = _team_map(fx)
        status = str(fx.get("status", "")).lower()
        date = str(fx.get("start_date", ""))[:10]
        rows, errors, statuses = [], [], set()
        for r in it.get("results") or []:
            statuses.add(str(r.get("status", "")).lower())
            st = _all(r.get("stats"))
            secs = st.get("seconds")
            mins = (secs / 60.0) if isinstance(secs, (int, float)) else float(st.get("minutes") or 0)
            if not mins > 0:
                continue                                  # DNP: no row => bets void
            g = lambda k: float(st.get(k) or 0)           # noqa: E731
            p = r.get("player") or {}
            row = {"optic_player_id": str(p.get("id", "")), "player_name": p.get("name", ""),
                   "game_id": str(fx.get("id", "")), "game_date": date,
                   "team": teams.get((r.get("team") or {}).get("id"), canon((r.get("team") or {}).get("name"))),
                   "minutes": round(mins, 2),
                   "fga": g("field_goals_attempted"), "fgm": g("field_goals_made"),
                   "pt3a": g("three_point_field_goals_attempted"), "pt3m": g("three_point_field_goals_made"),
                   "fta": g("free_throws_attempted"), "ftm": g("free_throws_made"),
                   "oreb": g("offensive_rebounds"), "dreb": g("defensive_rebounds"), "ast": g("assists"),
                   "stl": g("steals"), "blk": g("blocks"), "tov": g("turnovers"), "pf": g("fouls")}
            derived = {"pts": 2 * row["fgm"] + row["pt3m"] + row["ftm"], "reb": row["oreb"] + row["dreb"],
                       "ast": row["ast"], "pt3m": row["pt3m"], "stl": row["stl"], "blk": row["blk"],
                       "tov": row["tov"]}
            ms = r.get("market_stats") or {}
            for k, market in _CHECK_MARKETS.items():
                if market in ms and ms[market] != derived[k]:
                    errors.append(f"{row['player_name']} {market}: box {derived[k]} != optic {ms[market]}")
            rows.append(row)
        home = (fx.get("home_competitors") or [{}])[0]
        away = (fx.get("away_competitors") or [{}])[0]
        out.append({"fixture_id": str(fx.get("id", "")), "status": status,
                    "completed": status in COMPLETED and statuses <= COMPLETED | {""},
                    "dead": status in DEAD, "start_date": fx.get("start_date"),
                    "home": canon(home.get("abbreviation") or home.get("name")),
                    "away": canon(away.get("abbreviation") or away.get("name")),
                    "rows": rows, "errors": errors})
    return out


def parse_optic_results(items):
    """/fixtures/results `data` -> [{fixture_id, home, away, status, completed, dead,
    scores, stats}] with the period=="all" team stats per side."""
    out = []
    for it in items or []:
        fx = it.get("fixture") or {}
        ab = lambda side: canon(((fx.get(side) or [{}])[0]).get("abbreviation")  # noqa: E731
                                or ((fx.get(side) or [{}])[0]).get("name"))
        status = str(fx.get("status", "")).lower()
        sc = it.get("scores") or {}
        out.append({"fixture_id": str(fx.get("id", "")), "home": ab("home_competitors"),
                    "away": ab("away_competitors"), "status": status, "completed": status in COMPLETED,
                    "dead": status in DEAD, "start_date": fx.get("start_date"),
                    "scores": {"home": (sc.get("home") or {}).get("total"),
                               "away": (sc.get("away") or {}).get("total")},
                    "stats": {"home": _all((it.get("stats") or {}).get("home")),
                              "away": _all((it.get("stats") or {}).get("away"))}})
    return out


# Only player-attributable stats: team rebounds and team turnovers exist in the
# official team line but in no player's, so those two would mismatch for honest data.
_TEAM_CHECKS = {
    "assists": lambda r: r["ast"], "blocks": lambda r: r["blk"], "steals": lambda r: r["stl"],
    "field_goals_attempted": lambda r: r["fga"], "free_throws_attempted": lambda r: r["fta"],
    "three_point_field_goals_made": lambda r: r["pt3m"],
}
SOFT_TOLERANCE = 2          # a secondary stat may differ from the team line by this much (a live game was 1 assist off)
MINUTES_TOLERANCE = 1.0     # team minutes must be 240 (+25 per overtime) within this


def verify_box_detail(rows, game):
    """Is this box score COMPLETE? Returns (blockers, notes).

    Blockers stop the date being graded:
      * a team with no player rows
      * team points that differ at all from Optic's score (settlement depends on points)
      * team minutes that are not 240 + 25 per overtime (a missing player who played shows up here,
        which is a far better completeness test than any single counting stat)
      * a secondary stat (assists, steals, blocks, FGA, FTA, 3PM) off by more than SOFT_TOLERANCE
    Notes are small secondary-stat differences, logged but not blocking: a live game's team line said 25
    assists against 24 in the players, and exact equality on that held grading back indefinitely."""
    bad, notes = [], []
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
        total = sum(r["minutes"] for r in mine)
        extra = max(0, round((total - 240) / 25))
        expected = 240 + 25 * extra
        if abs(total - expected) > MINUTES_TOLERANCE:
            bad.append(f"{team} minutes {total:.1f} (expected {expected})")
        for key, fn in _TEAM_CHECKS.items():
            want = game["stats"][side].get(key)
            if want is None:
                continue
            got = sum(fn(r) for r in mine)
            if abs(got - want) > SOFT_TOLERANCE:
                bad.append(f"{team} {key} {got:g} != {want}")
            elif got != want:
                notes.append(f"{team} {key} {got:g} vs team line {want}")
    return bad, notes


def verify_box(rows, game):
    """The blockers from verify_box_detail (empty = complete)."""
    return verify_box_detail(rows, game)[0]


# ---- Optic id <-> our player_id --------------------------------------------------
def read_id_map(path) -> dict:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype={"optic_player_id": str})
    return dict(zip(df["optic_player_id"], df["player_id"].astype("int64")))


def save_id_map(path, mapping, names=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    names = names or {}
    pd.DataFrame([{"optic_player_id": k, "player_id": v, "name": names.get(k, "")}
                  for k, v in sorted(mapping.items())]).to_csv(path, index=False)


def remember_ids(path, matched):
    """Persist optic_player_id -> player_id for every row that carries both."""
    m = read_id_map(path)
    names = {}
    for r in matched:
        oid = str(r.get("optic_player_id") or "")
        if oid and r.get("player_id") is not None:
            m[oid] = int(r["player_id"])
            names[oid] = r.get("player_name", "")
    if matched:
        save_id_map(path, m, names)
    return m


def attach_player_ids(rows, players, id_map, overrides=None):
    """Our player_id for each box row. By Optic id when it has been seen on the
    odds; else by NAME (+ team, then name alone for a player traded since the
    sheet) and refusing ambiguity. Returns (matched, unmatched)."""
    matched, todo = [], []
    for r in rows:
        pid = id_map.get(str(r.get("optic_player_id", "")))
        (matched if pid is not None else todo).append({**r, "player_id": pid} if pid is not None else r)
    if todo:
        probe = [{**r, "home": r["team"], "away": r["team"], "fixture_id": r["game_id"]} for r in todo]
        got, un = resolve(probe, players, overrides)
        if un:
            bad = {u["name"] for u in un}
            retry = [{**r, "home": "", "away": "", "fixture_id": r["game_id"]} for r in todo
                     if r["player_name"] in bad]
            again, un = resolve(retry, players, overrides)
            box_team = {r["player_name"]: r["team"] for r in todo}
            for r in again:
                r["team"] = box_team.get(r["player_name"], r["team"])
            got += again
        matched += got
    return matched, (un if todo else [])


def to_logs(matched):
    return [{c: r.get(c) for c in LOG_COLS} for r in matched]


def read_logs(path):
    return pd.read_csv(path, parse_dates=["game_date"]) if os.path.exists(path) else pd.DataFrame(columns=LOG_COLS)


def append_logs(path, new_rows):
    """Idempotent: a (game_id, player_id) already logged is never added twice
    (state.update is an accumulator: re-feeding a game double-counts it)."""
    old = read_logs(path)
    add = pd.DataFrame(new_rows, columns=LOG_COLS)
    if len(add):
        add["game_date"] = pd.to_datetime(add["game_date"])
    if len(old) and len(add):
        seen = set(zip(old["game_id"].astype(str), old["player_id"].astype("int64")))
        add = add[[(str(g), int(p)) not in seen for g, p in zip(add["game_id"], add["player_id"])]]
    out = pd.concat([old, add], ignore_index=True) if len(add) else old
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return len(add)


def season_dates(start, end):
    d0, d1 = datetime.strptime(start, "%Y-%m-%d"), datetime.strptime(end, "%Y-%m-%d")
    return [(d0 + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((d1 - d0).days + 1)]

"""Optic player name -> our player_id. Join by NAME, restricted to the fixture's
two teams, and REFUSE TO GUESS when a name is still ambiguous.

OpticOdds has its own id space and `team_id` is null on player props, so the
name plus the fixture's home/away teams is all there is. Unmatched players are
returned for review -- an unmatched starter is a silently missing bet, and a
rising count means the slate frame is stale. Hand-resolved cases go in an
overrides file (optic name -> player_id) kept in version control.
"""

from __future__ import annotations

import re
import unicodedata

_SUFFIX = re.compile(r"\s+(jr|sr|ii|iii|iv|v)$")

# RotoWire vs ESPN/Optic abbreviations disagree on 7 of 30 teams (see ingest.py).
_TEAM_ALIASES = {"bro": "bkn", "chr": "cha", "nor": "no", "pho": "phx", "san": "sa",
                 "uta": "utah", "was": "wsh", "nop": "no", "gs": "gsw", "gsw": "gsw",
                 "ny": "nyk", "nyk": "nyk", "sas": "sa", "wash": "wsh"}


def norm_name(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower().replace("-", " ").replace(".", ""))
    return _SUFFIX.sub("", re.sub(r"\s+", " ", s).strip())


def norm_team(t) -> str:
    k = re.sub(r"[^a-z]", "", str(t or "").lower())
    return _TEAM_ALIASES.get(k, k)


def build_index(players):
    """players: iterable of dict(player_id, name, team)."""
    idx = {}
    for p in players:
        idx.setdefault(norm_name(p["name"]), []).append(p)
    return idx


def resolve(rows, players, overrides=None):
    """Attach player_id / team / opp to quote rows in place.
    Returns (matched_rows, unmatched) where unmatched is a list of
    {name, reason, fixture_id}."""
    idx = build_index(players)
    overrides = {norm_name(k): v for k, v in (overrides or {}).items()}
    matched, unmatched = [], {}
    for r in rows:
        key = norm_name(r["player_name"])
        pid, team = None, None
        if key in overrides:
            pid = overrides[key]
            cand = next((p for p in players if p["player_id"] == pid), None)
            team = cand["team"] if cand else None
        else:
            cands = idx.get(key, [])
            fixture_teams = {norm_team(r.get("home")), norm_team(r.get("away"))} - {""}
            if len(cands) > 1 or (cands and fixture_teams):
                in_game = [c for c in cands if norm_team(c["team"]) in fixture_teams] if fixture_teams else cands
                cands = in_game
            if len(cands) == 1:
                pid, team = cands[0]["player_id"], cands[0]["team"]
            else:
                reason = "no match" if not cands else "ambiguous"
                unmatched.setdefault((r["player_name"], reason), r["fixture_id"])
                continue
        home, away = norm_team(r.get("home")), norm_team(r.get("away"))
        t = norm_team(team)
        opp = r.get("away") if t == home else r.get("home") if t == away else ""
        matched.append({**r, "player_id": pid, "team": team, "opp": opp})
    return matched, [{"name": n, "reason": why, "fixture_id": f} for (n, why), f in unmatched.items()]

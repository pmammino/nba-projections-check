"""OpticOdds payloads -> paired two-sided market rows. Pure (no I/O).

OpticOdds returns each side of a market as its own record. De-vigging needs the
two sides back together, so records are grouped by (fixture, book, stat, player,
LINE) -- the line is in the key because an Over 24.5 de-vigged against an Under
25.5 describes a market that does not exist and always looks like free money.

Shape facts carried over from the NFL repo's live pulls (same v3 API):
  * no player-name field: `selection` IS the player on a prop (`name` is the
    full label "LeBron James Over 24.5" and is deliberately not a fallback)
  * `team_id` is null on player props; the parent fixture's two teams are what
    disambiguate a shared name
  * `timestamp` is epoch SECONDS (float)
  * `selection_line` holds the side word ("over"/"under") on O/U markets
"""

from __future__ import annotations

from .markets import STAT_DEFS, has_line, match_stat_key


def _first(obj, keys):
    if not isinstance(obj, dict):
        return None
    for k in keys:
        v = obj.get(k)
        if v is not None and v != "":
            return v
    return None


def _scalar(v, name_keys=("name", "display_name", "title")):
    if isinstance(v, dict):
        return _first(v, name_keys)
    return v


def _num(v):
    if v is None or v == "":
        return None
    try:
        n = float(v)
    except (TypeError, ValueError):
        return None
    return n if n == n and abs(n) != float("inf") else None


def _side_word(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("over", "yes", "o"):
        return "over"
    if s in ("under", "no", "u"):
        return "under"
    return None


def detect_side(raw):
    explicit = _scalar(_first(raw, ["selection_line", "side", "position"]))
    return _side_word(explicit) or _side_word(_scalar(_first(raw, ["selection", "name", "outcome"])))


def _abbr(competitors):
    c = competitors[0] if isinstance(competitors, list) and competitors else None
    return _first(c, ["abbreviation", "name"]) if c else None


def _competitor_map(fixture):
    m = {}
    for side in ("home_competitors", "away_competitors"):
        for c in fixture.get(side) or []:
            cid, ab = _first(c, ["id"]), _first(c, ["abbreviation", "name"])
            if cid and ab:
                m[str(cid)] = str(ab)
    return m


def flatten_payloads(payloads):
    """-> list of (raw_odd, context). Accepts bare odds, an envelope, or
    fixtures each carrying a nested `odds` array."""
    out = []
    for payload in payloads or []:
        rows = payload if isinstance(payload, list) else (
            payload.get("data") if isinstance(payload, dict) and isinstance(payload.get("data"), list)
            else [payload] if isinstance(payload, dict) else [])
        for row in rows:
            nested = row.get("odds") if isinstance(row, dict) else None
            if isinstance(nested, list):
                ctx = {
                    "fixture_id": _scalar(_first(row, ["id", "fixture_id", "game_id"])),
                    "home": _abbr(row.get("home_competitors")) or _scalar(_first(row, ["home_team_display", "home_team"])),
                    "away": _abbr(row.get("away_competitors")) or _scalar(_first(row, ["away_team_display", "away_team"])),
                    "start_date": _scalar(_first(row, ["start_date", "game_time", "start_time"])),
                    "team_by_id": _competitor_map(row),
                }
                out.extend((odd, ctx) for odd in nested)
            elif isinstance(row, dict):
                out.append((row, {}))
    return out


def read_odd(raw, ctx=None):
    ctx = ctx or {}
    if not isinstance(raw, dict):
        return None
    market = _scalar(_first(raw, ["market_id", "market", "market_name"]))
    # market_id is the exact id; fall back to the display name.
    stat = match_stat_key(market) or match_stat_key(_scalar(_first(raw, ["market", "market_name"])))
    pobj = raw.get("player") if isinstance(raw.get("player"), dict) else None
    pid = _scalar(_first(raw, ["player_id"]) or (_first(pobj, ["id"]) if pobj else None))
    pname = (_scalar(_first(raw, ["selection"])) or _scalar(_first(raw, ["normalized_selection"]))
             or (_scalar(pobj) if pobj else None) or _scalar(_first(raw, ["player_name"])))
    team_id = _scalar(_first(raw, ["team_id"]))
    team = ((ctx.get("team_by_id") or {}).get(str(team_id)) if team_id else None) \
        or _scalar(_first(raw, ["team", "team_abbreviation"]))
    limits = raw.get("limits") if isinstance(raw.get("limits"), dict) else {}
    return {
        "fixture_id": str(_scalar(_first(raw, ["fixture_id"])) or ctx.get("fixture_id") or ""),
        "sportsbook": str(_scalar(_first(raw, ["sportsbook", "sportsbook_name", "book"])) or ""),
        "market": "" if market is None else str(market),
        "stat": stat,
        "player_id": "" if pid is None else str(pid),
        "player_name": "" if pname is None else str(pname),
        "team_id": "" if team_id is None else str(team_id),
        "team": "" if team is None else str(team),
        "home": ctx.get("home"), "away": ctx.get("away"),
        "points": _num(_first(raw, ["points", "line", "handicap", "total"])),
        "price": _num(_first(raw, ["price", "odds", "american_odds"])),
        "side": detect_side(raw),
        "is_main": _first(raw, ["is_main", "main"]) is True,
        "timestamp": _first(raw, ["timestamp", "updated_at"]),
        "max_stake": _num(limits.get("max")) if limits else None,
        "start_date": ctx.get("start_date") or _scalar(_first(raw, ["start_date"])),
    }


def pair_odds(records, stats=None):
    """Group records into two-sided rows. Returns (rows, diagnostics).

    A row with only an Under is dropped (nothing to price against), as is a
    team entry inside a player market (no player_id, a team_id)."""
    allowed = set(stats) if stats else None
    groups, diag = {}, {"total": len(records), "unmatched_markets": {}, "missing_price": 0,
                        "missing_player": 0, "team_entries": 0, "no_side": 0}
    for rec in records:
        if not rec:
            continue
        if not rec["stat"] or (allowed and rec["stat"] not in allowed):
            k = rec["market"] or "(unnamed)"
            diag["unmatched_markets"][k] = diag["unmatched_markets"].get(k, 0) + 1
            continue
        if rec["price"] is None:
            diag["missing_price"] += 1
            continue
        if not rec["player_name"] and not rec["player_id"]:
            diag["missing_player"] += 1
            continue
        if not rec["player_id"] and rec["team_id"]:
            diag["team_entries"] += 1
            continue
        line = rec["points"] if has_line(rec["stat"]) else 0.5
        if line is None:
            diag["missing_price"] += 1
            continue
        side = rec["side"]
        if not side:
            if not has_line(rec["stat"]):
                side = "over"
            else:
                diag["no_side"] += 1
                continue
        pkey = rec["player_id"] or rec["player_name"].lower()
        key = (rec["fixture_id"], rec["sportsbook"], rec["stat"], pkey, line)
        g = groups.setdefault(key, {
            "fixture_id": rec["fixture_id"], "sportsbook": rec["sportsbook"], "stat": rec["stat"],
            "optic_player_id": rec["player_id"], "player_name": rec["player_name"],
            "team": rec["team"], "home": rec["home"], "away": rec["away"], "line": line,
            "over_odds": None, "under_odds": None, "timestamp": rec["timestamp"],
            "start_date": rec["start_date"], "max_stake": rec["max_stake"]})
        if not g["player_name"] and rec["player_name"]:
            g["player_name"] = rec["player_name"]
        if not g["team"] and rec["team"]:
            g["team"] = rec["team"]
        if side == "over":
            g["over_odds"] = rec["price"]
        else:
            g["under_odds"] = rec["price"]
    rows = []
    for g in groups.values():
        if g["over_odds"] is None and g["under_odds"] is None:
            continue
        # Unlike the NFL feed (overs only matter), NBA props are bet both ways,
        # so an Under-only quote is kept: book_votes mirrors it.
        g["one_sided"] = g["over_odds"] is None or g["under_odds"] is None
        rows.append(g)
    return rows, diag


def normalize_payloads(payloads, stats=None):
    flat = flatten_payloads(payloads)
    return pair_odds([read_odd(raw, ctx) for raw, ctx in flat], stats)

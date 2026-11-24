"""Capture one slot of one slate date, and (re)build the day's candidates.

Slots, matching the cadence asked for: `morning`, `pregame` (about two hours
before the first tip) and `closing` (as close to tip as the schedule allows).
`morning` and `pregame` are bettable boards; `closing` is the CLV reference.

The board (raw quotes) and the sim draws are the durable inputs. Candidates are
derived and rebuilt from them with the pricing model as it stood BEFORE that
date, so re-running never leaks a later day's information.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import crosswalk, feeds, store
from .books import books_to_fetch, resolve_book_ids
from .candidates import SLOT_ORDER, candidates_for_board
from .pricebook import get_book
from .teams import canon  # noqa: E402
from .consensus import estimate_hold_by_stat
from .markets import DEFAULT_STATS, optic_ids
from .model_fit import fit_model
from .normalize import normalize_payloads

SLOTS = tuple(SLOT_ORDER)


def slate_window(date: str):
    """UTC window covering one US slate day. NBA tips run ~17:00-03:00Z, so a
    date's slate is [date 10:00Z, date+1 10:00Z)."""
    d0 = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(hours=10)
    return d0.strftime("%Y-%m-%dT%H:%M:%SZ"), (d0 + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_start(f):
    try:
        return datetime.fromisoformat(str(f.get("start_date", "")).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def fixtures_in_window(client, date, status, log=None):
    """Fixtures of one slate day with the given status: [{id, start (datetime UTC), status}].

    The slate day is [date 10:00Z, date+1 10:00Z) so that US evening tips (which cross UTC
    midnight) belong to the right day. The API is asked for a WIDER, date-level range and the
    exact window is applied here: a live run returned last night's finished games for a
    time-of-day window, so the API's own handling of the time part is not trusted. When `log` is
    given it says what came back and why anything was dropped, so an empty slate explains itself."""
    lo = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(hours=10)
    hi = lo + timedelta(days=1)
    d0 = date
    d2 = (lo + timedelta(days=2)).strftime("%Y-%m-%d")
    try:
        raw = client.get_fixtures(start_date=d0, end_date=d2, status=status)
    except Exception as err:        # a status value the API rejects must not blind us: retry unfiltered
        if log:
            log(f"  fixtures: status={status!r} was rejected ({err}); retrying without it")
        raw = client.get_fixtures(start_date=d0, end_date=d2)
    out, dropped = [], {}
    for f in raw:
        why, start = None, _parse_start(f)
        if not f.get("id"):
            why = "no id"
        elif start is None:
            why = f"unparseable start_date={f.get('start_date')!r}"
        elif not lo <= start < hi:
            why = "outside the slate window"
        elif status == "unplayed" and f.get("is_live"):
            why = "live"
        elif status and str(f.get("status", "")).lower() not in (status, "scheduled", ""):
            why = f"status={f.get('status')}"
        if why:
            dropped[why] = dropped.get(why, 0) + 1
        else:
            out.append({"id": f["id"], "start": start, "status": f.get("status")})
    if log:
        log(f"  fixtures {d0}..{d2} status={status}: API returned {len(raw)}, {len(out)} in slate window "
            f"[{lo:%m-%d %H:%MZ}, {hi:%m-%d %H:%MZ})" + (f"; dropped {dropped}" if dropped else ""))
    return sorted(out, key=lambda x: x["start"])


def list_fixtures(client, date, log=None):
    """Unplayed fixtures for a slate date: [{id, start}]."""
    return fixtures_in_window(client, date, "unplayed", log)


def fetch_board_rows(client, fixture_ids, stats=None, books=None):
    """Live pull for specific fixtures -> paired quote rows (not yet crosswalked).
    NOT exercised against the live NBA feed from the build session."""
    live = client.get_sportsbooks()
    res = resolve_book_ids(books or books_to_fetch(), live)
    for w in res["missing"]:
        print(f"  warning: book '{w}' not found on the live list; skipped")
    for w, ids in res["ambiguous"].items():
        print(f"  warning: book '{w}' is ambiguous ({', '.join(ids)}); skipped")
    payloads = client.get_odds_for_fixtures(fixture_ids, res["ids"], optic_ids(stats or DEFAULT_STATS))
    rows, diag = normalize_payloads(payloads, stats or DEFAULT_STATS)
    return rows, diag


def save_slate_players(root, date, players):
    store.write_rows(store._path(root, "slates", f"{date}.csv"), players, ["player_id", "name", "team"])


def load_slate_players(root, date):
    raw = store.read_rows(store._path(root, "slates", f"{date}.csv"))
    return None if raw is None else [{**r, "player_id": int(r["player_id"])} for r in raw]


def merge_board(existing, new_rows, fixture_ids):
    """Replace the rows of `fixture_ids` in a board, keep every other fixture's."""
    fx = set(map(str, fixture_ids))
    return [r for r in (existing or []) if str(r["fixture_id"]) not in fx] + list(new_rows)


def capture(root, client, date, slot, fixture_ids, stats=None, books=None, overrides=None,
            price_model="both", session=None):
    """Capture `slot` for specific fixtures and merge into the day's board. Slots are per
    FIXTURE: a 10:00pm tip's pregame and closing are taken relative to ITS tip, not the
    first game's.

    price_model="feeds" or "both" (default): the RotoWire floor/median/ceiling feeds are fetched NOW
    and stored for these fixtures' teams, so the price reflects the feed as it stood at this
    capture (late minutes/injury news included). "both" additionally prices the combo markets from
    the day's simulator draws when they exist. "sim": the draws only (build them with `slate`)."""
    if slot not in SLOTS:
        raise ValueError(f"slot must be one of {SLOTS}")
    rows, diag = fetch_board_rows(client, fixture_ids, stats, books)
    if price_model in ("feeds", "both"):
        snap = feeds.fetch_snapshot(session)
        teams = {canon(r.get(k)) for r in rows for k in ("home", "away")} - {""}
        feeds.save_merged(root, date, slot, snap, teams)
        known = {p["player_id"]: p for p in (load_slate_players(root, date) or [])}
        known.update({p["player_id"]: p for p in feeds.players_of(snap)})
        save_slate_players(root, date, sorted(known.values(), key=lambda p: p["player_id"]))
    players = load_slate_players(root, date)
    if players is None:
        raise RuntimeError(f"No slate for {date}: build it first (it saves the players and draws).")
    matched, unmatched = crosswalk.resolve(rows, players, overrides)
    from . import results as _R
    _R.remember_ids(Path(root) / "inputs" / "optic_ids.csv", matched)
    store.save_board(root, date, slot, merge_board(store.load_board(root, date, slot), matched, fixture_ids))
    rebuild_candidates(root, date)
    return {"fixtures": len(fixture_ids), "quotes": len(rows), "matched": len(matched),
            "unmatched": unmatched, "diagnostics": diag}


def rebuild_candidates(root, date):
    """Derived: boards + each slot's price source + the model as of `date` ->
    candidates/{date}.csv. A slot priced from a feed snapshot uses THAT snapshot; one with
    none falls back to the day's simulator draws."""
    model, shares = fit_model(root, date)
    boards = {s: store.load_board(root, date, s) or [] for s in store.board_slots(root, date)}
    hold = estimate_hold_by_stat([{"stat": q["stat"], "hold": _hold(q)}
                                  for rows in boards.values() for q in rows])
    out = []
    for slot, rows in boards.items():
        book = get_book(root, date, slot)
        if book is None:
            continue
        out += candidates_for_board({"date": date, "slot": slot, "rows": rows}, book,
                                    model=model, hold_by_stat=hold, shares=shares)
    store.save_candidates(root, date, out)
    return len(out)


def _hold(q):
    from .odds import devig_two_way
    dv = devig_two_way(q.get("over_odds"), q.get("under_odds")) if q.get("over_odds") and q.get("under_odds") else None
    return dv["hold"] if dv else None

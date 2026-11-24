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

from . import crosswalk, store
from .books import books_to_fetch, resolve_book_ids
from .candidates import SLOT_ORDER, DrawBook, candidates_for_board
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


def fetch_board_rows(client, date, stats=None, books=None):
    """Live pull -> paired quote rows (not yet crosswalked). NOT exercised
    against the live NBA feed from the build session."""
    start, end = slate_window(date)
    fixtures = client.get_fixtures(start_date=start, end_date=end)
    fixture_ids = [f["id"] for f in fixtures if f.get("id") and not f.get("is_live")
                   and str(f.get("status", "")).lower() in ("unplayed", "scheduled", "")]
    live = client.get_sportsbooks()
    res = resolve_book_ids(books or books_to_fetch(), live)
    for w in res["missing"]:
        print(f"  warning: book '{w}' not found on the live list; skipped")
    for w, ids in res["ambiguous"].items():
        print(f"  warning: book '{w}' is ambiguous ({', '.join(ids)}); skipped")
    payloads = client.get_odds_for_fixtures(fixture_ids, res["ids"], optic_ids(stats or DEFAULT_STATS))
    rows, diag = normalize_payloads(payloads, stats or DEFAULT_STATS)
    return rows, diag, len(fixture_ids)


def save_slate_players(root, date, players):
    store.write_rows(store._path(root, "slates", f"{date}.csv"), players, ["player_id", "name", "team"])


def load_slate_players(root, date):
    raw = store.read_rows(store._path(root, "slates", f"{date}.csv"))
    return None if raw is None else [{**r, "player_id": int(r["player_id"])} for r in raw]


def capture(root, client, date, slot, stats=None, books=None, overrides=None):
    if slot not in SLOTS:
        raise ValueError(f"slot must be one of {SLOTS}")
    players = load_slate_players(root, date)
    if players is None:
        raise RuntimeError(f"No slate for {date}: run `slate` first (it saves the players and draws).")
    rows, diag, n_fixtures = fetch_board_rows(client, date, stats, books)
    matched, unmatched = crosswalk.resolve(rows, players, overrides)
    store.save_board(root, date, slot, matched)
    summary = {"fixtures": n_fixtures, "quotes": len(rows), "matched": len(matched),
               "unmatched": unmatched, "diagnostics": diag}
    rebuild_candidates(root, date)
    return summary


def rebuild_candidates(root, date):
    """Derived: boards + sim + the model as of `date` -> candidates/{date}.csv."""
    ids, sim = store.load_sim(root, date)
    if sim is None:
        return 0
    draws = DrawBook(sim, ids)
    model, shares = fit_model(root, date)
    boards = {s: store.load_board(root, date, s) or [] for s in store.board_slots(root, date)}
    hold = estimate_hold_by_stat([{"stat": q["stat"], "hold": _hold(q)}
                                  for rows in boards.values() for q in rows])
    out = []
    for slot, rows in boards.items():
        out += candidates_for_board({"date": date, "slot": slot, "rows": rows}, draws,
                                    model=model, hold_by_stat=hold, shares=shares)
    store.save_candidates(root, date, out)
    return len(out)


def _hold(q):
    from .odds import devig_two_way
    dv = devig_two_way(q.get("over_odds"), q.get("under_odds")) if q.get("over_odds") and q.get("under_odds") else None
    return dv["hold"] if dv else None

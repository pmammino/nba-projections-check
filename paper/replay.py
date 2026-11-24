"""Replay every persona over the captured candidates and write graded ledgers;
build the settled candidate pool the filter tab reads.

Deterministic and from scratch every run: a ledger is a pure function of
(candidates, actuals, persona rules), so changing a rule is a re-run. Bankroll
compounding is computed in date order from a known start, so correcting an
earlier day propagates forward instead of leaving the bankroll permanently wrong.

Bets are placed only at BET_SLOTS; the `closing` capture is reference-only (CLV
and the market's final word), never a price anyone bets.
"""

from __future__ import annotations

from . import store
from .candidates import BET_SLOTS, CLOSING_SLOT
from .consensus import DEFAULT_ASSUMED_HOLD, book_votes, median, sigmoid
from .markets import STAT_DEFS, stat_actual
from .odds import american_to_decimal, grade_outcome
from .paper_filter import apply_filters, summarize, verdict
from .personas import (PERSONAS, STARTING_BANKROLL_UNITS, advance_bankroll, simulate_day,
                       with_basis)

LEDGER_COLS = ["date", "persona", "slot", "player_id", "name", "team", "opp", "stat", "book", "line",
               "side", "odds", "implied", "fair", "our_prob", "edge", "model_edge", "p_sharp",
               "n_books", "stake_units", "scaled_by", "bankroll_before", "status", "actual",
               "pnl_units", "close_fair", "clv"]


def settle(bet, actuals_by_player, final_teams):
    """-> (status, actual, pnl). pending until the bet's team has a FINAL box score."""
    if bet.get("team") not in final_teams:
        return "pending", None, None
    box = actuals_by_player.get(bet["player_id"])
    if box is None or not (box.get("minutes") or 0) > 0:
        return "void", None, 0.0                    # DNP: book refunds the stake
    actual = stat_actual(box, bet["stat"])
    if actual is None:
        return "pending", None, None
    out = grade_outcome(actual, bet["line"], bet["side"])
    if out == "push":
        return "push", actual, 0.0
    stake = bet.get("stake_units", 1.0)
    pnl = stake * (american_to_decimal(bet["odds"]) - 1) if out == "won" else -stake
    return out, actual, round(pnl, 4)


def closing_index(board_rows, hold_by_stat=None):
    """Per (player, stat, line): book votes at the close, for CLV."""
    groups = {}
    for q in board_rows or []:
        if q.get("player_id") is None:
            continue
        groups.setdefault((int(q["player_id"]), q["stat"], q["line"]), []).append(q)
    return {k: book_votes(v, (hold_by_stat or {}).get(k[1], DEFAULT_ASSUMED_HOLD))
            for k, v in groups.items()}


def close_fair_for(index, bet, min_others=2):
    """The market's closing fair P(side) WITHOUT the book we bet at, or None
    when fewer than `min_others` other books quoted that exact line at the close."""
    votes = (index or {}).get((bet["player_id"], bet["stat"], bet["line"]))
    if not votes:
        return None
    others = [v for v in votes if v["book"] != bet["book"]]
    if len(others) < min_others:
        return None
    p_over = sigmoid(median(v["logit"] for v in others))
    return 1 - p_over if bet["side"] == "under" else p_over


def load_day(root, date):
    """-> (candidates, closing_index, actuals_by_player, final_teams)."""
    cands = store.load_candidates(root, date) or []
    close = closing_index(store.load_board(root, date, CLOSING_SLOT))
    act = store.load_actuals(root, date)
    by_player = {a["player_id"]: a for a in act} if act else {}
    final_teams = {a["team"] for a in act} if act else set()
    return cands, close, by_player, final_teams


def run_persona(root, persona, dates, basis="proj"):
    bankroll, rows = STARTING_BANKROLL_UNITS, []
    totals = {"bets": 0, "settled": 0, "staked": 0.0, "pnl": 0.0}
    for date in dates:
        cands, close, by_player, final_teams = load_day(root, date)
        priced = with_basis([c for c in cands if c["slot"] in BET_SLOTS], basis)
        bets = simulate_day(priced, persona, bankroll)
        before, day_pnl = bankroll, 0.0
        for b in bets:
            status, actual, pnl = settle(b, by_player, final_teams)
            cf = close_fair_for(close, b)
            if pnl is not None:
                day_pnl += pnl
                totals["settled"] += status in ("won", "lost", "push")
            totals["bets"] += 1
            totals["staked"] += b["stake_units"]
            totals["pnl"] += pnl or 0.0
            rows.append({**b, "persona": persona["id"], "bankroll_before": round(before, 4),
                         "status": status, "actual": actual, "pnl_units": pnl,
                         "close_fair": cf, "clv": None if cf is None else round(cf - b["implied"], 4)})
        bankroll = advance_bankroll(persona, bankroll, day_pnl)
    return rows, totals, bankroll


def write_ledgers(root, persona, rows):
    by_date = {}
    for r in rows:
        by_date.setdefault(r["date"], []).append(r)
    for date, rs in by_date.items():
        store.write_rows(store.ledger_path(root, persona["id"], date), rs, LEDGER_COLS)


def replay(root, dates=None, personas=None, basis="proj", write=True):
    dates = dates or store.all_dates(root)
    results = {}
    for p in personas or PERSONAS:
        rows, totals, bankroll = run_persona(root, p, dates, basis)
        if write:
            write_ledgers(root, p, rows)
        results[p["id"]] = {"persona": p, "rows": rows, "totals": totals, "bankroll": bankroll,
                            "roi": totals["pnl"] / totals["staked"] if totals["staked"] > 0 else None}
    return results


def settled_pool(root, dates=None):
    """Every candidate on every date with status/actual/close_fair attached --
    the dataset paper_filter.apply_filters works from."""
    out, complete = [], []
    for date in dates or store.all_dates(root):
        cands, close, by_player, final_teams = load_day(root, date)
        if final_teams:
            complete.append(date)
        for c in cands:
            if c["slot"] == CLOSING_SLOT:
                status, actual, cf = "pending", None, None
            else:
                status, actual, _ = settle({**c, "stake_units": 1.0}, by_player, final_teams)
                cf = close_fair_for(close, c)
            out.append({**c, "status": status, "actual": actual, "close_fair": cf})
    return out, complete


def filter_report(root, filters=None, dates=None):
    pool, complete = settled_pool(root, dates)
    bets = apply_filters(pool, filters)
    s = summarize(bets, complete)
    return {"bets": bets, "summary": s, "verdict": verdict(s)}

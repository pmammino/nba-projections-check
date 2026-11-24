"""Fit the pricing model (book weights, projection shares) from stored history,
strictly point-in-time: pricing date D only ever sees dates < D.

Nothing here needs outcomes except the projection shares; the book weights come
from the market alone (how much of each book's early disagreement the rest of
the market had adopted by the close), which converges far faster than win/loss.
"""

from __future__ import annotations

from . import store
from .candidates import BET_SLOTS, CLOSING_SLOT
from .pricebook import get_book
from .consensus import (DEFAULT_ASSUMED_HOLD, SHARE_BAND, book_votes, build_lead_pairs, logit,
                        pool_votes, shares_as_of, weights_as_of, weights_for_stat, _clip_logit)
from .pricing import win_prob


def quotes_for(root, dates):
    out = []
    for date in dates:
        for slot in store.board_slots(root, date):
            for r in store.load_board(root, date, slot) or []:
                if r.get("player_id") is None:
                    continue
                out.append({"date": date, "slot": slot, "player_id": int(r["player_id"]),
                            "stat": r["stat"], "line": r["line"], "book": r["sportsbook"],
                            "over_odds": r["over_odds"], "under_odds": r["under_odds"]})
    return out


def lead_pairs(root, dates):
    return build_lead_pairs(quotes_for(root, dates), late_slot=CLOSING_SLOT,
                            early_slots=list(BET_SLOTS))


def share_samples(root, dates, pairs=None):
    """One observation per market on a BET board: the books' pooled logit, our price's
    logit, and what happened. Weights are as of each market's own date, so a market priced
    on day 3 is judged against the pool as it stood. Works for either price source
    (feed snapshot or simulator draws)."""
    pairs = pairs if pairs is not None else lead_pairs(root, dates)
    wcache, samples = {}, []
    from .markets import stat_actual
    for date in dates:
        act = store.load_actuals(root, date)
        if not act:
            continue
        by_player = {a["player_id"]: a for a in act}
        for slot in BET_SLOTS:
            db = get_book(root, date, slot)
            if db is None:
                continue
            groups = {}
            for r in store.load_board(root, date, slot) or []:
                if r.get("player_id") is not None and db.priced(r["player_id"], r["stat"]):
                    groups.setdefault((int(r["player_id"]), r["stat"], r["line"]), []).append(r)
            for (pid, stat, line), rows in groups.items():
                box = by_player.get(pid)
                if not box or not (box.get("minutes") or 0) > 0:
                    continue
                actual = stat_actual(box, stat)
                if actual is None or actual == line:
                    continue
                votes = book_votes(rows, DEFAULT_ASSUMED_HOLD)
                if len(votes) < 3:
                    continue
                if date not in wcache:
                    wcache[date] = weights_as_of(pairs, date)
                pooled = pool_votes(votes, weights_for_stat(wcache[date], stat))
                if abs(pooled["prob"] - 0.5) > SHARE_BAND:
                    continue
                p_over = win_prob(db.draws(pid, stat), line, "over")
                samples.append({"stat": stat, "slot": slot, "date": date, "cluster": f"{date}|{pid}",
                                "l_books": _clip_logit(pooled["prob"]), "l_proj": _clip_logit(p_over),
                                "y": 1.0 if actual > line else 0.0})
    return samples


def fit_model(root, date):
    """-> (model, shares) as they stood going INTO `date`."""
    past = [d for d in store.all_dates(root) if d < date]
    pairs = lead_pairs(root, past)
    model = weights_as_of(pairs, date)
    shares = shares_as_of(share_samples(root, past, pairs), date)
    return model, shares

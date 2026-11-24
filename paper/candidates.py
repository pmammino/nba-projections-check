"""One captured board -> candidate bets with everything a filter needs.

A "board" is one capture (a slot such as morning / pregame / closing) of one
slate date: the paired two-sided quotes from every book, plus the simulator
draws for the players on it. Ported from the NFL repo's paper-data.mjs, with the
NFL fitted-distribution price replaced by the simulator's own draws.

Three prices per candidate, because they answer different questions:
  p_proj   what the SIMULATOR alone says (the price under evaluation)
  p_pool   the books' weighted consensus with the projection's earned share
  p_sharp  the sharp books' consensus alone (Pinnacle/Circa). An "edge" against
           it needs no projection: it is a price better than the sharpest part
           of the market thinks is fair.

Every row with an edge >= MIN_CANDIDATE_EDGE under ANY of the three is kept and
the choosing is done afterwards (paper_filter / personas), exactly so "only the
sharp books" or "ladders vs best rung" can be asked after the fact. The
collapse-to-best-price step belongs to the strategy, not the capture.
"""

from __future__ import annotations

from .books import canonical_book, is_bettable
from .consensus import (DEFAULT_ASSUMED_HOLD, book_fair_prob, book_votes, logit, median,
                        pool_votes, share_for, sigmoid, weights_for_stat)
from .odds import american_to_prob
from .pricing import win_prob

MIN_CANDIDATE_EDGE = 0.03
ODDS_FLOOR, ODDS_CEILING = -1000, 3000
BET_SLOTS = ("morning", "pregame")      # slots a bet may be placed at
CLOSING_SLOT = "closing"                # reference only: CLV and the market's final word
SLOT_ORDER = ["morning", "pregame", "closing"]


def _r(x, d=4):
    return None if x is None or x != x else round(x, d)


class DrawBook:
    """Simulator draws addressable by (player_id, stat).

    `sim` is game_sim's `return_draws="all"` dict (stat -> (n_sims, n_players));
    `player_ids` the column order (players["player_id"])."""

    def __init__(self, sim, player_ids):
        from .markets import stat_draws  # local: keep import graph flat
        self._stat_draws, self.sim = stat_draws, sim
        self.col = {int(p): i for i, p in enumerate(player_ids)}
        self._cache = {}

    def supports(self, stat) -> bool:
        from .markets import STAT_DEFS
        return all(part in self.sim for part in STAT_DEFS[stat]["parts"])

    def has(self, player_id) -> bool:
        try:
            return int(player_id) in self.col
        except (TypeError, ValueError):
            return False

    def draws(self, player_id, stat):
        key = (int(player_id), stat)
        if key not in self._cache:
            self._cache[key] = self._stat_draws(self.sim, stat)[:, self.col[int(player_id)]]
        return self._cache[key]


def candidates_for_board(board, draws, model=None, hold_by_stat=None, shares=None,
                         min_edge=MIN_CANDIDATE_EDGE, bettable=None):
    """board: {date, slot, rows}. rows are normalize.pair_odds rows that have
    been crosswalked (carry our `player_id`, plus `opp`). Returns candidate dicts."""
    date, slot = board["date"], board["slot"]
    hold_by_stat = hold_by_stat or {}
    quotes = [q for q in board["rows"] if q.get("player_id") not in (None, "") and draws.has(q["player_id"]) and draws.supports(q["stat"])]

    # Per market (player, stat, line): who quoted, what the rest of the market says.
    markets = {}
    for q in quotes:
        markets.setdefault((q["player_id"], q["stat"], q["line"]), []).append(q)
    info = {}
    for key, qs in markets.items():
        stat = key[1]
        hold = hold_by_stat.get(stat, DEFAULT_ASSUMED_HOLD)
        votes = book_votes(qs, hold)
        sharp = book_votes(qs, hold, set_name="sharp")
        info[key] = {
            "votes": votes, "n_books": len(votes), "n_sharp": len(sharp),
            "p_sharp_over": sigmoid(median(v["logit"] for v in sharp)) if sharp else None,
            "p_books_over": (pool_votes(votes, weights_for_stat(model, stat))["prob"]
                             if len(votes) >= 2 else None),
        }

    # Per player-stat ladder: the main line is the most-quoted (then nearest a
    # coin flip); `rung` is how many lines away each is.
    ladders = {}
    for key in info:
        ladders.setdefault((key[0], key[1]), []).append(key)
    for keys in ladders.values():
        keys.sort(key=lambda k: k[2])
        main = min(keys, key=lambda k: (-info[k]["n_books"],
                                        abs((info[k]["p_books_over"] or 0.5) - 0.5)))
        mi = keys.index(main)
        for i, k in enumerate(keys):
            info[k]["main_line"], info[k]["rung"] = main[2], i - mi

    rows = []
    for q in quotes:
        key = (q["player_id"], q["stat"], q["line"])
        mk = info[key]
        d = draws.draws(q["player_id"], q["stat"])
        p_over_cond = win_prob(d, q["line"], "over")
        share = share_for(shares, q["stat"], slot)
        w = weights_for_stat(model, q["stat"])
        pooled = pool_votes(mk["votes"], w, projection_prob=p_over_cond,
                            projection_share=share) if mk["votes"] else None
        hold = hold_by_stat.get(q["stat"], DEFAULT_ASSUMED_HOLD)
        for side in ("over", "under"):
            odds = q["over_odds"] if side == "over" else q["under_odds"]
            opp_odds = q["under_odds"] if side == "over" else q["over_odds"]
            implied = american_to_prob(odds)
            if odds is None or implied is None or odds < ODDS_FLOOR or odds > ODDS_CEILING:
                continue
            flip = (lambda p: None if p is None else (1 - p if side == "under" else p))  # noqa: E731
            p_proj = win_prob(d, q["line"], side)
            p_pool = flip(pooled["prob"]) if pooled else None
            p_sharp = flip(mk["p_sharp_over"])
            p_fair = flip(mk["p_books_over"])
            best = max([p - implied for p in (p_proj, p_pool, p_sharp) if p is not None] or [-1])
            if not best >= min_edge:
                continue
            fp = book_fair_prob(q["over_odds"], q["under_odds"], side, hold)
            book = canonical_book(q["sportsbook"])
            rows.append({
                "date": date, "slot": slot, "live": slot in BET_SLOTS or slot == CLOSING_SLOT,
                "fixture_id": q["fixture_id"], "player_id": int(q["player_id"]),
                "name": q["player_name"], "team": q.get("team", ""), "opp": q.get("opp", ""),
                "stat": q["stat"], "book": book, "bettable": is_bettable(book, bettable),
                "side": side, "line": q["line"], "odds": odds, "opposite_odds": opp_odds,
                "implied": _r(implied), "fair": _r(fp["fair"]) if fp else _r(implied),
                "hold": _r(fp["hold"]) if fp and fp["hold"] is not None else None,
                "one_sided": opp_odds is None,
                "p_proj": _r(p_proj), "p_pool": _r(p_pool), "p_sharp": _r(p_sharp),
                "p_fair": _r(p_fair), "n_books": mk["n_books"], "n_sharp": mk["n_sharp"],
                "main_line": mk["main_line"], "rung": mk["rung"],
                "max_stake": q.get("max_stake"), "captured_at": q.get("timestamp"),
                "game_start": q.get("start_date"),
            })
    return rows

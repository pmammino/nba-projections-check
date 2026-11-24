"""The engine behind filtered paper trading: filter a board's candidates the way
a bettor would, settle them, roll them up. Pure. Ported from the NFL repo's
paper-filter.mjs (the "Filtered Paper Trading" tab).

Controls, in the order a bettor applies them
  which boards   slots and dates
  which books    where a bet may be placed; reference books (Pinnacle) are
                 priced but cannot be staked unless asked for
  which bets     odds range, sides, stats, books quoting the exact line,
                 distance from the main line, an edge floor under one of three
                 prices
  sharp confirm  the sharp books' own fair price must not be worse than the
                 price taken. A market no sharp book quoted cannot be confirmed,
                 so it is DROPPED, not waved through
  shopping       one bet per market, at the best price among the books left
  ladder         "all": every qualifying rung. "best": one bet per player, stat
                 and side -- the rung with the biggest edge
  timing         "first": placed once, at the earliest capture where it
                 qualifies. "every": each capture is its own bet (counts the
                 same market up to twice)

The order is fixed: shop within a capture, pick the rung within a capture, THEN
take the first capture that has one. Picking the capture first would let the
2-hour-out price choose a bet the morning board never offered.

Stakes are one unit flat: the stake that makes ROI comparable across filters.
Kelly and budgets belong to the personas.
"""

from __future__ import annotations

import math

from .candidates import BET_SLOTS, SLOT_ORDER
from .odds import american_to_decimal

# NFL's team settled on -400..+200 for most personal betting; kept as default.
DEFAULT_ODDS_MIN, DEFAULT_ODDS_MAX = -400, 200
MIN_BETS_FOR_A_VERDICT = 200
BASES = ("proj", "pool", "sharp")

DEFAULT_FILTERS = {
    "slots": list(BET_SLOTS), "dates": None, "books": None, "bettable_only": True,
    "stats": None, "sides": "both", "odds_min": DEFAULT_ODDS_MIN, "odds_max": DEFAULT_ODDS_MAX,
    "basis": "proj", "min_edge": 0.03, "max_edge": None,
    "sharp_confirm": None,      # None = off; a number = sharp fair - price must be at least this
    "min_books": 3, "max_rung": None, "ladder": "best", "shop": True, "timing": "first",
}

PRESETS = [
    {"id": "realistic", "label": "Realistic", "filters": {},
     "description": "What a careful bettor would place: odds -400..+200, lines at least three books quote, one bet per player/stat at the best rung and price, placed at the first capture where it qualifies."},
    {"id": "sharp", "label": "Sharp books",
     "filters": {"books": ["fanduel", "draftkings", "pinnacle", "circasports"],
                 "bettable_only": False, "sharp_confirm": 0.0},
     "description": "Only FanDuel and DraftKings (plus Pinnacle and Circa as reference prices), and only bets the sharp books' own price does not contradict. The test of whether the simulator beats the hardest part of the market."},
    {"id": "ladder", "label": "Ladders",
     "filters": {"ladder": "all", "min_books": 1, "odds_min": -1000, "odds_max": 3000},
     "description": "Every qualifying rung. High volume and long odds; compare with Realistic."},
    {"id": "everything", "label": "Everything",
     "filters": {"ladder": "all", "shop": False, "timing": "every", "min_books": 1,
                 "odds_min": -1000, "odds_max": 3000},
     "description": "No filtering beyond a 3% edge: every book, rung and capture counted. The number to be most suspicious of."},
]


def preset_filters(preset_id):
    p = next((x for x in PRESETS if x["id"] == preset_id), None)
    return {**DEFAULT_FILTERS, **(p["filters"] if p else {})}


def edge_of(row, basis):
    p = {"pool": row.get("p_pool"), "sharp": row.get("p_sharp")}.get(basis, row.get("p_proj"))
    imp = row.get("implied")
    return None if p is None or imp is None else p - imp


def odds_in_range(american, lo, hi) -> bool:
    """Compared as decimal odds (continuous across even money), so bounds may be
    given in either order."""
    d, a, b = (american_to_decimal(x) for x in (american, lo, hi))
    if d is None or a is None or b is None:
        return False
    return min(a, b) - 1e-9 <= d <= max(a, b) + 1e-9


def _slot_rank(slot):
    return SLOT_ORDER.index(slot) if slot in SLOT_ORDER else len(SLOT_ORDER)


def passes_row(r, f) -> bool:
    if f["slots"] and r["slot"] not in f["slots"]:
        return False
    if f["dates"] and r["date"] not in f["dates"]:
        return False
    if f["bettable_only"] and not r["bettable"]:
        return False
    if f["books"] and r["book"] not in f["books"]:
        return False
    if f["stats"] and r["stat"] not in f["stats"]:
        return False
    if f["sides"] != "both" and r["side"] != f["sides"]:
        return False
    if not odds_in_range(r["odds"], f["odds_min"], f["odds_max"]):
        return False
    if r["n_books"] < f["min_books"]:
        return False
    if f["max_rung"] is not None and abs(r["rung"]) > f["max_rung"]:
        return False
    e = edge_of(r, f["basis"])
    if e is None or e < f["min_edge"] - 1e-12:
        return False
    if f["max_edge"] is not None and e > f["max_edge"] + 1e-12:
        return False
    if f["sharp_confirm"] is not None:
        if r.get("p_sharp") is None or r["p_sharp"] - r["implied"] < f["sharp_confirm"] - 1e-12:
            return False
    return True


def _best_per(rows, key_of, better):
    best = {}
    for r in rows:
        k = key_of(r)
        if k not in best or better(r, best[k]):
            best[k] = r
    return list(best.values())


def to_bet(r, basis):
    d = american_to_decimal(r["odds"])
    pnl = {"won": d - 1, "lost": -1.0, "push": 0.0, "void": 0.0}.get(r.get("status"))
    cf = r.get("close_fair")
    return {**r, "edge": edge_of(r, basis), "dec": d, "pnl": pnl,
            "clv": None if cf is None else cf - r["implied"]}


def apply_filters(rows, filters=None):
    f = {**DEFAULT_FILTERS, **(filters or {})}
    kept = [r for r in rows if passes_row(r, f)]
    edge = lambda r: edge_of(r, f["basis"])  # noqa: E731
    dec = lambda r: american_to_decimal(r["odds"])  # noqa: E731
    if f["shop"]:
        kept = _best_per(kept, lambda r: (r["slot"], r["date"], r["player_id"], r["stat"], r["line"], r["side"]),
                         lambda a, b: a["odds"] != b["odds"] and dec(a) > dec(b))
    if f["ladder"] == "best":
        kept = _best_per(
            kept, lambda r: (r["slot"], r["date"], r["player_id"], r["stat"], r["side"], "" if f["shop"] else r["book"]),
            lambda a, b: edge(a) > edge(b) + 1e-12 or (abs(edge(a) - edge(b)) <= 1e-12 and abs(a["rung"]) < abs(b["rung"])))
    if f["timing"] == "first":
        def key_of(r):
            base = (r["date"], r["player_id"], r["stat"], r["side"]) if f["ladder"] == "best" \
                else (r["date"], r["player_id"], r["stat"], r["line"], r["side"])
            return base + ("" if f["shop"] else r["book"],)
        first = {}
        for r in kept:
            k = key_of(r)
            first[k] = min(first.get(k, 99), _slot_rank(r["slot"]))
        kept = [r for r in kept if _slot_rank(r["slot"]) == first[key_of(r)]]
    bets = [to_bet(r, f["basis"]) for r in kept]
    bets.sort(key=lambda b: (b["date"], _slot_rank(b["slot"]), -(b["edge"] or 0)))
    return bets


def summarize(bets, complete_dates=()):
    """ROI, win rate, the band around the ROI, and CLV for a set of bets.

    `void` (player did not play: books refund it) is a push, not a loss. A bet
    still `pending` on a COMPLETE date has no box score to settle against and is
    counted as `no_line`, never silently as a win."""
    done = set(complete_dates)
    won = lost = push = void = unplayed = no_line = 0
    pnl, returns, clv, dec_sum, edge_sum = 0.0, [], [], 0.0, 0.0
    for b in bets:
        dec_sum += b["dec"] or 0
        edge_sum += b["edge"] or 0
        if b.get("clv") is not None:
            clv.append(b["clv"])
        st = b.get("status")
        if st in ("won", "lost"):
            won += st == "won"
            lost += st == "lost"
            pnl += b["pnl"]
            returns.append(b["pnl"])
        elif st == "push":
            push += 1
        elif st == "void":
            void += 1
        elif b["date"] in done:
            no_line += 1
        else:
            unplayed += 1
    n = len(returns)
    mean = pnl / n if n else None
    sd = math.sqrt(sum((r - mean) ** 2 for r in returns) / (n - 1)) if n > 1 else None
    cn = len(clv)
    cmean = sum(clv) / cn if cn else None
    csd = math.sqrt(sum((x - cmean) ** 2 for x in clv) / (cn - 1)) if cn > 1 else None
    return {
        "bets": len(bets), "settled": n, "won": won, "lost": lost, "push": push, "void": void,
        "unplayed": unplayed, "no_line": no_line, "pnl": pnl,
        "roi": pnl / n if n else None,
        "roi_band": 2 * sd / math.sqrt(n) if sd is not None else None,
        "per_bet_sd": sd, "win_rate": won / n if n else None,
        "avg_decimal": dec_sum / len(bets) if bets else None,
        "avg_edge": edge_sum / len(bets) if bets else None,
        "clv": ({"n": cn, "avg": cmean,
                 "avg_band": 2 * csd / math.sqrt(cn) if csd is not None else None,
                 "beat_rate": sum(1 for x in clv if x > 0) / cn,
                 "beat_band": 2 * math.sqrt(0.25 / cn)} if cn else None),
    }


def bets_to_resolve(roi, per_bet_sd):
    """Settled one-unit bets needed to see `roi` at two standard errors."""
    if not roi or not per_bet_sd:
        return None
    return math.ceil((2 * per_bet_sd / abs(roi)) ** 2)


def verdict(s, floor=MIN_BETS_FOR_A_VERDICT) -> str:
    if s["settled"] < floor:
        return "too few bets to call"
    if s["roi"] is None or s["roi_band"] is None:
        return "no ROI"
    if s["roi"] - s["roi_band"] > 0:
        return "positive"
    if s["roi"] + s["roi_band"] < 0:
        return "negative"
    return "indistinguishable from zero"


def group_summaries(bets, key_of, complete_dates=(), label=str):
    groups = {}
    for b in bets:
        groups.setdefault(key_of(b), []).append(b)
    out = [{"key": k, "label": label(k), "summary": summarize(v, complete_dates)} for k, v in groups.items()]
    return sorted(out, key=lambda g: (-g["summary"]["bets"], str(g["label"])))

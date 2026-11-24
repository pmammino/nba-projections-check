"""What does THE MARKET believe? Per-book fair probabilities, pooled in logit
space, with optional weights earned by how well each book's early price
predicted where the rest of the market ended up. Pure; ported from the NFL
repo's consensus.mjs + source-weights.mjs.

One-sided quotes (a book posting only the Over) cannot be de-vigged, so an
assumed hold is stripped instead and the vote is flagged (`two_sided` False) --
a one-sided Pinnacle vote is a weaker statement than a two-sided one and gets
its own track record.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import median as _median

from .books import RETAIL_BOOKS, SHARP_BOOKS, canonical_book, normalize_book
from .odds import DEFAULT_DEVIG_METHOD, american_to_prob, devig_two_way

MARKET_SETS = {"retail": RETAIL_BOOKS, "sharp": SHARP_BOOKS, "all": None}
DEFAULT_ASSUMED_HOLD = 0.07      # NBA props typically hold ~6-9%; refit with estimate_hold_by_stat
LOGIT_EPS = 1e-6
MAX_LOGIT = 6.0

PROJECTION_SOURCE = "projection"
PRIOR_WEIGHT = 0.15
SHARE_CAUTION = 1.0
MAX_PROJECTION_SHARE = 0.75
SHARE_BAND = 0.25
PRIOR_CLUSTERS = 150
MIN_WEIGHT = 0.01
MIN_OTHERS = 2
LEAD_BAND = 0.4
MAX_DEVIATION = 1.5


def logit(p: float) -> float:
    x = min(1 - LOGIT_EPS, max(LOGIT_EPS, p))
    return math.log(x / (1 - x))


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def median(xs):
    xs = list(xs)
    return float("nan") if not xs else float(_median(xs))


def _clip(x, lo, hi):
    return min(hi, max(lo, x))


def _clip_logit(p):
    return _clip(logit(p), -MAX_LOGIT, MAX_LOGIT)


def book_in_set(book, set_name: str) -> bool:
    if set_name == "all":
        return True
    members = MARKET_SETS.get(set_name)
    return members is not None and normalize_book(book) in members


def estimate_hold_by_stat(rows) -> dict:
    """Median observed hold per stat from two-sided rows (measured, never circular)."""
    by = defaultdict(list)
    for r in rows or []:
        h = r.get("hold")
        if h is not None and math.isfinite(h) and h > 0:
            by[r["stat"]].append(h)
    return {s: median(v) for s, v in by.items()}


def book_fair_prob(over_odds, under_odds, side, assumed_hold=None, method=DEFAULT_DEVIG_METHOD):
    """One book's fair P(side). Returns dict(fair, raw, hold, source) or None."""
    want_under = side == "under"
    own = under_odds if want_under else over_odds
    raw = american_to_prob(own)
    if raw is None:
        return None
    dv = None
    if over_odds is not None and under_odds is not None:
        dv = devig_two_way(over_odds, under_odds, method)
    if dv:
        return {"fair": dv["fair_under"] if want_under else dv["fair_over"],
                "raw": raw, "hold": dv["hold"], "source": "devig"}
    h = assumed_hold if (assumed_hold and assumed_hold > 0) else DEFAULT_ASSUMED_HOLD
    fair = raw / (1 + h)
    if not (0 < fair < 1):
        return None
    return {"fair": fair, "raw": raw, "hold": None, "source": "assumed-hold"}


def source_key(book, two_sided: bool) -> str:
    return f"{canonical_book(book)}:{2 if two_sided else 1}"


def book_votes(rows, assumed_hold=DEFAULT_ASSUMED_HOLD, method=DEFAULT_DEVIG_METHOD,
               set_name="all"):
    """One vote per book for one market: fair P(over) as a logit.
    `rows` are {book, over_odds, under_odds}. An Under-only quote is mirrored
    rather than dropped. First row per book wins."""
    seen, votes = set(), []
    for r in rows or []:
        raw_book = r.get("book") or r.get("sportsbook")   # captured rows say `sportsbook`
        if not book_in_set(raw_book, set_name):
            continue
        book = canonical_book(raw_book)
        if not book or book in seen:
            continue
        oo, uo = r.get("over_odds"), r.get("under_odds")
        under_only = oo is None and uo is not None
        fp = book_fair_prob(oo, uo, "under" if under_only else "over", assumed_hold, method)
        if not fp:
            continue
        p_over = 1 - fp["fair"] if under_only else fp["fair"]
        if not (0 < p_over < 1):
            continue
        seen.add(book)
        two = fp["source"] == "devig"
        votes.append({"book": book, "key": source_key(book, two), "two_sided": two,
                      "logit": _clip_logit(p_over), "hold": fp["hold"],
                      "assumed_hold": assumed_hold})
    return votes


def weight_of(weights, key):
    g = (weights or {}).get(key)
    return max(MIN_WEIGHT, g) if g is not None and math.isfinite(g) else PRIOR_WEIGHT


def pool_votes(votes, weights=None, projection_prob=None, projection_share=0.0):
    """Weighted logit mean of the books (each drawn in to the median by at most
    MAX_DEVIATION), plus the projection's earned share on top. None when no
    BOOK quoted: a projection can never create a consensus by itself."""
    if not votes:
        return None
    centre = median(v["logit"] for v in votes)
    sw = sl = sw2 = 0.0
    for v in votes:
        w = weight_of(weights, v["key"])
        l = _clip(v["logit"], centre - MAX_DEVIATION, centre + MAX_DEVIATION)
        sw += w
        sl += w * l
        sw2 += w * w
    l_books = sl / sw
    has_proj = projection_prob is not None and 0 < projection_prob < 1
    beta = _clip(projection_share if projection_share is not None and math.isfinite(projection_share) else 0.0,
                 0.0, MAX_PROJECTION_SHARE) if has_proj else 0.0
    l_proj = _clip(_clip_logit(projection_prob), centre - MAX_DEVIATION, centre + MAX_DEVIATION) if has_proj else l_books
    prob = sigmoid((1 - beta) * l_books + beta * l_proj)
    devigged = sum(1 for v in votes if v["two_sided"])
    return {"prob": prob, "books_prob": sigmoid(l_books), "book_count": len(votes),
            "devigged_count": devigged,
            "prob_source": "devig" if devigged == len(votes) else "mixed" if devigged else "assumed-hold",
            "effective_sources": (sw * sw) / sw2, "projection_share": beta,
            "unweighted": sigmoid(centre)}


def consensus_prob(rows, side="over", set_name="retail", hold_by_stat=None, stat=None,
                   method=DEFAULT_DEVIG_METHOD):
    """Simple (unweighted) consensus: median of per-book fair probabilities."""
    hold = (hold_by_stat or {}).get(stat, DEFAULT_ASSUMED_HOLD)
    votes = book_votes(rows, hold, method, set_name)
    if not votes:
        return None
    p_over = sigmoid(median(v["logit"] for v in votes))
    return {"prob": p_over if side == "over" else 1 - p_over, "book_count": len(votes)}


def weights_for_stat(model, stat):
    """Global weights overlaid by the stat's own."""
    out = dict((model or {}).get("weights", {}))
    out.update(((model or {}).get("stat_weights", {}) or {}).get(stat, {}))
    return out


# ---------------------------------------------------------------------------
# Assessment: how much of each source's early disagreement did the market adopt?
# ---------------------------------------------------------------------------
def build_lead_pairs(quotes, late_slot="closing", early_slots=None, hold_by_stat=None,
                     method=DEFAULT_DEVIG_METHOD, projection_of=None):
    """Group quotes into early/late pairs per market.

    `quotes`: dicts with date, slot, player_id, stat, line, book, over_odds,
    under_odds. `projection_of(q)` may return our P(over) for a quote (only
    where it was point-in-time), else None.
    """
    groups = {}
    for q in quotes:
        key = (q["date"], q["player_id"], q["stat"], q["line"])
        g = groups.setdefault(key, {"date": q["date"], "player_id": q["player_id"],
                                    "stat": q["stat"], "line": q["line"], "slots": {}})
        g["slots"].setdefault(q["slot"], []).append(q)
    pairs = []
    for g in groups.values():
        late_rows = g["slots"].get(late_slot)
        if not late_rows:
            continue
        hold = (hold_by_stat or {}).get(g["stat"], DEFAULT_ASSUMED_HOLD)
        late = book_votes(late_rows, hold, method)
        if not late:
            continue
        for slot, rows in g["slots"].items():
            if slot == late_slot or (early_slots and slot not in early_slots):
                continue
            early = book_votes(rows, hold, method)
            if not early:
                continue
            proj = projection_of(rows[0]) if projection_of else None
            pairs.append({"date": g["date"], "slot": slot, "stat": g["stat"],
                          "player_id": g["player_id"], "line": g["line"],
                          "cluster": f"{g['date']}|{g['player_id']}",
                          "early": early, "late": late, "projection": proj})
    return pairs


def _med_logit(votes):
    return median(v["logit"] for v in votes)


def lead_observations(pairs, min_others=MIN_OTHERS, band=LEAD_BAND):
    out = defaultdict(list)
    for p in pairs:
        for v in p["early"]:
            oe = [u for u in p["early"] if u["book"] != v["book"]]
            ol = [u for u in p["late"] if u["book"] != v["book"]]
            if len(oe) < min_others or len(ol) < min_others:
                continue
            lo = _med_logit(oe)
            if abs(sigmoid(lo) - 0.5) > band:
                continue
            out[v["key"]].append({"x": v["logit"] - lo, "y": _med_logit(ol) - lo, "cluster": p["cluster"]})
        pr = p.get("projection")
        if pr is not None and len(p["early"]) >= min_others and len(p["late"]) >= min_others:
            lo = _med_logit(p["early"])
            if abs(sigmoid(lo) - 0.5) <= band:
                out[PROJECTION_SOURCE].append({"x": _clip_logit(pr) - lo,
                                               "y": _med_logit(p["late"]) - lo,
                                               "cluster": p["cluster"]})
    return out


def lead_stats(obs):
    """Slope through the origin of y on x, errors clustered (CR1)."""
    n = len(obs)
    sxx = sum(o["x"] ** 2 for o in obs)
    sxy = sum(o["x"] * o["y"] for o in obs)
    if n < 2 or not sxx > 0:
        return {"n": n, "clusters": 0, "lead": None, "se": None, "z": None}
    lead = sxy / sxx
    by = defaultdict(float)
    for o in obs:
        by[o["cluster"]] += o["x"] * (o["y"] - lead * o["x"])
    G = len(by)
    if G < 2:
        return {"n": n, "clusters": G, "lead": lead, "se": None, "z": None}
    ss = sum(s * s for s in by.values())
    se = math.sqrt(ss * (G / (G - 1))) / sxx
    return {"n": n, "clusters": G, "lead": lead, "se": se, "z": lead / se if se > 0 else None}


def assess_sources(pairs, **opts):
    return {k: lead_stats(v) for k, v in lead_observations(pairs, **opts).items()}


def fit_source_weights(assessment, prior_clusters=PRIOR_CLUSTERS, prior_weight=PRIOR_WEIGHT):
    """w = (clusters*lead + K*prior)/(clusters + K); a negative lead reads as
    'no information', not 'do the opposite'."""
    weights, detail = {}, {}
    for key, s in assessment.items():
        if key == PROJECTION_SOURCE:
            detail[key] = {**s, "prior": 0, "weight": None}
            continue
        ev = max(0.0, s["lead"]) if s["lead"] is not None else prior_weight
        c = s["clusters"] or 0
        w = (c * ev + prior_clusters * prior_weight) / (c + prior_clusters)
        weights[key] = max(MIN_WEIGHT, w)
        detail[key] = {**s, "prior": prior_weight, "weight": weights[key]}
    return {"weights": weights, "detail": detail}


def fit_stat_weights(pairs, global_weights=None, prior_clusters=PRIOR_CLUSTERS,
                     prior_weight=PRIOR_WEIGHT):
    by = defaultdict(list)
    for p in pairs:
        by[p["stat"]].append(p)
    out = {}
    for stat, ps in by.items():
        w = {}
        for key, s in assess_sources(ps).items():
            if key == PROJECTION_SOURCE:
                continue
            base = (global_weights or {}).get(key, prior_weight)
            ev = max(0.0, s["lead"]) if s["lead"] is not None else base
            c = s["clusters"] or 0
            w[key] = max(MIN_WEIGHT, (c * ev + prior_clusters * base) / (c + prior_clusters))
        if w:
            out[stat] = w
    return out


def weights_as_of(pairs, date, **opts):
    """Weights going INTO `date`: fitted on strictly earlier dates only. A weight
    that has seen the slate it prices is grading with the answer in hand."""
    prior = [p for p in pairs if p["date"] < date]
    fit = fit_source_weights(assess_sources(prior), **opts)
    return {**fit, "stat_weights": fit_stat_weights(prior, fit["weights"], **opts),
            "dates_used": sorted({p["date"] for p in prior})}


# ---------------------------------------------------------------------------
# The projection's share of the price
# ---------------------------------------------------------------------------
def share_key(stat, slot):
    return f"{stat}|{slot}"


def fit_projection_share(obs, prior_clusters=PRIOR_CLUSTERS, cap=MAX_PROJECTION_SHARE,
                         caution=SHARE_CAUTION):
    """One-parameter penalised logistic fit  logit P = L_books + share*(L_proj - L_books).
    Prior 0, worth `prior_clusters` player-days. Returns the shrunk share marked
    down by `caution` SEs and floored at 0; `raw` keeps the sign so a report can
    say a cell is NEGATIVE (anti-informative) rather than just 'zero'."""
    n = len(obs)
    clusters = len({o["cluster"] for o in obs})
    if n == 0:
        return {"n": 0, "clusters": 0, "share": 0.0, "raw": 0.0, "se": None, "z": None}
    x = [o["l_proj"] - o["l_books"] for o in obs]

    def info(b):
        h = 0.0
        for o, xi in zip(obs, x):
            p = sigmoid(o["l_books"] + b * xi)
            h += p * (1 - p) * xi * xi
        return h

    lam = prior_clusters * info(0.0) / clusters if clusters else 0.0
    b = 0.0
    for _ in range(60):
        g = h = 0.0
        for o, xi in zip(obs, x):
            p = sigmoid(o["l_books"] + b * xi)
            g += (o["y"] - p) * xi
            h += p * (1 - p) * xi * xi
        g -= lam * b
        h += lam
        if not h > 0:
            break
        step = g / h
        b += step
        if abs(step) < 1e-9:
            break
    h = info(b)
    by = defaultdict(float)
    for o, xi in zip(obs, x):
        by[o["cluster"]] += (o["y"] - sigmoid(o["l_books"] + b * xi)) * xi
    G = len(by)
    ss = sum(v * v for v in by.values())
    se = math.sqrt(ss * (G / (G - 1))) / h if G > 1 and h > 0 else None
    used = 0.0 if se is None else _clip(b - caution * se, 0.0, cap)
    return {"n": n, "clusters": clusters, "share": used, "raw": b, "se": se,
            "z": b / se if se else None}


def fit_projection_shares(samples, **opts):
    cells = defaultdict(list)
    for o in samples:
        cells[share_key(o["stat"], o["slot"])].append(o)
    return {k: fit_projection_share(v, **opts) for k, v in cells.items()}


def shares_as_of(samples, date, **opts):
    return fit_projection_shares([o for o in samples if o["date"] < date], **opts)


def share_for(shares, stat, slot) -> float:
    return ((shares or {}).get(share_key(stat, slot)) or {}).get("share", 0.0)

"""Simulated bettors, and the engine that decides what each would have bet.
Pure -- no I/O. Ported from the NFL repo's personas.mjs.

WHAT THIS IS FOR, AND NOT FOR. The question behind paper trading is "would these
projections help someone find bets?", and the honest answer depends on WHO asks.
Nobody tails four hundred edges a day. So there are two instruments:

  THE CANDIDATE SET (every qualifying edge) is the statistical instrument: with
  thousands of bets its ROI band is tight enough to say whether the model is
  predictive. The Firehose persona IS that set, collapsed to best price.
  THE PERSONAS are the realism instrument. A persona taking 5 bets a day makes
  ~900 a season (2-sigma ROI band roughly +/-7%) -- they show what a strategy
  would have FELT like and cannot, on their own, settle whether the model works.

The unit of time is the SLATE DAY (NFL: the week), so per-week limits became
per-day limits and were rescaled for NBA volume.

Dimensions that separate bettors: books, max bets, selection rule, diversify
(stacking correlated props on one player/game), staking, bankroll, and
`sharp_confirm` (the sharp books must not contradict the bet).

`max_per_player` matters more here than in the NFL: points, PR, PA and PRA on
one player are one bet four times -- they win and lose together.
"""

from __future__ import annotations

import math

from .odds import american_to_decimal, expected_value, kelly_fraction

STARTING_BANKROLL_UNITS = 100.0   # 1 unit = 1% of the STARTING bankroll
STAKING_KINDS = ("flat", "kelly", "proportional", "tiered")


def _basis_edge(e, persona):
    return e.get("model_edge") if persona.get("edge_basis") == "novig" else e.get("edge")


def weekly_capacity(persona, bankroll_units):  # name kept for parity; unit is a day
    b = persona.get("bankroll") or {"kind": "flat"}
    if b["kind"] == "daily-budget":
        return b["units"]
    if b["kind"] == "compounding":
        return bankroll_units
    return math.inf


def eligible(edges, persona):
    books = {b.lower() for b in persona["books"]} if persona.get("books") else None
    stats = set(persona["stats"]) if persona.get("stats") else None
    out = []
    for e in edges:
        if e.get("bettable") is False:      # reference book: priced, never staked
            continue
        if books and str(e["book"]).lower() not in books:
            continue
        if stats and e["stat"] not in stats:
            continue
        if persona.get("require_two_sided") and e.get("one_sided"):
            continue
        edge = _basis_edge(e, persona)
        if edge is None or edge < persona.get("min_edge", 0):
            continue
        if american_to_decimal(e["odds"]) is None:
            continue
        if persona.get("min_limit") and e.get("max_stake") is not None and e["max_stake"] < persona["min_limit"]:
            continue
        sc = persona.get("sharp_confirm")
        if sc is not None:
            # No sharp quote => cannot be confirmed => dropped, not waved through.
            if e.get("p_sharp") is None or e["p_sharp"] - e["implied"] < sc - 1e-12:
                continue
        out.append(e)
    return out


def dedupe(edges, persona):
    """One bet per player-stat: the best price available to THIS persona (runs
    after the book filter, so a single-book persona gets the best price IT could
    reach -- which is what makes shopper-vs-single-book measure line shopping).
    Lines/sides/slots at one player-stat are alternatives, not additive."""
    if persona.get("dedupe") is False:
        return list(edges)
    best = {}
    for e in edges:
        k = (e["player_id"], e["stat"])
        cur = best.get(k)
        if cur is None or (_basis_edge(e, persona) or -math.inf) > (_basis_edge(cur, persona) or -math.inf):
            best[k] = e
    return list(best.values())


def rank(edges, persona):
    def score(e):
        if persona.get("select") == "top-ev":
            ev = expected_value(e["our_prob"], e["odds"])
            return -math.inf if ev is None else ev
        return _basis_edge(e, persona)
    # Stable tie-break so a re-run of the same day gives the same ledger.
    return sorted(edges, key=lambda e: (-score(e), -(_basis_edge(e, persona) or 0),
                                        str(e["player_id"]), e["stat"]))


def diversify(ranked, persona):
    d = persona.get("diversify") or {}
    max_game, max_player = d.get("max_per_game", math.inf), d.get("max_per_player", math.inf)
    max_bets = persona.get("max_bets_per_day", math.inf)
    per_game, per_player, taken = {}, {}, []
    for e in ranked:
        if len(taken) >= max_bets:
            break
        g = e.get("fixture_id") or f"{e.get('team')}|{e.get('opp')}"
        if per_game.get(g, 0) >= max_game or per_player.get(e["player_id"], 0) >= max_player:
            continue
        per_game[g] = per_game.get(g, 0) + 1
        per_player[e["player_id"]] = per_player.get(e["player_id"], 0) + 1
        taken.append(e)
    return taken


def _raw_stakes(bets, persona, bankroll):
    s = persona.get("staking") or {"kind": "flat", "units": 1}
    if s["kind"] == "flat":
        return [s.get("units", 1)] * len(bets)
    if s["kind"] == "kelly":
        base = bankroll if (persona.get("bankroll") or {}).get("kind") == "compounding" else STARTING_BANKROLL_UNITS
        return [min(kelly_fraction(e["our_prob"], american_to_decimal(e["odds"])) * s.get("fraction", 0.25),
                    s.get("cap", 0.03)) * base for e in bets]
    if s["kind"] == "tiered":
        tiers = s.get("tiers") or [{"min_edge": 0.10, "units": 3}, {"min_edge": 0.05, "units": 2},
                                   {"min_edge": 0, "units": 1}]
        return [next((t["units"] for t in tiers if _basis_edge(e, persona) >= t["min_edge"]), 1)
                for e in bets]
    if s["kind"] == "proportional":
        total = sum(max(0, _basis_edge(e, persona)) for e in bets)
        budget = (persona.get("bankroll") or {}).get("units", STARTING_BANKROLL_UNITS)
        if not total > 0:
            return [budget / max(1, len(bets))] * len(bets)
        return [max(0, _basis_edge(e, persona)) / total * budget for e in bets]
    raise ValueError(f"Unknown staking kind: {s['kind']}")


def stake(bets, persona, bankroll):
    """Scale proportionally if the day's capacity would be exceeded: that keeps
    the persona's SELECTION intact (we want to know how its choices performed,
    not how a budget ceiling truncated them)."""
    raw = _raw_stakes(bets, persona, bankroll)
    cap, total = weekly_capacity(persona, bankroll), sum(raw)
    scale = cap / total if math.isfinite(cap) and total > cap and total > 0 else 1.0
    return [{**e, "stake_units": round(r * scale, 4), "scaled_by": round(scale, 4)}
            for e, r in zip(bets, raw)]


def with_basis(rows, basis="proj"):
    """Attach our_prob / edge / model_edge for the chosen price basis.
    proj = the simulator alone; pool = books + earned projection share;
    sharp = sharp-book consensus. Rows without that price are dropped."""
    key = {"proj": "p_proj", "pool": "p_pool", "sharp": "p_sharp"}[basis]
    out = []
    for r in rows:
        p = r.get(key)
        if p is None:
            continue
        out.append({**r, "our_prob": p, "edge": p - r["implied"],
                    "model_edge": p - (r["fair"] if r.get("fair") is not None else r["implied"])})
    return out


def simulate_day(edges, persona, bankroll=STARTING_BANKROLL_UNITS):
    pool = dedupe(eligible(edges or [], persona), persona)
    return stake(diversify(rank(pool, persona), persona), persona, bankroll)


def advance_bankroll(persona, bankroll, day_pnl):
    if (persona.get("bankroll") or {}).get("kind") != "compounding":
        return bankroll
    return max(0.0, round(bankroll + day_pnl, 4))


PERSONAS = [
    {"id": "firehose", "label": "The Firehose",
     "description": "Every qualifying edge, one unit each, best price across all bettable books. Not a person: the statistical benchmark the others are measured against.",
     "min_edge": 0.03, "staking": {"kind": "flat", "units": 1}, "bankroll": {"kind": "flat"}},
    {"id": "disciplined", "label": "The Disciplined Flat Bettor",
     "description": "Five bets a day, one unit each, DraftKings only. Never two bets on one player. The most realistic recreational profile.",
     "books": ["draftkings"], "min_edge": 0.03, "max_bets_per_day": 5, "select": "top-edge",
     "diversify": {"max_per_game": 2, "max_per_player": 1},
     "staking": {"kind": "flat", "units": 1}, "bankroll": {"kind": "daily-budget", "units": 5}},
    {"id": "shopper", "label": "The Line Shopper",
     "description": "Same rules as the Disciplined Flat Bettor, but shops every bettable book for the best price. The gap between the two is what line shopping is worth.",
     "min_edge": 0.03, "max_bets_per_day": 5, "select": "top-edge",
     "diversify": {"max_per_game": 2, "max_per_player": 1},
     "staking": {"kind": "flat", "units": 1}, "bankroll": {"kind": "daily-budget", "units": 5}},
    {"id": "kelly", "label": "The Kelly Compounder",
     "description": "Quarter-Kelly on a bankroll that compounds all season, up to 12 bets a day across all books. Stakes grow after good runs and shrink after bad ones.",
     "min_edge": 0.03, "max_bets_per_day": 12, "select": "top-ev",
     "diversify": {"max_per_game": 3, "max_per_player": 1},
     "staking": {"kind": "kelly", "fraction": 0.25, "cap": 0.03}, "bankroll": {"kind": "compounding"}},
    {"id": "budget", "label": "The Daily Budget",
     "description": "Ten units a day, fully committed, spread across the top five edges in proportion to edge size. Bets the same amount every day however good the slate is.",
     "min_edge": 0.03, "max_bets_per_day": 5, "select": "top-edge",
     "diversify": {"max_per_game": 2, "max_per_player": 1},
     "staking": {"kind": "proportional"}, "bankroll": {"kind": "daily-budget", "units": 10}},
    {"id": "purist", "label": "The Purist",
     "description": "Only markets where both sides are quoted, so the price can be de-vigged and the edge measured against what the market believes. Ranked by that de-vigged disagreement.",
     "min_edge": 0.04, "require_two_sided": True, "edge_basis": "novig", "max_bets_per_day": 8,
     "select": "top-edge", "diversify": {"max_per_game": 2, "max_per_player": 1},
     "staking": {"kind": "kelly", "fraction": 0.25, "cap": 0.03}, "bankroll": {"kind": "compounding"}},
    {"id": "sharp", "label": "The Sharp Follower",
     "description": "The Line Shopper, but a bet is only placed when the sharp books' own fair price does not contradict it. Tests whether the simulator beats the hardest part of the market, not just the soft books.",
     "min_edge": 0.03, "sharp_confirm": 0.0, "max_bets_per_day": 5, "select": "top-edge",
     "diversify": {"max_per_game": 2, "max_per_player": 1},
     "staking": {"kind": "flat", "units": 1}, "bankroll": {"kind": "daily-budget", "units": 5}},
]
PERSONA_BY_ID = {p["id"]: p for p in PERSONAS}

"""Which books we price against and which we may stake at.

Same two-list design as the NFL repo: BETTABLE books can be the price on a bet;
REFERENCE books are captured to sharpen the fair-value consensus and are never
staked (betting a price nobody could take books a return nobody could earn).
"""

from __future__ import annotations

import re

BETTABLE_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars", "betrivers",
                  "hardrock", "thescore"]
# Circa is both bettable and sharp in the NFL repo. For NBA the user asked for
# Pinnacle + Circa as sharp references; Circa is kept OFF the bettable list so a
# sharp-reference price can never become a bet. Move it across if you hold an
# account there.
REFERENCE_BOOKS = ["pinnacle", "circa_sports"]

# Sharp set for the consensus (names normalised before matching).
SHARP_BOOKS = ["circa", "circasports", "circavegas", "pinnacle", "bookmaker", "betcris"]
RETAIL_BOOKS = ["draftkings", "fanduel", "betmgm", "mgm", "caesars", "betrivers",
                "hardrock", "hardrockbet", "thescore", "betr"]

_ALIASES = {"mgm": "betmgm", "hardrockbet": "hardrock", "circa": "circasports"}


def normalize_book(name) -> str:
    return "" if name is None else re.sub(r"[^a-z0-9]+", "", str(name).lower())


def canonical_book(name) -> str:
    k = normalize_book(name)
    return _ALIASES.get(k, k)


def books_to_fetch(bettable=None, reference=None):
    bettable = BETTABLE_BOOKS if bettable is None else bettable
    reference = REFERENCE_BOOKS if reference is None else reference
    seen = {normalize_book(b) for b in bettable}
    return list(bettable) + [b for b in reference if normalize_book(b) not in seen]


def is_bettable(book, bettable=None) -> bool:
    """Unknown books are NOT bettable (no account behind them)."""
    keys = {canonical_book(b) for b in (BETTABLE_BOOKS if bettable is None else bettable)}
    return canonical_book(book) in keys


def resolve_book_ids(requested, live_books):
    """Resolve human names to live OpticOdds ids. Exact normalised match wins,
    then a UNIQUE containment match; ambiguity and misses are reported, never
    guessed (an unknown id silently returns no odds)."""
    rows = []
    for b in live_books or []:
        if isinstance(b, str):
            rows.append({"id": b, "name": b})
        else:
            rows.append({"id": b.get("id") or b.get("name"), "name": b.get("name") or b.get("id")})
    rows = [r for r in rows if r["id"]]
    ids, matched, missing, ambiguous = [], {}, [], {}
    for want in requested or []:
        key = normalize_book(want)
        if not key:
            continue
        exact = [r for r in rows if normalize_book(r["id"]) == key or normalize_book(r["name"]) == key]
        hits = exact or [r for r in rows if key in normalize_book(r["id"]) or key in normalize_book(r["name"])]
        distinct = list({r["id"]: r for r in hits}.values())
        if not distinct:
            missing.append(want)
        elif len(distinct) > 1:
            ambiguous[want] = [r["id"] for r in distinct]
        else:
            matched[want] = distinct[0]["id"]
            ids.append(distinct[0]["id"])
    return {"ids": list(dict.fromkeys(ids)), "matched": matched, "missing": missing,
            "ambiguous": ambiguous}

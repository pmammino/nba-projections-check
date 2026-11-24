"""Odds maths and de-vigging. Pure, no I/O. Ported from the NFL repo's
odds.mjs / devig.mjs so both apps price a market identically."""

from __future__ import annotations

import math

DEVIG_METHODS = ("multiplicative", "additive", "power")
DEFAULT_DEVIG_METHOD = "multiplicative"


def is_american(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and abs(x) >= 100


def american_to_decimal(odds):
    if odds is None or not is_american(float(odds)):
        return None
    odds = float(odds)
    return 1 + odds / 100 if odds > 0 else 1 + 100 / -odds


def american_to_prob(odds):
    d = american_to_decimal(odds)
    return None if d is None else 1 / d


def kelly_fraction(p: float, decimal_odds: float) -> float:
    """Full-Kelly fraction of bankroll; 0 when there is no edge."""
    if decimal_odds is None or not (0 <= p <= 1):
        return 0.0
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    return max(0.0, (b * p - (1 - p)) / b)


def expected_value(p: float, american: float):
    """Expected profit per unit staked at the price actually available."""
    d = american_to_decimal(american)
    if d is None or not (p is not None and 0 <= p <= 1):
        return None
    return p * (d - 1) - (1 - p)


def _devig_mult(q):
    s = sum(q)
    return [x / s for x in q]


def _devig_add(q):
    e = (sum(q) - 1) / len(q)
    return [x - e for x in q]


def _devig_power(q):
    f = lambda k: sum(x ** k for x in q) - 1  # noqa: E731
    lo, hi, guard = 1.0, 2.0, 0
    while f(hi) > 0 and guard < 64:
        hi *= 2
        guard += 1
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2
    return [x ** k for x in q]


_IMPL = {"multiplicative": _devig_mult, "additive": _devig_add, "power": _devig_power}


def devig_probabilities(raw, method: str = DEFAULT_DEVIG_METHOD):
    if method not in _IMPL:
        raise ValueError(f"Unknown de-vig method: {method}")
    if len(raw) < 2 or not all(math.isfinite(p) and 0 < p < 1 for p in raw):
        return None
    if sum(raw) <= 1:  # no margin to remove (stale / crossed quote)
        return _devig_mult(raw)
    fair = _IMPL[method](list(raw))
    if not all(math.isfinite(p) and 0 < p < 1 for p in fair):
        return None
    return fair


def devig_two_way(over_odds, under_odds, method: str = DEFAULT_DEVIG_METHOD):
    """Returns dict(fair_over, fair_under, raw_over, raw_under, hold) or None."""
    ro, ru = american_to_prob(over_odds), american_to_prob(under_odds)
    if ro is None or ru is None:
        return None
    fair = devig_probabilities([ro, ru], method)
    if not fair:
        return None
    return {"fair_over": fair[0], "fair_under": fair[1], "raw_over": ro,
            "raw_under": ru, "hold": ro + ru - 1, "method": method}


def grade_outcome(actual: float, line: float, side: str) -> str:
    if actual == line:
        return "push"
    if side == "under":
        return "won" if actual < line else "lost"
    return "won" if actual > line else "lost"

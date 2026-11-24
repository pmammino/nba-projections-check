"""Turn simulator draws into probabilities. This is the NBA-specific heart of
the port: NFL fits a two-piece normal / Poisson to floor-median-ceiling; here the
simulator already gives the full distribution, so P(over) is just the share of
draws beyond the line, and combo markets (PRA) are coherent for free because
every stat in a draw comes from the same simulated game.
"""

from __future__ import annotations

import numpy as np


def prob_over(draws, line: float, smooth: bool = True):
    """P(win Over), P(push) for each column of `draws` (n_sims, n_players) or a
    1-d draw vector. Over wins strictly above the line; integer lines can push.

    Smoothing is a Jeffreys-style (k+0.5)/(n+1): with a few thousand draws a
    raw frequency of exactly 0 or 1 would be a claim of certainty the sample
    cannot support, and an infinite logit downstream."""
    d = np.asarray(draws, dtype="float64")
    n = d.shape[0]
    over = (d > line).sum(axis=0)
    push = (d == line).sum(axis=0)
    if smooth:
        p_over = (over + 0.5) / (n + 1)
        p_push = push / n
    else:
        p_over, p_push = over / n, push / n
    return p_over, p_push


def win_prob(draws, line: float, side: str = "over") -> float:
    """P(win) for one player's draw vector, conditional on no push.

    A push returns the stake, so the probability that matters for the bet is
    P(win | not push): comparable with the break-even probability of the price
    on both half-point and whole-number lines."""
    d = np.asarray(draws, dtype="float64")
    n = len(d)
    over = float((d > line).sum())
    under = float((d < line).sum())
    k = over if side == "over" else under
    other = under if side == "over" else over
    denom = k + other
    if denom == 0:
        return float("nan")
    # smooth the conditional frequency the same way
    return (k + 0.5) / (denom + 1)

"""A discrete distribution from what the feeds give us: a mean plus the 25th and 75th
percentiles. Pure numpy/math.

The feeds publish, per player and stat, an expectation (the "median" feed: fractional,
e.g. 0.93 assists) and integer 25th/75th percentiles (the floor/ceiling feeds). Counts
are over-dispersed relative to Poisson, so each stat is a NEGATIVE BINOMIAL with the
given mean; its dispersion is the one value of k whose quartiles reproduce the
published floor and ceiling. Many k can reproduce a pair of integers, so the middle
of the consistent range (in log k) is taken rather than an arbitrary edge of it.

ZERO INFLATION. A bench player is often either a non-factor (0 points) or a real
contributor, so "mean 4.6, 25th percentile 0, 75th percentile 8" (Aldama's real feed row)
is a distribution with a big mass at zero that no plain NB can produce: the fit misses
the quartiles by ~4 points of cdf. A zero-inflation term pi is therefore tried too, but
ONLY as far as it is needed: the smallest pi that reproduces the quartiles is used, and pi=0
whenever a plain NB already does. The mean is preserved either way.

A pair that no (k, pi) can reproduce falls back to the closest fit and is reported through
the returned error, so a feed that is inconsistent with itself is visible, not silently
priced.

Yes/no markets (double/triple double) are a Bernoulli with the published probability.
"""

from __future__ import annotations

import math

import numpy as np

K_GRID = np.geomspace(0.1, 5000.0, 56)          # ~15% steps: fine enough that a pair of quartiles is attainable
N_DRAWS = 2000


def nb_pmf(mean: float, k: float, upto: int) -> np.ndarray:
    """P(X = 0..upto) for NB with the given mean and size k (variance = mean + mean^2/k).
    k -> infinity is Poisson."""
    x = np.arange(upto + 1)
    if mean <= 1e-12:
        out = np.zeros(upto + 1)
        out[0] = 1.0
        return out
    p = k / (k + mean)
    logpmf = (np.array([math.lgamma(v + k) for v in x]) - math.lgamma(k) - np.array([math.lgamma(v + 1) for v in x])
              + k * math.log(p) + x * math.log1p(-p))
    pmf = np.exp(logpmf)
    return pmf / pmf.sum()


def _support(mean, hi) -> int:
    return int(max(25, math.ceil(mean * 4 + 6 * math.sqrt(max(mean, 1)) + 10), (hi or 0) * 3 + 10))


def _violation(cdf, lo, hi) -> float:
    """0 when `lo` is a valid 25th and `hi` a valid 75th percentile of this cdf."""
    def at(i):
        return cdf[min(max(i, 0), len(cdf) - 1)] if i >= 0 else 0.0
    v = 0.0
    for target, q in ((0.25, lo), (0.75, hi)):
        q = int(round(q))
        below, upto = at(q - 1), at(q)
        v += max(0.0, below - target) + max(0.0, target - upto)
    return v


def _nb_matrix(mean: float, ks: np.ndarray, n: int) -> np.ndarray:
    """pmf rows for every k at once, by the recurrence
    P(x+1) = P(x) * (x+k)/(x+1) * (1-p): n numpy ops instead of n*len(ks) lgamma calls."""
    p = ks / (ks + mean)
    P = np.empty((len(ks), n + 1))
    P[:, 0] = np.exp(ks * np.log(p))
    for x in range(n):
        P[:, x + 1] = P[:, x] * (x + ks) / (x + 1) * (1 - p)
    return P / P.sum(axis=1, keepdims=True)


PI_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6)
_FIT_CACHE: dict = {}


def fit_count(mean: float, lo: float, hi: float):
    """-> (pmf, k, error). `lo`/`hi` are the published 25th/75th percentiles. Cached on
    the (rounded) inputs: a slate repeats the same fits at every capture."""
    key = (round(float(mean), 6), float(lo), float(hi))
    if key not in _FIT_CACHE:
        _FIT_CACHE[key] = _fit_count(*key)
    return _FIT_CACHE[key]


def _fit_count(mean: float, lo: float, hi: float):
    mean = max(float(mean), 0.0)
    n = _support(mean, hi)
    if mean <= 1e-12:
        out = np.zeros(n + 1)
        out[0] = 1.0
        return out, float("inf"), 0.0
    best = None
    for pi in PI_GRID:
        # NB with mean m/(1-pi), then a point mass pi added at zero: overall mean stays m.
        P = _nb_matrix(mean / (1 - pi), K_GRID, n)
        P = (1 - pi) * P
        P[:, 0] += pi
        C = np.cumsum(P, axis=1)
        loss = np.zeros(len(K_GRID))
        for target, q in ((0.25, lo), (0.75, hi)):
            q = min(int(round(q)), n)
            below = C[:, q - 1] if q >= 1 else np.zeros(len(K_GRID))
            loss += np.maximum(0.0, below - target) + np.maximum(0.0, target - C[:, q])
        m = loss.min()
        if best is None or m < best[0] - 1e-9:      # strictly better only: the smallest pi that works wins
            ok = np.where(loss <= m + 1e-9)[0]
            i = ok[len(ok) // 2]                    # the middle of the consistent range
            best = (m, P[i], float(K_GRID[i]))
        if best[0] <= 1e-9:
            break
    return best[1], best[2], float(best[0])


def quantile_draws(pmf: np.ndarray, n: int = N_DRAWS) -> np.ndarray:
    """Deterministic pseudo-draws: the distribution's quantile function on an even
    grid. No RNG, so a re-run prices identically and the draws reproduce the pmf
    to within 1/n."""
    cdf = np.cumsum(pmf)
    u = (np.arange(n) + 0.5) / n
    return np.minimum(np.searchsorted(cdf, u, side="left"), len(pmf) - 1).astype("float64")


def bernoulli_draws(p: float, n: int = N_DRAWS) -> np.ndarray:
    p = min(max(float(p), 0.0), 1.0)
    u = (np.arange(n) + 0.5) / n
    return (u >= 1 - p).astype("float64")


def interval_prob(pmf: np.ndarray, lo: float, hi: float) -> float:
    """P(lo <= X <= hi) under the fitted distribution."""
    c = np.cumsum(pmf)
    a = c[int(lo) - 1] if lo >= 1 else 0.0
    return float(c[min(int(hi), len(c) - 1)] - a)

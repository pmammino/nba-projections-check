"""
Per-player floor and ceiling by simulation.

Floor = 25th percentile, ceiling = 75th, from a Monte Carlo over three sources
of uncertainty. They are separate on purpose, because they are separate
questions and they have very different sizes:

  1. ROLE      how many minutes he actually plays. Empirical, from 7,182 real
               backtest player-games. This dominates.
  2. RATE      how good he truly is per minute. The posterior Gamma/Beta
               variance, which the state already carries -- this is why
               state.py persists (alpha, beta) rather than the mean.
  3. SAMPLING  Binomial(minutes, rate) noise in a single game even if minutes
               and true rate were known exactly -- minutes as trial count,
               not Poisson(rate*minutes): see config.py's "Simulator
               calibration" section for why, and for a tail-shape knob
               (SIM_TAIL_GAMMA) that narrows the 05-95 band independently of
               the 25-75 one. ast/stl/blk/tov/reb still over-cover their
               25-75 band after all of this -- also documented there, as a
               known, investigated, unresolved limitation.

WHY SIMULATE RATHER THAN ADD QUANTILES. Fantasy points are a weighted sum
across stats, and the per-stat quantiles do NOT add: a big night is big across
the board because it runs through the same minutes draw. Summing marginal 75th
percentiles would overstate the ceiling badly. Drawing minutes ONCE per
iteration and reusing it for every stat induces the correct positive
correlation, which is the whole reason this is a simulation.

THE MINUTES DRAW IS EMPIRICAL, NOT PARAMETRIC. Measured residual behaviour by
projected-minutes bucket:

  proj mins     bias   sd   q10    q25   q75   q90   skew   P(act < 50% proj)
  0-8          +1.15  5.6  -4.5   -2.7  +3.0  +10.2  +1.27      24.3%
  8-14         -0.54  7.7  -9.7   -7.1  +4.8  +10.7  +0.56      32.5%
  14-20        -0.86  7.8 -11.7   -6.4  +4.5   +9.1  +0.15      19.2%
  20-26        -0.94  7.2 -10.2   -5.5  +3.9   +8.0  -0.23       7.8%
  26-32        -0.65  6.8  -9.4   -4.4  +3.9   +7.0  -0.82       3.8%
  32+          -4.59  7.7 -15.0   -9.6  +1.2   +4.4  -0.68       4.0%

Three things a Normal would get wrong, all of which a residual bootstrap gets
right for free:
  - SD is roughly CONSTANT at ~7 minutes, not proportional to the projection
    (power-law exponent 0.12). So relative uncertainty is ~106% of projection
    for a 5-minute player and ~21% for a 36-minute starter.
  - SKEW FLIPS SIGN. Low-minute players can only surprise upward; starters are
    capped above and can only lose minutes.
  - The left tail is fat. A 10-minute player finishes under half his projection
    a third of the time.

A KNOWN BIAS RIDES ALONG. The 32+ bucket is over-projected by ~3.4 minutes at
the median, and only 30.7% of those players reach their projection. Root cause
is not pinned down -- unprojected players absorbing minutes explains only part
of it. Because the residuals are sampled empirically, the simulated median
inherits the correction automatically, so the INTERVALS are honest even though
the underlying point projection is not yet fixed. That is a patch over a bug,
not a fix for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Projected-minutes buckets for the residual bootstrap.
MIN_BUCKETS = [0, 8, 14, 20, 26, 32, 100]

# Fallback residual quantile grid per bucket, measured on the 2025-26 backtest.
# Used when no residual table is supplied. Each row is the empirical residual
# distribution (actual minus projected minutes) at 11 evenly spaced quantiles.
# source: backtest_slates.py, 7,182 player-games over 45 slates.
MINUTES_RESID_Q = {
    0: [-5.3, -4.5, -3.2, -2.7, -1.4, -0.5, 1.2, 3.0, 6.1, 10.2, 18.0],
    1: [-11.6, -9.7, -8.2, -7.1, -3.6, -1.4, 1.6, 4.8, 7.6, 10.7, 20.0],
    2: [-17.3, -11.7, -8.5, -6.4, -3.0, -0.8, 1.7, 4.5, 6.8, 9.1, 19.0],
    3: [-23.1, -10.2, -7.6, -5.5, -2.5, -0.5, 1.6, 3.9, 6.0, 8.0, 16.0],
    4: [-28.9, -9.4, -6.6, -4.4, -1.7, 0.3, 2.1, 3.9, 5.5, 7.0, 13.0],
    5: [-36.2, -15.0, -11.7, -9.6, -6.1, -3.4, -1.1, 1.2, 2.9, 4.4, 9.0],
}
_QGRID = np.linspace(0.0, 1.0, 11)


def minutes_residual_table(backtest: pd.DataFrame,
                           proj_col: str = "minutes_proj",
                           act_col: str = "minutes_act") -> dict:
    """
    Build the residual quantile table from a real backtest.

    Prefer this over the baked-in fallback: it re-measures against your own
    minutes model, and the residuals are only valid for the model that produced
    them. Change the minutes model and this table is stale.
    """
    d = backtest[[proj_col, act_col]].dropna().copy()
    d["resid"] = d[act_col] - d[proj_col]
    d["b"] = pd.cut(d[proj_col], MIN_BUCKETS, labels=False,
                    include_lowest=True)
    out = {}
    for b, g in d.groupby("b"):
        if len(g) < 50:
            continue
        out[int(b)] = list(np.quantile(g["resid"], _QGRID))
    return out


def _warp_tail_u(u: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compress the uniform draw `u` outside [0.25, 0.75] toward that boundary,
    leaving u inside it untouched.

    `spread` rescales the WHOLE residual around its median, so it can
    calibrate the 25-75 band or the 05-95 band but not both if the empirical
    residual sample's true shape is more peaked (thinner tails relative to
    its own IQR) than the bootstrap implies -- which backtesting shows it is,
    for every stat, including ones whose 25-75 band is already exact. `gamma`
    is a second, independent knob for exactly that: u=0.25 and u=0.75 are
    fixed points of the warp, so it narrows p05/p95 without moving p25/p75.
    gamma=1.0 is the identity (no warp). gamma>1.0 pulls tail mass in.
    """
    if gamma == 1.0:
        return u
    out = u.copy()
    lo = u < 0.25
    hi = u > 0.75
    if lo.any():
        t = u[lo] / 0.25
        out[lo] = 0.25 * np.power(t, 1.0 / gamma)
    if hi.any():
        t = (u[hi] - 0.75) / 0.25
        out[hi] = 0.75 + 0.25 * np.power(t, 1.0 / gamma)
    return out


def _draw_minutes(proj: np.ndarray, n_sims: int, rng: np.random.Generator,
                  table: dict, spread: float,
                  tail_gamma: float = 1.0) -> np.ndarray:
    """
    Sample minutes by inverse-CDF on the bucket's empirical residuals.

    `spread` scales the residual around its own median, so interval width can
    be calibrated without moving the centre. `tail_gamma` separately narrows
    just the 05-95 tails (see `_warp_tail_u`); the backtested default lives at
    config.SIM_TAIL_GAMMA, applied by callers rather than defaulted here so
    this function stays a pure, literal transform.
    """
    buckets = np.clip(
        np.digitize(proj, MIN_BUCKETS[1:-1], right=True), 0, len(MIN_BUCKETS) - 2)
    out = np.empty((n_sims, len(proj)), dtype="float64")
    u = rng.random((n_sims, len(proj)))
    u = _warp_tail_u(u, tail_gamma)

    for b in np.unique(buckets):
        q = np.asarray(table.get(int(b), MINUTES_RESID_Q.get(
            int(b), MINUTES_RESID_Q[3])), dtype="float64")
        med = np.interp(0.5, _QGRID, q)
        cols = np.where(buckets == b)[0]
        resid = np.interp(u[:, cols], _QGRID, q)
        out[:, cols] = proj[cols] + med + spread * (resid - med)

    return np.clip(out, 0.0, config.MINUTES["min_ceiling"])


def simulate(proj: pd.DataFrame, n_sims: int = 4000, seed: int = 0,
             scoring: dict | None = None,
             minutes_table: dict | None = None,
             spread: float = config.SIM_MINUTES_SPREAD,
             rate_spread: float = config.SIM_RATE_SPREAD,
             tail_gamma: float = config.SIM_TAIL_GAMMA,
             include_sampling: bool = True) -> pd.DataFrame:
    """
    Simulate a projected slate and return per-player quantiles.

    `proj` is the output of project.project_slate -- it must still carry the
    per-stat rate columns and their `<stat>__sd` posterior standard deviations.

    Returns one row per player with, for each scoring stat and for fantasy
    points: p25 (floor), p50, p75 (ceiling), plus p05/p95 and a variance
    decomposition telling you how much of the band comes from role vs rate vs
    sampling.

    `spread`, `rate_spread`, and `tail_gamma` default to the backtested
    values in config.py (see the "Simulator calibration" section there for
    how they were tuned and what's still a known limitation).
    """
    rng = np.random.default_rng(seed)
    scoring = scoring or DK_NBA_SIM
    table = minutes_table or MINUTES_RESID_Q
    n = len(proj)
    if n == 0:
        return pd.DataFrame()

    m = _draw_minutes(proj["minutes_proj"].to_numpy(dtype="float64"),
                      n_sims, rng, table, spread, tail_gamma)

    # --- rate uncertainty -------------------------------------------------
    # Gamma for per-minute counts, Beta for conversion rates, matched to the
    # posterior mean and sd so the draw carries the state's own uncertainty.
    rate = {}
    for stat in config.VOLUME_RATES:
        mu = proj[stat].to_numpy(dtype="float64")
        sd = proj.get(f"{stat}__sd")
        mu = np.where(np.isfinite(mu) & (mu > 0), mu, 1e-6)
        if sd is None:
            rate[stat] = np.tile(mu, (n_sims, 1))
            continue
        sd = np.nan_to_num(sd.to_numpy(dtype="float64"), nan=0.0) * rate_spread
        sd = np.maximum(sd, 1e-9)
        shape = np.square(mu / sd)
        rate[stat] = rng.gamma(shape, mu / shape, size=(n_sims, n))

    for stat in config.SHARE_RATES:
        mu = np.clip(proj[stat].to_numpy(dtype="float64"), 1e-4, 0.999)
        sd = proj.get(f"{stat}__sd")
        if sd is None:
            rate[stat] = np.tile(mu, (n_sims, 1))
            continue
        sd = np.nan_to_num(sd.to_numpy(dtype="float64"), nan=0.0) * rate_spread
        var = np.clip(np.square(sd), 1e-12, mu * (1 - mu) * 0.999)
        k = mu * (1 - mu) / var - 1.0
        k = np.maximum(k, 1.0)
        rate[stat] = rng.beta(mu * k, (1 - mu) * k, size=(n_sims, n))

    # --- sampling noise within the game -----------------------------------
    # Binomial(n=minutes, p=rate), not Poisson(rate*minutes): Poisson's
    # Var=lambda is a hard floor, Binomial's Var=n*p*(1-p) sits strictly
    # below it whenever p>0, and minutes is a natural trial count (see
    # config.py's "Simulator calibration" section for the backtest that
    # motivated the switch, and its documented limits for low-rate stats).
    draws = {}
    n_trials = np.maximum(np.round(m), 0).astype("int64")
    for stat in config.VOLUME_RATES:
        raw = stat[:-3]
        if include_sampling:
            p = np.clip(np.nan_to_num(rate[stat], nan=0.0, posinf=0.0,
                                      neginf=0.0), 1e-6, 0.999)
            draws[raw] = rng.binomial(n_trials, p).astype("float64")
        else:
            draws[raw] = np.maximum(rate[stat] * m, 0.0)

    fga = draws["fga"]
    fta = draws["fta"]
    if include_sampling:
        # np.clip leaves NaN as NaN, and rng.binomial rejects NaN p outright.
        # A player with no attempts of a given type has an undefined
        # conversion rate, which is legitimate -- substitute the population
        # default for the draw rather than dropping the player.
        def _p(stat, default):
            v = np.nan_to_num(rate[stat], nan=default,
                              posinf=default, neginf=default)
            return np.clip(v, 0.0, 1.0)

        def _n(x):
            return np.nan_to_num(x, nan=0.0, posinf=0.0,
                                 neginf=0.0).clip(0).astype("int64")

        pt3a = rng.binomial(_n(fga), _p("pt3a_share", 0.35))
        fg2a = _n(fga - pt3a)
        pt3m = rng.binomial(_n(pt3a), _p("fg3_pct", 0.35))
        fg2m = rng.binomial(fg2a, _p("fg2_pct", 0.50))
        ftm = rng.binomial(_n(fta), _p("ft_pct", 0.77))
    else:
        pt3a = fga * rate["pt3a_share"]
        fg2a = fga - pt3a
        pt3m = pt3a * rate["fg3_pct"]
        fg2m = fg2a * rate["fg2_pct"]
        ftm = fta * rate["ft_pct"]

    sim = {
        "minutes": m,
        "pts": 2 * fg2m + 3 * pt3m + ftm,
        "reb": draws["oreb"] + draws["dreb"],
        "ast": draws["ast"], "stl": draws["stl"], "blk": draws["blk"],
        "tov": draws["tov"], "fga": fga, "pt3m": pt3m,
        "oreb": draws["oreb"], "dreb": draws["dreb"], "pf": draws["pf"],
    }
    sim["fp"] = sum(sim[k] * v for k, v in scoring.items() if k in sim)

    # Double-double needs the joint distribution -- E[f(X)] != f(E[X]) for an
    # indicator, so it can only be computed here and never from point means.
    cats = np.stack([sim["pts"], sim["reb"], sim["ast"],
                     sim["stl"], sim["blk"]])
    sim["dd"] = ((cats >= 10).sum(axis=0) >= 2).astype("float64")
    sim["td"] = ((cats >= 10).sum(axis=0) >= 3).astype("float64")

    # --- quantiles --------------------------------------------------------
    out = {"player_id": proj["player_id"].to_numpy()}
    for c in ("team", "position", "usage_tier", "minutes_proj"):
        if c in proj.columns:
            out[c] = proj[c].to_numpy()
    for k, v in sim.items():
        qs = np.quantile(v, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
        out[f"{k}_p05"], out[f"{k}_p25"] = qs[0], qs[1]
        out[f"{k}_p50"], out[f"{k}_p75"] = qs[2], qs[3]
        out[f"{k}_p95"] = qs[4]
        out[f"{k}_mean"] = v.mean(axis=0)
    res = pd.DataFrame(out)
    res["fp_floor"] = res["fp_p25"]
    res["fp_ceiling"] = res["fp_p75"]
    res["fp_iqr"] = res["fp_p75"] - res["fp_p25"]
    # Coefficient of variation on the band, which is what actually separates a
    # safe cash play from a tournament play.
    res["fp_iqr_rel"] = res["fp_iqr"] / res["fp_p50"].replace(0, np.nan)
    return res


def variance_decomposition(proj: pd.DataFrame, n_sims: int = 3000,
                           seed: int = 0, scoring: dict | None = None,
                           minutes_table: dict | None = None) -> pd.DataFrame:
    """
    How much of the fantasy-point band comes from role, rate and sampling?

    Re-runs the simulation with each source switched off in turn. Not an exact
    variance decomposition -- the sources interact, because rate uncertainty
    scales with the minutes draw -- so the shares are indicative and will not
    sum to exactly 100%. Reported as measured rather than normalized, so the
    interaction is visible instead of hidden.
    """
    full = simulate(proj, n_sims, seed, scoring, minutes_table)
    no_role = simulate(proj, n_sims, seed, scoring, minutes_table, spread=0.0)
    no_rate = simulate(proj, n_sims, seed, scoring, minutes_table,
                       rate_spread=1e-6)
    no_samp = simulate(proj, n_sims, seed, scoring, minutes_table,
                       include_sampling=False)

    v_full = np.square(full["fp_p75"] - full["fp_p25"])
    rows = []
    for lab, alt in (("role (minutes)", no_role), ("rate (posterior)", no_rate),
                     ("sampling (in-game)", no_samp)):
        v_alt = np.square(alt["fp_p75"] - alt["fp_p25"])
        rows.append({"source": lab,
                     "share_of_iqr_var": float(
                         np.nanmean(1 - v_alt / v_full.replace(0, np.nan)))})
    return pd.DataFrame(rows)


DK_NBA_SIM = {
    "pts": 1.0, "pt3m": 0.5, "reb": 1.25, "ast": 1.5,
    "stl": 2.0, "blk": 2.0, "tov": -0.5,
}

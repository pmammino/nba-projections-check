"""
Posterior state: build it from the preseason sheet, then update it incrementally
as games arrive.

THE KEY DESIGN DECISION lives here. We persist the conjugate parameters, not the
posterior mean:

    alpha0, beta0   the preseason prior, expressed as pseudo-evidence.
                    NEVER decays. This is what stops a 12-minute sample from
                    owning the projection.
    ex, em          decayed observed evidence (successes, exposure).
                    Decayed forward in time on every update.
    last_date       when ex/em were last decayed to.

Posterior mean = (alpha0 + ex/phi) / (beta0 + em/phi)

Storing it this way buys three things:
  1. the nightly update is O(new games), not a full refit
  2. posterior VARIANCE is free -- it feeds the range-of-outcomes sim directly,
     with no separate variance model
  3. prior_weight = beta0 / (beta0 + em/phi) is readable at any time, so
     "why didn't this projection move?" always has a numeric answer
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln

from . import config, rates as rates_mod

STATE_COLS = [
    "player_id", "stat", "kind", "alpha0", "beta0", "ex", "em", "last_date",
]


# ---------------------------------------------------------------------------
# Prior construction
# ---------------------------------------------------------------------------
def build_priors(preseason: pd.DataFrame, player_id_col: str = "newsid") -> pd.DataFrame:
    """
    Turn the preseason projection sheet into prior state.

    For a per-minute stat:   beta0 = PRIOR_STRENGTH (minutes)
                             alpha0 = rate * beta0
    For a conversion rate:   a0 + b0 = PRIOR_STRENGTH (attempts)
                             a0 = p * (a0 + b0)

    Players whose projected rate is undefined (0 projected minutes, or 0
    projected attempts for a conversion rate) get NO prior row for that stat.
    They will be picked up by the population fallback in posterior_rates().
    """
    df = rates_mod.normalize_columns(preseason, "preseason")
    rate_vec = rates_mod.to_rates(df)
    pid = df[player_id_col].astype("int64")

    # Unsigned / two-way players: the projection itself is more speculative,
    # so weaken the prior rather than pretend it is as good as a starter's.
    no_team = df["team"].isna() if "team" in df else pd.Series(False, index=df.index)
    strength_mult = np.where(no_team, config.NO_TEAM_PRIOR_MULT, 1.0)

    rows = []
    for stat in config.ALL_STATS:
        kind = "count" if stat in config.VOLUME_RATES else "binom"
        strength = config.PRIOR_STRENGTH[stat] * strength_mult
        r = rate_vec[stat].to_numpy(dtype="float64")

        ok = np.isfinite(r)
        if kind == "binom":
            ok &= (r >= 0) & (r <= 1)

        rows.append(
            pd.DataFrame(
                {
                    "player_id": pid.to_numpy()[ok],
                    "stat": stat,
                    "kind": kind,
                    # count:  alpha0 = lambda * beta0, beta0 in minutes
                    # binom:  alpha0 = p * N,          beta0 = (1-p) * N
                    "alpha0": (r[ok] * strength[ok]) if kind == "count"
                              else (r[ok] * strength[ok]),
                    "beta0": (strength[ok]) if kind == "count"
                             else ((1.0 - r[ok]) * strength[ok]),
                    "ex": 0.0,
                    "em": 0.0,
                    "last_date": pd.NaT,
                }
            )
        )

    state = pd.concat(rows, ignore_index=True)
    state["last_date"] = pd.to_datetime(state["last_date"])
    return state[STATE_COLS]


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------
def update(state: pd.DataFrame, logs: pd.DataFrame, as_of: pd.Timestamp,
           player_id_col: str = "player_id") -> pd.DataFrame:
    """
    Fold new game logs into the state and decay all evidence to `as_of`.

    `logs` must be one row per player-game with the canonical raw columns plus
    a `game_date`. Games already folded in must NOT be passed again -- this is
    an accumulator, not an idempotent recompute. Track the watermark with
    `state.last_date.max()`.

    CADENCE IS IRRELEVANT TO THE RESULT. The conjugate posterior depends only
    on the accumulated totals, so folding games in one at a time, in batches of
    four, or all at once gives bit-identical output (asserted in
    test_model.py). This matters because "we update every game, which is
    noisy" is a natural worry and the cadence is not where the noise is --
    batching or a weekly refresh changes nothing. What a rolling window would
    change is WHICH GAMES COUNT, and that was tested separately in
    compare_windows.py: every window variant was worse on accuracy and
    1.1x to 2.7x noisier game-to-game than full history.

    Decay is applied in two places:
      - existing ex/em are aged from their last_date to as_of
      - each new game's contribution is weighted by its own age at as_of
    """
    as_of = pd.Timestamp(as_of)
    state = state.copy()
    logs = logs.copy()
    logs["game_date"] = pd.to_datetime(logs["game_date"])

    if (logs["game_date"] > as_of).any():
        raise ValueError("logs contain games dated after as_of")

    ev = rates_mod.evidence_frame(logs)
    ev[player_id_col] = logs[player_id_col].to_numpy()
    ev["game_date"] = logs["game_date"].to_numpy()

    new_parts = []
    for stat in config.ALL_STATS:
        tau = config.DECAY_TAU_DAYS[stat]
        age_days = (as_of - ev["game_date"]).dt.total_seconds() / 86400.0
        w = np.exp(-age_days / tau)

        x = ev[f"{stat}__x"].to_numpy(dtype="float64")
        n = ev[f"{stat}__n"].to_numpy(dtype="float64")

        # A game with zero exposure carries zero evidence -- not evidence of
        # zero. NaN exposure (DNP) likewise.
        valid = np.isfinite(x) & np.isfinite(n) & (n > 0)

        part = pd.DataFrame(
            {
                "player_id": ev[player_id_col].to_numpy()[valid],
                "stat": stat,
                "ex_new": (w.to_numpy()[valid] * x[valid]),
                "em_new": (w.to_numpy()[valid] * n[valid]),
            }
        )
        new_parts.append(part.groupby(["player_id", "stat"], as_index=False).sum())

    new = pd.concat(new_parts, ignore_index=True)

    # Age the evidence already in state.
    has_prev = state["last_date"].notna()
    gap_days = (as_of - state["last_date"]).dt.total_seconds() / 86400.0
    tau_vec = state["stat"].map(config.DECAY_TAU_DAYS).astype("float64")
    decay = np.where(has_prev, np.exp(-gap_days.fillna(0.0) / tau_vec), 1.0)
    state["ex"] = state["ex"] * decay
    state["em"] = state["em"] * decay

    merged = state.merge(new, on=["player_id", "stat"], how="outer")

    # A player appearing in logs but not in the preseason sheet (call-up,
    # 10-day, undrafted) has no prior. Leave alpha0/beta0 NaN; posterior_rates
    # backfills from the population prior.
    merged["ex"] = merged["ex"].fillna(0.0) + merged["ex_new"].fillna(0.0)
    merged["em"] = merged["em"].fillna(0.0) + merged["em_new"].fillna(0.0)
    merged["last_date"] = as_of
    merged["kind"] = merged["stat"].map(
        lambda s: "count" if s in config.VOLUME_RATES else "binom"
    )
    return merged[STATE_COLS]


# ---------------------------------------------------------------------------
# Readout
# ---------------------------------------------------------------------------
def _mixture_mean(alpha0, beta0, ex, em, is_count, pop, kind_count):
    """
    Two-component mixture posterior mean, weights by exact marginal likelihood.

    Component 1 is the preseason prior at full strength; component 2 is diffuse
    and centred on the population rate. For Poisson data under Gamma(a, b) the
    joint marginal is, dropping factors common to both components,

        log ML = lgamma(a + X) - lgamma(a) + a*log(b) - (a + X)*log(b + N)

    and for Binomial data under Beta(a, b),

        log ML = lbeta(a + X, b + N - X) - lbeta(a, b).

    The dropped factors depend only on the data, so they cancel in the weight
    ratio -- the weights are exact, not approximated.

    See config.MIXTURE_PRIOR for what this is and is not worth.
    """
    M = config.MIXTURE_PRIOR
    w1 = M["weight_preseason"]
    frac = M["diffuse_strength_frac"]

    X = np.asarray(ex, dtype="float64")
    N = np.asarray(em, dtype="float64")
    a0 = np.asarray(alpha0, dtype="float64")
    b0 = np.asarray(beta0, dtype="float64")
    popv = np.asarray(pop, dtype="float64")

    strength = np.where(kind_count, b0, a0 + b0)
    bd = np.maximum(strength * frac, 1e-6)

    with np.errstate(all="ignore"):
        # component 1: the shipped prior
        m1 = np.where(kind_count, (a0 + X) / (b0 + N),
                      (a0 + X) / (a0 + b0 + N))
        l1 = np.where(
            kind_count,
            gammaln(a0 + X) - gammaln(a0) + a0 * np.log(b0)
            - (a0 + X) * np.log(b0 + N),
            betaln(a0 + X, b0 + np.maximum(N - X, 0.0)) - betaln(a0, b0),
        )
        # component 2: diffuse, centred on the population rate
        a2 = popv * bd
        b2 = np.where(kind_count, bd, (1.0 - popv) * bd)
        m2 = np.where(kind_count, (a2 + X) / (b2 + N),
                      (a2 + X) / (a2 + b2 + N))
        l2 = np.where(
            kind_count,
            gammaln(a2 + X) - gammaln(a2) + a2 * np.log(b2)
            - (a2 + X) * np.log(b2 + N),
            betaln(a2 + X, b2 + np.maximum(N - X, 0.0)) - betaln(a2, b2),
        )

        k1 = np.log(w1) + l1
        k2 = np.log(1.0 - w1) + l2
        mx = np.maximum(k1, k2)
        u1 = np.exp(k1 - mx)
        u2 = np.exp(k2 - mx)
        out = (u1 * m1 + u2 * m2) / (u1 + u2)
        wd = u2 / (u1 + u2)

    # No evidence yet means no conflict to detect: fall back to component 1,
    # which is the prior itself. Without this guard a player with zero exposure
    # would get a mixture of the prior and the population mean, quietly
    # changing every opening-night projection.
    no_data = ~np.isfinite(out) | (N <= 0)
    return np.where(no_data, m1, out), np.where(no_data, 0.0, wd)


def posterior_rates(state: pd.DataFrame,
                    population: pd.Series | None = None) -> pd.DataFrame:
    """
    Collapse state into a wide frame of posterior means, plus diagnostics.

    Returns one row per player with:
        <stat>          posterior mean rate
        <stat>__sd      posterior sd of the rate (for the sim)
        <stat>__pw      prior weight in [0, 1] -- 1.0 means untouched by data

    `population` is an optional stat -> rate Series used as a fallback prior
    for players with no preseason row. Pass the minutes-weighted league mean.
    """
    s = state.copy()
    phi = s["stat"].map(config.OVERDISPERSION).astype("float64")
    strength = s["stat"].map(config.PRIOR_STRENGTH).astype("float64")

    if population is not None:
        miss = s["alpha0"].isna()
        if miss.any():
            pop_rate = s.loc[miss, "stat"].map(population).astype("float64")
            # Weak population prior: a quarter of the normal strength, because
            # "league average" is a much worse guess than a scouted projection.
            pop_strength = strength[miss] * 0.25
            is_count = s.loc[miss, "kind"].eq("count")
            s.loc[miss, "alpha0"] = np.where(
                is_count, pop_rate * pop_strength, pop_rate * pop_strength
            )
            s.loc[miss, "beta0"] = np.where(
                is_count, pop_strength, (1.0 - pop_rate) * pop_strength
            )

    ex = s["ex"] / phi
    em = s["em"] / phi

    a = s["alpha0"] + ex
    is_count = s["kind"].eq("count")
    # count:  Gamma(a, b) with b = beta0 + exposure_minutes
    # binom:  Beta(a, b)  with b = beta0 + (trials - successes)
    b = np.where(is_count, s["beta0"] + em, s["beta0"] + (em - ex))

    mean = np.where(is_count, a / b, a / (a + b))
    var = np.where(
        is_count,
        a / np.square(b),
        (a * b) / (np.square(a + b) * (a + b + 1.0)),
    )
    prior_w = np.where(
        is_count,
        s["beta0"] / (s["beta0"] + em),
        (s["alpha0"] + s["beta0"]) / (s["alpha0"] + s["beta0"] + em),
    )

    s["mean"] = mean
    s["sd"] = np.sqrt(np.clip(var, 0.0, None))
    s["pw"] = np.clip(prior_w, 0.0, 1.0)

    if config.MIXTURE_PRIOR["enabled"]:
        pop_v = (s["stat"].map(population) if population is not None
                 else pd.Series(np.nan, index=s.index))
        # Without a population series there is nothing for the diffuse
        # component to centre on, so fall back to the single-component result
        # rather than inventing a centre.
        if pop_v.notna().any():
            pop_v = pop_v.fillna(s["mean"])
            mm, wd = _mixture_mean(s["alpha0"], s["beta0"], ex, em,
                                   is_count, pop_v, is_count.to_numpy())
            s["mean"] = np.where(np.isfinite(mm), mm, s["mean"])
            s["mix_w_diffuse"] = wd

    vals = ["mean", "sd", "pw"]
    short = {"sd": "sd", "pw": "pw"}
    if "mix_w_diffuse" in s.columns:
        vals.append("mix_w_diffuse")
        short["mix_w_diffuse"] = "mixw"
    wide = s.pivot_table(index="player_id", columns="stat", values=vals)
    wide.columns = [
        c[1] if c[0] == "mean" else f"{c[1]}__{short[c[0]]}"
        for c in wide.columns
    ]
    return wide.reset_index()


def population_rates(preseason: pd.DataFrame) -> pd.Series:
    """Minutes-weighted league-average rate per stat, for the cold-start path."""
    df = rates_mod.normalize_columns(preseason, "preseason")
    df = rates_mod.derive_components(df)
    out = {}
    for stat in config.VOLUME_RATES:
        raw = stat[:-3]
        out[stat] = df[raw].sum() / df["minutes"].sum()
    for stat, (num_col, den_col) in config.SHARE_RATES.items():
        out[stat] = df[num_col].sum() / df[den_col].sum()
    return pd.Series(out)


# ---------------------------------------------------------------------------
# Role-change reset
# ---------------------------------------------------------------------------
def reset_player(state: pd.DataFrame, player_id: int, mode: str = "soft",
                 new_prior: pd.Series | None = None) -> pd.DataFrame:
    """
    Handle a discontinuity -- a trade, a starter going down, a return from a
    long absence -- by resetting evidence rather than waiting for decay.

    This is the counterpart to the long tau values in config. Because decay is
    deliberately slow, this is the ONLY mechanism that lets the model react
    quickly to a genuine role change, and it has to be triggered externally by
    the news feed.

    mode="soft"   halve the accumulated evidence. Use for a trade or a rotation
                  change: the player is the same, the context is not.
    mode="hard"   zero the accumulated evidence. Use when the old sample is
                  actively misleading -- a return from a season-long injury, or
                  a position change.

    `new_prior` optionally replaces the preseason prior with an in-season
    revised projection, keyed by stat. Pass a revised RotoWire projection here
    rather than feeding it in as evidence: a revised projection is a new PRIOR,
    not an observation, and treating it as evidence would double-count the
    games it was already built from.
    """
    s = state.copy()
    m = s["player_id"] == player_id
    if not m.any():
        return s

    factor = {"soft": 0.5, "hard": 0.0}[mode]
    s.loc[m, "ex"] = s.loc[m, "ex"] * factor
    s.loc[m, "em"] = s.loc[m, "em"] * factor

    if new_prior is not None:
        for stat, rate in new_prior.items():
            row = m & s["stat"].eq(stat)
            if not row.any() or not np.isfinite(rate):
                continue
            strength = config.PRIOR_STRENGTH[stat]
            is_count = s.loc[row, "kind"].eq("count").all()
            s.loc[row, "alpha0"] = rate * strength
            s.loc[row, "beta0"] = strength if is_count else (1.0 - rate) * strength
    return s

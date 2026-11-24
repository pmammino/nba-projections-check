"""
Calibration and diagnostics for the hand-set parameters in config.

Three functions, in the order you will want them:

    implied_prior_floor()   run this FIRST and after every config edit. It is
                            pure arithmetic, needs no data, and catches the
                            beta0/tau conflict described in config. The initial
                            hand-set values failed this check.

    split_half_beta0()      the real estimator for beta0. Needs one season of
                            game logs. Replaces the hand-set PRIOR_STRENGTH.

    fit_elasticities()      the real estimator for ELASTICITY. Needs game logs
                            joined to historical spreads/totals. Replaces the
                            hand-set ELASTICITY table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, rates as rates_mod


# ---------------------------------------------------------------------------
# 1. Prior floor diagnostic -- no data required
# ---------------------------------------------------------------------------
def implied_prior_floor(mpg: float = 30.0, attempts_per_game: float = 10.0,
                        days_between_games: float = 2.0) -> pd.DataFrame:
    """
    What fraction of the projection will STILL be prior at the end of a season,
    given the current beta0/tau/phi settings?

    Exponential decay caps the total evidence weight at

        sum_k exp(-2k/tau) = 1 / (1 - exp(-days_between/tau))

    games' worth, no matter how many games are played. Multiply by exposure per
    game to get the asymptotic evidence, then the prior weight floor is

        beta0 / (beta0 + evidence/phi)

    A floor above ~0.35 for a high-volume stat means the prior will dominate
    all season and the model is not really learning. A floor near 0.02 means
    the prior is decorative and you have built a season-to-date average with
    extra steps.
    """
    rows = []
    for stat in config.ALL_STATS:
        b0 = config.PRIOR_STRENGTH[stat]
        tau = config.DECAY_TAU_DAYS[stat] * config.DECAY_SCALE
        phi = config.OVERDISPERSION[stat]
        is_count = stat in config.VOLUME_RATES

        eff_games = 1.0 / (1.0 - np.exp(-days_between_games / tau))
        per_game = mpg if is_count else attempts_per_game
        evidence = eff_games * per_game
        floor = b0 / (b0 + evidence / phi)

        # Half-life of the prior's dominance: games until pw crosses 0.5.
        n50 = phi * b0 / per_game
        rows.append({
            "stat": stat, "beta0": b0, "tau": round(tau), "phi": phi,
            "eff_games_cap": round(eff_games, 1),
            "games_to_pw_50": round(n50, 1),
            "pw_floor": round(floor, 3),
            "verdict": ("PRIOR DOMINATES" if floor > 0.35
                        else "prior decorative" if floor < 0.03
                        else "ok"),
        })
    return pd.DataFrame(rows).sort_values("pw_floor", ascending=False)


# ---------------------------------------------------------------------------
# 2. beta0 from split-half reliability
# ---------------------------------------------------------------------------
def split_half_beta0(logs: pd.DataFrame, n_splits: int = 40,
                     min_exposure: float = 300.0,
                     seed: int = 0) -> pd.DataFrame:
    """
    Estimate prior strength per stat from split-half reliability.

        beta0 = M * (1 - r) / r

    where r is the correlation between two random halves of a player's season
    and M is the per-half exposure. This is the number of minutes (or attempts)
    at which signal and noise carry equal weight -- exactly the quantity beta0
    is supposed to be.

    Randomize by GAME, not by possession, so within-game correlation (a player
    who is hot all night) is preserved in both halves rather than averaged
    away. Averaging it away inflates r and understates beta0.

    `logs`: one row per player-game with the canonical raw columns.
    Restricted to players clearing `min_exposure` so the correlation is not
    driven by garbage-time samples.
    """
    rng = np.random.default_rng(seed)
    ev = rates_mod.evidence_frame(logs)
    ev["player_id"] = logs["player_id"].to_numpy()

    out = []
    for stat in config.ALL_STATS:
        xcol, ncol = f"{stat}__x", f"{stat}__n"
        sub = ev[["player_id", xcol, ncol]].dropna()
        sub = sub[sub[ncol] > 0]
        if sub.empty:
            continue

        rs, ms = [], []
        for _ in range(n_splits):
            half = rng.random(len(sub)) < 0.5
            a = sub[half].groupby("player_id")[[xcol, ncol]].sum()
            b = sub[~half].groupby("player_id")[[xcol, ncol]].sum()
            j = a.join(b, how="inner", lsuffix="_a", rsuffix="_b")
            j = j[(j[f"{ncol}_a"] >= min_exposure / 2)
                  & (j[f"{ncol}_b"] >= min_exposure / 2)]
            if len(j) < 30:
                continue
            ra = j[f"{xcol}_a"] / j[f"{ncol}_a"]
            rb = j[f"{xcol}_b"] / j[f"{ncol}_b"]
            r = np.corrcoef(ra, rb)[0, 1]
            if np.isfinite(r) and r > 0:
                rs.append(r)
                ms.append((j[f"{ncol}_a"].mean() + j[f"{ncol}_b"].mean()) / 2.0)

        if not rs:
            continue
        r_bar = float(np.mean(rs))
        m_bar = float(np.mean(ms))
        out.append({
            "stat": stat,
            "r_split_half": round(r_bar, 3),
            "exposure_per_half": round(m_bar, 1),
            "beta0_est": round(m_bar * (1 - r_bar) / r_bar, 0),
            "beta0_config": config.PRIOR_STRENGTH[stat],
        })

    df = pd.DataFrame(out)
    if len(df):
        df["ratio"] = (df["beta0_est"] / df["beta0_config"]).round(2)
    return df


# ---------------------------------------------------------------------------
# 3. Market elasticities
# ---------------------------------------------------------------------------
def fit_elasticities(logs: pd.DataFrame, env: pd.DataFrame,
                     post_rates: pd.DataFrame) -> pd.DataFrame:
    """
    Fit the per-stat market elasticities that config currently hand-sets.

        log(E[X]) = log(rate_prior * exposure)          <- OFFSET, not a term
                  + b_pace * pace_mult
                  + b_own  * own_mult
                  + b_opp  * opp_mult
                  + b_home * is_home

    The offset is the point. Putting the player's own posterior rate in as an
    offset rather than a covariate means the fitted coefficients capture ONLY
    the environmental effect, not the player's baseline. Without it you would
    be re-fitting player skill and the coefficients would be meaningless.

    Fit by Poisson IRLS for counts and a log-link binomial for conversion
    rates. Implemented in numpy so this has no statsmodels dependency.

    Returns a frame directly comparable to config.ELASTICITY.
    """
    df = logs.merge(env, on=["game_id", "team"], how="inner")
    df = df.merge(post_rates, on="player_id", how="inner")
    ev = rates_mod.evidence_frame(df)

    design_cols = ["pace_mult", "own_mult", "opp_mult", "is_home"]
    X = df[design_cols].to_numpy(dtype="float64")

    rows = []
    for stat in config.ALL_STATS:
        if stat not in df.columns:
            continue
        y = ev[f"{stat}__x"].to_numpy(dtype="float64")
        n = ev[f"{stat}__n"].to_numpy(dtype="float64")
        base = df[stat].to_numpy(dtype="float64")

        ok = (np.isfinite(y) & np.isfinite(n) & np.isfinite(base)
              & (n > 0) & (base > 0))
        if ok.sum() < 200:
            continue
        offset = np.log(base[ok] * n[ok])
        beta = _poisson_irls(X[ok], y[ok], offset)

        rows.append({
            "stat": stat,
            **{f"b_{c}": round(float(b), 3) for c, b in zip(design_cols, beta)},
            "config": config.ELASTICITY.get(stat),
            "n": int(ok.sum()),
        })
    return pd.DataFrame(rows)


def _poisson_irls(X: np.ndarray, y: np.ndarray, offset: np.ndarray,
                  max_iter: int = 40, tol: float = 1e-8) -> np.ndarray:
    """Poisson GLM with a log link and a fixed offset, by IRLS."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = offset + X @ beta
        mu = np.exp(np.clip(eta, -30, 30))
        w = mu
        z = (y - mu) / np.maximum(mu, 1e-9)
        xtwx = X.T @ (w[:, None] * X)
        xtwz = X.T @ (w * z)
        step = np.linalg.solve(
            xtwx + 1e-8 * np.eye(X.shape[1]), xtwz
        )
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


# ---------------------------------------------------------------------------
# 4. Possession model
# ---------------------------------------------------------------------------
def fit_possession_model(team_box: pd.DataFrame, lines: pd.DataFrame,
                         leave_one_out: bool = True) -> dict:
    """
    Refit config.POSS_MODEL on real games.

    `team_box`: one row per team-game with fga, oreb, tov, fta, team, game_id.
    `lines`:    one row per team-game with game_id, total.

    The team pace prior is computed LEAVE-ONE-OUT by default: each game's prior
    excludes that game. Without this the prior contains the outcome it is being
    asked to predict, and the fitted coefficient on it is inflated -- the
    resulting model then looks good in-sample and underperforms live.

    Returns a dict shaped like config.POSS_MODEL, plus diagnostics.
    """
    box = team_box.copy()
    box["poss"] = box["fga"] - box["oreb"] + box["tov"] + 0.44 * box["fta"]

    if leave_one_out:
        agg = box.groupby("team")["poss"].agg(["sum", "count"])
        box = box.join(agg, on="team")
        box["pace_prior_team"] = (box["sum"] - box["poss"]) / (box["count"] - 1)
    else:
        box["pace_prior_team"] = box.groupby("team")["poss"].transform("mean")

    per_game = box.groupby("game_id").agg(
        poss_actual=("poss", "mean"),
        pace_prior=("pace_prior_team", "mean"),
    ).reset_index()

    tot = lines.groupby("game_id")["total"].first().reset_index()
    df = per_game.merge(tot, on="game_id").dropna()
    if len(df) < 50:
        raise ValueError(f"need >=50 games, got {len(df)}")

    y = df["poss_actual"].to_numpy()
    X = np.column_stack([
        np.ones(len(df)), df["pace_prior"].to_numpy(), df["total"].to_numpy()
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    # No-prior fallback, for early season.
    X2 = np.column_stack([np.ones(len(df)), df["total"].to_numpy()])
    beta2, *_ = np.linalg.lstsq(X2, y, rcond=None)

    return {
        "POSS_MODEL": {
            "intercept": round(float(beta[0]), 3),
            "b_pace_prior": round(float(beta[1]), 3),
            "b_total": round(float(beta[2]), 3),
        },
        "POSS_MODEL_NO_PRIOR": {
            "intercept": round(float(beta2[0]), 3),
            "b_pace_prior": 0.0,
            "b_total": round(float(beta2[1]), 3),
        },
        "POSS_LEAGUE_AVG": round(float(y.mean()), 1),
        "n_games": len(df),
        "r2": round(float(r2), 3),
        "mae": round(float(np.abs(pred - y).mean()), 2),
        "mae_league_constant": round(float(np.abs(y.mean() - y).mean()), 2),
    }

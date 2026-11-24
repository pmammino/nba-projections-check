"""
Turn spread and total into the environment terms the rate adjustment consumes.

The identity is exact. For a team with spread S (negative = favorite) and
game total T:

    implied_team_points = (T - S) / 2
    implied_opp_points  = (T + S) / 2

The TRAP is treating the total as a pace signal. Total = pace x combined
efficiency, and a single number cannot identify both. A 240 total is either a
fast game between mediocre offenses or a slow game between good ones.

So we anchor pace on a team-level pace PRIOR (from team game logs, decayed) and
attribute the market's residual to efficiency. That is an assumption, not a
derivation, and it is the weakest link in this module. Stated plainly so nobody
later mistakes it for a fact.

Three channels, deliberately routed to three different places:
  (a) pace       -> per-minute VOLUME rates          (this module -> project.py)
  (b) efficiency -> CONVERSION rates and scoring     (this module -> project.py)
  (c) |spread|   -> MINUTES, via garbage time        (minutes.py, NOT here)

(c) is the one people put in the wrong place. A 15-point spread does not make a
starter less efficient per minute; it makes him play four fewer minutes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def implied_points(spread: pd.Series, total: pd.Series) -> pd.DataFrame:
    """Exact decomposition of spread/total into per-side implied points."""
    spread = spread.astype("float64")
    total = total.astype("float64")
    return pd.DataFrame(
        {
            "team_pts": (total - spread) / 2.0,
            "opp_pts": (total + spread) / 2.0,
        }
    )


def game_environment(games: pd.DataFrame,
                     pace_prior: pd.Series | None = None) -> pd.DataFrame:
    """
    Build the environment frame.

    `games` needs: game_id, team, opp, spread, total, is_home, game_date.
    `pace_prior` maps team -> possessions per 48 min. Missing teams fall back to
    the league average, which makes the possessions term a no-op for them.

    Returns the input plus:
        team_pts, opp_pts    implied points per side
        poss_hat             expected possessions per 48 for this game
        own_ortg, opp_ortg   implied points per 100 possessions per side
        pace_mult, own_mult, opp_mult   log-space ratios vs league average
    """
    g = games.copy()
    pts = implied_points(g["spread"], g["total"])
    g["team_pts"] = pts["team_pts"]
    g["opp_pts"] = pts["opp_pts"]

    # Possessions from the fitted model: team pace prior AND the market total.
    # See config.POSS_MODEL for why the total belongs here rather than only in
    # the efficiency channel -- it predicts actual possessions better than a
    # team pace prior does.
    if pace_prior is None:
        mdl = config.POSS_MODEL_NO_PRIOR
        pp = pd.Series(0.0, index=g.index)
    else:
        own = g["team"].map(pace_prior).astype("float64")
        opp = g["opp"].map(pace_prior).astype("float64")
        # Both teams face the same possessions, so a game's pace is a blend.
        pp = (own + opp) / 2.0
        missing = pp.isna()
        pp = pp.fillna(config.POSS_LEAGUE_AVG)
        mdl = config.POSS_MODEL
        if missing.all():
            mdl = config.POSS_MODEL_NO_PRIOR
            pp = pd.Series(0.0, index=g.index)

    g["poss_hat"] = (
        mdl["intercept"]
        + mdl["b_pace_prior"] * pp
        + mdl["b_total"] * g["total"].astype("float64")
    )
    # Guard: a garbage line must not produce a negative or absurd pace.
    g["poss_hat"] = g["poss_hat"].clip(
        0.80 * config.POSS_LEAGUE_AVG, 1.25 * config.POSS_LEAGUE_AVG
    )

    g["own_ortg"] = 100.0 * g["team_pts"] / g["poss_hat"]
    g["opp_ortg"] = 100.0 * g["opp_pts"] / g["poss_hat"]

    g["pace_mult"] = np.log(g["poss_hat"] / config.POSS_LEAGUE_AVG)
    g["own_mult"] = np.log(g["own_ortg"] / config.ORTG_LEAGUE_AVG)
    g["opp_mult"] = np.log(g["opp_ortg"] / config.ORTG_LEAGUE_AVG)
    return g


def adjust_rates(rates: pd.DataFrame, env_row: pd.Series) -> pd.DataFrame:
    """
    Apply the environment to a frame of posterior rates for one game.

    log(rate_adj) = log(rate) + pace*pace_mult + own*own_mult
                              + opp*opp_mult + home*is_home

    Conversion rates are clipped to (0, 1) after adjustment -- the log-linear
    form does not respect that bound on its own, and a 0.62 3P% leaking through
    would poison the points reconstruction.
    """
    out = rates.copy()
    is_home = float(env_row.get("is_home", 0))

    for stat, (b_pace, b_own, b_opp, b_home) in config.ELASTICITY.items():
        if stat not in out.columns:
            continue
        log_adj = (
            b_pace * env_row["pace_mult"]
            + b_own * env_row["own_mult"]
            + b_opp * env_row["opp_mult"]
            + b_home * is_home
        )
        out[stat] = out[stat] * np.exp(log_adj)
        if stat in config.SHARE_RATES:
            out[stat] = out[stat].clip(1e-4, 0.999)

    return out


def team_pace_prior(team_logs: pd.DataFrame, as_of: pd.Timestamp,
                    tau_days: float = 30.0) -> pd.Series:
    """
    Decayed team pace estimate, possessions per 48 minutes.

    Possessions estimated with the standard box-score formula:
        POSS = FGA - OREB + TOV + 0.44 * FTA

    `team_logs` needs: team, game_date, minutes, fga, oreb, tov, fta.
    The 0.44 FTA coefficient is itself an estimate and drifts a little with
    rule changes; it is the conventional value.
    """
    t = team_logs.copy()
    t["game_date"] = pd.to_datetime(t["game_date"])
    age = (pd.Timestamp(as_of) - t["game_date"]).dt.total_seconds() / 86400.0
    w = np.exp(-age / tau_days)

    poss = t["fga"] - t["oreb"] + t["tov"] + 0.44 * t["fta"]
    per48 = 48.0 * poss / (t["minutes"] / 5.0)

    num = (w * per48).groupby(t["team"]).sum()
    den = w.groupby(t["team"]).sum()
    return num / den

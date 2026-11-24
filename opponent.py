"""
Opponent profile layer: a stat-specific opponent adjustment.

WHY THIS IS A SEPARATE LAYER AND NOT A FILLED-IN COLUMN IN ELASTICITY.

config.ELASTICITY's opponent coefficient multiplies ONE SCALAR per game --
log(opponent implied ORtg / league), derived from spread and total. That scalar
cannot distinguish "this team allows blocks" from "this team allows assists".
It only knows the opponent is good or bad overall, and nine of thirteen stats
carried a zero there.

The fix is not a bigger number in that column, because the input is wrong. It
is a different input: the opponent's empirical stat-allowed rate for THIS stat,
shrunk toward league average. These are genuinely stat-specific -- across the
nine team profiles the first principal component explains only 38% of variance,
and FGA-allowed correlates -0.27 with TOV-allowed. One factor does not cover
it.

Fitted out of sample (Poisson, zeros included, posterior as offset, profiles
built walk-forward, train through 2026-02-11 and scored after). For the nine
stats that previously had NO opponent term:

    stat          market scalar   stat profile   coefficient
    fta_pm            +1.18%         +1.91%          0.65
    fga_pm            +1.00%         +1.19%          0.81
    pt3a_share        +0.20%         +0.93%          0.93
    fg2_pct           +0.43%         +0.85%          1.05
    tov_pm            +0.39%         +0.77%          0.95
    ast_pm            -0.37%         +0.41%          0.92
    fg3_pct           +0.02%         +0.09%            --   not shipped
    ft_pct            -0.20%         -0.14%            --   not shipped
    oreb_pm           -0.55%         -0.56%            --   not shipped

Eight of nine prefer the profile over the market scalar. The coefficients
cluster near 1.0, which is the clean reading: the opponent's allowed rate
transfers to the player almost one-for-one.

Three stats are deliberately left out. FT% is the one I would have predicted --
free throws are unguarded, and the measured response is negative. 3P% is noise.
Offensive rebounds come out negative, which I cannot explain; it is left off
rather than shipped on the assumption it will come good.

HETEROGENEITY WORTH KNOWING. The gain is not uniform across players, and the
pattern differs by stat type. For dense stats it is largest for LOW-rate
players (FTA: +14.0% in the bottom quartile, +0.5% in the top); for sparse
count stats it is largest for HIGH-rate players and NEGATIVE at the bottom
(blocks: -2.4% bottom, +1.6% third quartile). Shipped uniformly because the
overall effect is positive for every stat included, but `min_tier_rate` is
there if you want to gate it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Pseudo-exposure of the league prior, in the stat's own denominator units.
# Raw defence-versus-position style numbers are notoriously noisy; this is what
# stops a team with twelve games from showing an extreme multiplier.
SHRINK = 1500.0

STAT_SPEC = {
    "fga_pm": ("fga", "minutes"), "fta_pm": ("fta", "minutes"),
    "oreb_pm": ("oreb", "minutes"), "dreb_pm": ("dreb", "minutes"),
    "ast_pm": ("ast", "minutes"), "stl_pm": ("stl", "minutes"),
    "blk_pm": ("blk", "minutes"), "tov_pm": ("tov", "minutes"),
    "pf_pm": ("pf", "minutes"),
    "pt3a_share": ("pt3a", "fga"), "fg2_pct": ("fg2m", "fg2a"),
    "fg3_pct": ("pt3m", "pt3a"), "ft_pct": ("ftm", "fta"),
}


def build_profiles(logs: pd.DataFrame, as_of: pd.Timestamp | None = None,
                   shrink: float = SHRINK) -> pd.DataFrame:
    """
    Opponent stat-allowed multipliers, one row per opponent team.

    `logs` must be player game logs with `opp`, `game_date` and the raw stat
    columns. Only games STRICTLY BEFORE `as_of` are used -- a profile that
    includes tonight's game is predicting its own input.

    Returns a frame indexed by team with one multiplier column per stat, where
    1.0 means exactly league average. Teams with no history come back at 1.0,
    so the layer is a no-op for them rather than a guess.
    """
    d = logs
    if as_of is not None:
        d = d[d["game_date"] < pd.Timestamp(as_of)]
    d = d[~d["dnp"]] if "dnp" in d.columns else d
    if d.empty:
        return pd.DataFrame()

    d = d.copy()
    if "fg2a" not in d:
        d["fg2a"] = d["fga"] - d["pt3a"]
    if "fg2m" not in d:
        d["fg2m"] = d["fgm"] - d["pt3m"]

    out = {}
    for stat, (num, den) in STAT_SPEC.items():
        if num not in d or den not in d:
            continue
        lg_n, lg_d = d[num].sum(), d[den].sum()
        if lg_d <= 0 or lg_n <= 0:
            continue
        lg = lg_n / lg_d
        g = d.groupby("opp").agg(n=(num, "sum"), m=(den, "sum"))
        out[stat] = ((g["n"] + lg * shrink) / (g["m"] + shrink)) / lg

    prof = pd.DataFrame(out)
    prof.index.name = "team"
    return prof


def adjust_rates_for_opponent(rates: pd.DataFrame, opponent: str,
                              profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the opponent profile multiplicatively:

        rate_adj = rate * mu_opponent ** beta_stat

    Same functional form as the market and absence layers, so all three
    compose and none needs to know about the others.

    Only the stats in config.OPPONENT_PROFILE["beta"] move. A stat absent from
    that table was measured and did not earn a term -- see the module docstring
    for which and why.
    """
    out = rates.copy()
    if profiles is None or profiles.empty or opponent not in profiles.index:
        return out

    row = profiles.loc[opponent]
    for stat, beta in config.OPPONENT_PROFILE["beta"].items():
        if stat not in out.columns or stat not in row.index:
            continue
        mu = float(row[stat])
        if not np.isfinite(mu) or mu <= 0:
            continue
        mu = float(np.clip(mu, config.OPPONENT_PROFILE["mu_clip"][0],
                           config.OPPONENT_PROFILE["mu_clip"][1]))
        out[stat] = out[stat] * (mu ** beta)
        if stat in config.SHARE_RATES:
            out[stat] = out[stat].clip(1e-4, 0.999)
    return out


def profile_report(profiles: pd.DataFrame, stat: str, n: int = 5) -> pd.DataFrame:
    """Toughest and softest opponents for one stat, for eyeballing."""
    if stat not in profiles.columns:
        return pd.DataFrame()
    s = profiles[stat].sort_values()
    return pd.concat([s.head(n).rename("multiplier").to_frame().assign(end="toughest"),
                      s.tail(n).rename("multiplier").to_frame().assign(end="softest")])

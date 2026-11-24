"""
Availability: adjusting the remaining players when someone is ruled out.

THE ORGANIZING IDEA, and it is the same one that makes the NFL share pipeline
work: absences barely change a team's TOTALS. They change the SHARES.

Team possessions are set by pace, not by who is dressed. Team minutes are 240
either way. Team defensive rebounds are set by how often the opponent misses.
So when PG1 is ruled out, his usage does not evaporate -- it redistributes, and
the modeling question is only *how unevenly*.

That gives a clean division of labour, and it is what keeps this module from
double-counting against the rest of the pipeline:

    absence betas  ->  RELATIVE redistribution weights  (this module)
    conservation   ->  the LEVEL                        (minutes.py, project.py)

The betas below say a player's FGA rate rises 2.0% per 10 points of absent
rotation usage. Applied alone that would inflate team attempts. But
project.py already reconciles team volume to the market's implied possessions,
and minutes.py already normalizes to 240. So the betas move players relative to
each other and the existing constraints set the total. Nothing is applied
twice.

All coefficients measured on all 1,235 games of 2025-26 (32,261 player-games)
from ESPN box scores plus DNP flags. See fit_absence.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Usage load and absent share
# ---------------------------------------------------------------------------
def usage_load(post: pd.DataFrame, minutes_basis: pd.Series) -> pd.Series:
    """
    A player's possession-consuming load: minutes x events per minute.

    Usage events are FGA + 0.44*FTA + TOV -- a possession ends in a shot, a
    trip to the line, or a turnover. Weighting by minutes matters: a 34-minute
    high-usage guard is a far bigger hole than a 14-minute specialist with the
    same per-minute rate, and an unweighted share would treat them alike.
    """
    per_min = (post["fga_pm"].fillna(0)
               + 0.44 * post["fta_pm"].fillna(0)
               + post["tov_pm"].fillna(0))
    return per_min * minutes_basis


def absent_usage_share(roster: pd.DataFrame, absent_ids: set[int],
                       load_col: str = "usg_load") -> float:
    """
    Fraction of the team's rotation usage that is unavailable.

    This is the single predictor every beta below is calibrated against, and it
    is continuous on purpose. A binary "is PG1 out" flag cannot express a night
    when PG1 and the starting centre are both out, which is common -- 83% of
    team-games in 2025-26 had at least 15% of rotation usage absent.
    """
    total = roster[load_col].sum()
    if total <= 0:
        return 0.0
    out = roster.loc[roster["player_id"].isin(absent_ids), load_col].sum()
    return float(np.clip(out / total, 0.0, 0.95))


def rotation_flag(roster: pd.DataFrame, mpg_col: str = "mpg",
                  min_mpg: float = 12.0,
                  require_observed: bool = True) -> pd.Series:
    """
    Who counts toward the rotation denominator.

    Deep-bench absences genuinely move nothing, so including them would add
    noise to `absent_usage_share` rather than signal. 12 mpg is the threshold
    the fit used.

    `require_observed` ALSO REQUIRES THE PLAYER TO HAVE ACTUALLY PLAYED. This
    matters more than it sounds. Without it the rotation is drawn from the
    preseason sheet, which lists players at >=12 projected mpg who never appear
    for that team at all -- measured at 21.5% of the flagged set, about three
    phantoms per team. Those phantoms are counted ABSENT every single night,
    which inflated absent_usage_share to a median of 0.31 against a fitted
    range of roughly 0.10-0.18. The absence response was then being driven 2-3x
    outside the range it was estimated on, and a layer ablation showed it
    making the projection WORSE.

    Falls back to mpg alone when there is no observed exposure to filter on --
    either no obs_w column, or nobody in the roster has played yet. That second
    guard is essential: on opening night obs_w is zero for everyone, and
    without it the rotation would come back EMPTY and absence detection would
    silently stop working for the whole first game.
    """
    ok = roster[mpg_col] >= min_mpg
    if require_observed and "obs_w" in roster.columns:
        obs = roster["obs_w"].fillna(0.0)
        if (obs > 0).any():
            ok = ok & (obs > 0)
    return ok


def absent_share_by_position(roster: pd.DataFrame, absent_ids: set[int],
                             load_col: str = "usg_load",
                             pos_col: str = "position") -> pd.DataFrame:
    """
    Decompose the absent usage share by the ABSENT player's position.

    Returns the roster with:
        absent_C / absent_F / absent_G   share vacated at each position
        absent_total                     their sum
        absent_same                      share vacated at the SURVIVOR's own
                                         position (for the minutes bonus)

    Keying on the absent player's position rather than on same-vs-different
    is what won out of sample. It also matters conceptually: guards vacate more
    usage per unit of share (+0.139 vs +0.096 for centres) and guard absence is
    the ONLY absence that measurably hurts the remaining players' shooting.
    Whether the survivor shares that position turns out to matter for MINUTES
    and not for usage.
    """
    out = roster.copy()
    rot = out[rotation_flag(out)]
    total = rot[load_col].sum()

    for p in ("C", "F", "G"):
        out[f"absent_{p}"] = 0.0
    out["absent_total"] = 0.0
    out["absent_same"] = 0.0
    if total <= 0:
        return out

    gone = rot[rot["player_id"].isin(absent_ids)]
    by_pos = (gone.groupby(gone[pos_col].astype(str).str.strip())[load_col]
              .sum() / total).to_dict()

    for p in ("C", "F", "G"):
        out[f"absent_{p}"] = float(np.clip(by_pos.get(p, 0.0), 0.0, 0.95))
    out["absent_total"] = out[["absent_C", "absent_F", "absent_G"]].sum(axis=1) \
                             .clip(0.0, 0.95)

    own_pos = out[pos_col].astype(str).str.strip()
    out["absent_same"] = own_pos.map(
        {p: float(np.clip(by_pos.get(p, 0.0), 0.0, 0.95)) for p in ("C", "F", "G")}
    ).fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Role tiers
# ---------------------------------------------------------------------------
def usage_tier(post: pd.DataFrame, roster: pd.DataFrame) -> pd.Series:
    """
    Assign each player a usage tier, which scales their absence response.

    This is not cosmetic. Measured response per 10 points of absent usage:

        tier        extra minutes   extra usage/min   TS% change
        low            +1.46           +0.006          -0.0062
        mid-low        +1.35           +0.011          -0.0024
        mid-high       +1.01           +0.014          -0.0009
        high           +0.85           +0.013          -0.0034

    Low-usage players get the MOST extra minutes but absorb the LEAST extra
    creation, and pay the most efficiency for what they do absorb (IV slope
    -1.07 vs -0.07 for mid-high). Role players pressed into creating break
    down; high-usage players were already doing it. Applying one uniform
    absence bump across a roster gets this exactly backwards.
    """
    per_min = (post["fga_pm"].fillna(0)
               + 0.44 * post["fta_pm"].fillna(0)
               + post["tov_pm"].fillna(0))
    s = pd.Series(per_min.to_numpy(), index=post.index)
    # Fixed cut points rather than within-team quantiles: a quartile split on a
    # 10-man roster would label a team's third option "high usage".
    cuts = config.ABSENCE["usage_tier_cuts"]
    return pd.Series(
        np.select(
            [s >= cuts[2], s >= cuts[1], s >= cuts[0]],
            ["high", "mid-high", "mid-low"], default="low",
        ),
        index=post.index,
    )


# ---------------------------------------------------------------------------
# Applying the response
# ---------------------------------------------------------------------------
def adjust_rates_for_absence(rates: pd.DataFrame, absent: pd.DataFrame,
                             tier: pd.Series) -> pd.DataFrame:
    """
    Apply the per-stat absence response multiplicatively.

        rate_adj = rate * exp(beta_stat * tier_mult * effective_share)

    where the effective share weights each position's vacated usage by how
    much usage that position actually vacates:

        effective = sum_p usage_mult[p] * absent_p
                    (C 0.873, F 1.082, G 1.264)

    Shooting percentages get their own, much sharper weighting, because guard
    absence is the only absence that moves them:

        shooting_effective = shrink * sum_p shooting_mult[p] * absent_p
                             (C -0.17, F 0.58, G 1.92; shrink 0.5)

    `absent` must carry absent_C / absent_F / absent_G columns aligned to
    `rates` -- see absent_share_by_position.

    Note what does and does not move, because the pattern is a good sanity
    check on the fit:
      - FTA rate moves most, then TOV rate: more drives and contact, and
        ball-handling burden on players unused to it
      - 3PA SHARE goes DOWN: losing a creator costs assisted catch-and-shoot
        threes, so the shot mix shifts inside. The creation-loss channel
        showing up in shot selection
      - rebound, steal and foul RATES are ~flat; they respond through minutes
      - FT% is immune. Free throws are unguarded
    """
    A = config.ABSENCE
    out = rates.copy()
    tier_mult = tier.map(A["tier_mult"]).astype("float64").fillna(1.0).to_numpy()

    pos_cols = ["absent_C", "absent_F", "absent_G"]
    if not all(c in absent.columns for c in pos_cols):
        raise KeyError(
            "adjust_rates_for_absence needs absent_C/F/G columns; "
            "build them with absent_share_by_position()"
        )
    shares = {p: absent[f"absent_{p}"].to_numpy(dtype="float64")
              for p in ("C", "F", "G")}

    effective = sum(A["usage_mult_by_absent_pos"].get(
        p, A["usage_mult_default"]) * v for p, v in shares.items())
    shooting_eff = A["shooting_shrinkage"] * sum(
        A["shooting_mult_by_absent_pos"].get(p, A["shooting_mult_default"]) * v
        for p, v in shares.items())

    if not np.any(effective > 0):
        return out

    shooting = set(A["shooting_betas"])
    skip = shooting if not A["apply_shooting_betas"] else set()

    for stat, beta in A["beta"].items():
        if stat not in out.columns or stat in skip:
            continue
        x = shooting_eff if stat in shooting else effective
        out[stat] = out[stat] * np.exp(beta * tier_mult * x)
        if stat in config.SHARE_RATES:
            out[stat] = out[stat].clip(1e-4, 0.999)
    return out


def minutes_response(mpg: pd.Series, absent: pd.DataFrame,
                     tier: pd.Series, position: pd.Series) -> pd.Series:
    """
    Extra minutes from absence, before the 240 normalization.

        extra = tier_mult * ( sum_p beta[p] * absent_p
                              + same_bonus[own_pos] * absent_own_pos )

    Additive in minutes rather than multiplicative: the measured effect is
    roughly a fixed number of extra minutes, not a fixed percentage, and a
    percentage would hand implausible gains to players already at 36.

    The same-position bonus is where centres stand apart -- +16.8 minutes per
    1.0 share against +4.5 for forwards and +5.2 for guards. A team carries
    two or three centres, so one out has to be covered by another centre;
    guards and forwards are interchangeable enough that the bonus is small.

    Normalization in minutes.py sets the level; this only sets who gains.
    """
    A = config.ABSENCE
    tier_mult = tier.map(A["tier_mult_minutes"]).astype("float64") \
                    .fillna(1.0).to_numpy()

    extra = np.zeros(len(mpg), dtype="float64")
    for p in ("C", "F", "G"):
        col = f"absent_{p}"
        if col in absent.columns:
            extra += A["beta_minutes_by_absent_pos"].get(
                p, A["beta_minutes_default"]) * absent[col].to_numpy(dtype="float64")

    if "absent_same" in absent.columns:
        pos = position.astype(str).str.strip()
        bonus = pos.map(A["beta_minutes_same_pos_bonus"]).astype("float64") \
                   .fillna(A["beta_minutes_same_bonus_default"]).to_numpy()
        extra += bonus * absent["absent_same"].to_numpy(dtype="float64")

    if not np.any(extra > 0):
        return mpg.copy()
    return mpg + tier_mult * extra


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def absence_report(roster: pd.DataFrame, absent_ids: set[int],
                   load_col: str = "usg_load") -> dict:
    """
    What the model thinks an inactive list means. Worth logging every slate:
    an absent_share of 0.00 when a star is listed out almost always means an
    id join failed, not that the star does not matter.
    """
    rot = roster[rotation_flag(roster)]
    share = absent_usage_share(rot, absent_ids, load_col)
    out = rot[rot["player_id"].isin(absent_ids)]
    return {
        "absent_share": round(share, 4),
        "n_absent_rotation": len(out),
        "n_rotation": len(rot),
        "absent_ids_not_on_roster": sorted(absent_ids - set(roster["player_id"])),
        "largest_absence": (
            None if out.empty
            else round(float(out[load_col].max() / rot[load_col].sum()), 4)
        ),
    }

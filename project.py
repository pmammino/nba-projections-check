"""
Assemble the game-level projection.

    posterior rate  x  projected minutes  x  environment adjustment
      (state.py)        (minutes.py)          (market.py)

Absence is handled the same way -- see availability.py. The betas there move
players RELATIVE to each other; the team reconciliation below sets the LEVEL.
That split is what stops the absence bump from being applied twice: bumping
every remaining player's FGA rate by 2% would inflate team attempts, but the
possession constraint pulls the team total back and leaves the redistribution
intact.

Then reconcile at team level, because 590 players updated independently will
not add up. Two constraints:

    minutes  -> 240 per team-game        (handled in minutes.py)
    FGA      -> team implied attempts    (handled here)

The FGA reconciliation also quietly patches the correlation problem. Usage moves
FGA, FTA, TOV and AST together, and independent per-stat updating loses that. A
proper fix is a latent usage factor; letting the team constraint absorb it is
the pragmatic version and is what runs here.

Reconciliation is variance-weighted: players whose posterior is still mostly
prior get moved more than players with a large observed sample. That is the
right direction -- the uncertain estimates are the ones that should flex.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import availability, config, market, opponent as opp_mod, rates as rates_mod


def project_slate(post: pd.DataFrame, mins: pd.DataFrame,
                  env: pd.DataFrame, reconcile: bool = True,
                  opp_profiles: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Project every player-game on a slate.

    post: wide posterior frame from state.posterior_rates()
    mins: output of minutes.project_minutes()
    env:  output of market.game_environment(), one row per team-game
    """
    env_idx = env.set_index(["game_id", "team"])
    out_parts = []

    for (game_id, team), env_row in env_idx.iterrows():
        roster = mins[(mins["game_id"] == game_id) & (mins["team"] == team)]
        if roster.empty:
            continue

        r = roster.merge(post, on="player_id", how="left")
        rate_cols = [c for c in config.ALL_STATS if c in r.columns]
        if not rate_cols:
            continue

        # Two independent multiplicative adjustments, applied in sequence:
        # game environment (pace, implied efficiency) then teammate absence.
        # Both are exp(beta * x) forms, so order does not matter and neither
        # needs to know about the other.
        adj = market.adjust_rates(r[rate_cols], env_row)

        # Stat-specific opponent profile, on top of the market scalar. Skipped
        # entirely when no profiles are supplied, so every existing caller
        # behaves exactly as before.
        if opp_profiles is not None and config.OPPONENT_PROFILE["enabled"]:
            adj = opp_mod.adjust_rates_for_opponent(
                adj, env_row["opp"], opp_profiles)
        pos_cols = ["absent_C", "absent_F", "absent_G"]
        if all(c in r.columns for c in pos_cols) and (r[pos_cols].sum(axis=1) > 0).any():
            tier = r["usage_tier"] if "usage_tier" in r else pd.Series(
                "mid-low", index=r.index)
            adj = availability.adjust_rates_for_absence(adj, r[pos_cols], tier)
        r[rate_cols] = adj

        r["game_id"] = game_id
        r["team"] = team
        r["opp"] = env_row["opp"]
        r["is_home"] = env_row.get("is_home", 0)
        r["implied_team_pts"] = env_row["team_pts"]
        r["poss_hat"] = env_row["poss_hat"]
        out_parts.append(r)

    if not out_parts:
        return pd.DataFrame()

    df = pd.concat(out_parts, ignore_index=True)
    df = _counts_from_rates(df)

    if reconcile:
        df = _reconcile_team_volume(df)
        df = _counts_from_rates(df)

    df["pts"] = 2 * df["fg2m"] + 3 * df["pt3m"] + df["ftm"]
    df["reb"] = df["oreb"] + df["dreb"]
    return df


def _counts_from_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Rates x minutes -> projected counting stats."""
    m = df["minutes_proj"]
    for stat in config.VOLUME_RATES:
        df[stat[:-3]] = df[stat] * m

    df["pt3a"] = df["fga"] * df["pt3a_share"]
    df["fg2a"] = df["fga"] - df["pt3a"]
    df["fg2m"] = df["fg2a"] * df["fg2_pct"]
    df["pt3m"] = df["pt3a"] * df["fg3_pct"]
    df["ftm"] = df["fta"] * df["ft_pct"]
    df["fgm"] = df["fg2m"] + df["pt3m"]
    return df


def _reconcile_team_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale each team's FGA rate so projected team attempts match the market.

    Target attempts come from the possession estimate, not from the sum of the
    player projections:
        team_fga_target ~ poss_hat * (1 - ft_trip_share - tov_share)
    which we approximate from the team's own projected shot/turnover mix rather
    than hardcoding league splits.

    The adjustment is variance-weighted by prior weight: a player still running
    on 90% prior absorbs more of the correction than one with 800 minutes
    logged.
    """
    df = df.copy()
    if "fga_pm__pw" not in df.columns:
        return df

    for (game_id, team), grp in df.groupby(["game_id", "team"]):
        poss = grp["poss_hat"].iloc[0]
        proj_fga = grp["fga"].sum()
        proj_tov = grp["tov"].sum()
        proj_fta = grp["fta"].sum()

        # Possessions end in a shot, a turnover, or a trip to the line.
        proj_poss = proj_fga + proj_tov + 0.44 * proj_fta - grp["oreb"].sum()
        if proj_poss <= 0 or not np.isfinite(poss):
            continue

        gap = poss / proj_poss
        if not np.isfinite(gap) or abs(gap - 1.0) < 1e-6:
            continue

        # Distribute the correction in proportion to prior weight, normalized so
        # the team-level effect equals `gap`.
        pw = grp["fga_pm__pw"].fillna(1.0).to_numpy()
        w = pw / pw.mean() if pw.mean() > 0 else np.ones(len(grp))
        player_gap = 1.0 + (gap - 1.0) * w

        idx = grp.index
        for stat in ("fga_pm", "fta_pm", "tov_pm"):
            df.loc[idx, stat] = df.loc[idx, stat] * player_gap

    return df


def fantasy_points(df: pd.DataFrame, scoring: dict[str, float]) -> pd.Series:
    """
    Apply a scoring system. Kept separate from the model on purpose: the
    components should be right before anything gets collapsed into one number.

    Note that double-double / triple-double bonuses CANNOT be computed from
    means -- E[f(X)] != f(E[X]) for an indicator. Those need the simulation,
    which is what the posterior sd columns are for.
    """
    total = pd.Series(0.0, index=df.index)
    for col, pts in scoring.items():
        if col not in df.columns:
            raise KeyError(f"scoring references missing column {col!r}")
        total = total + df[col] * pts
    return total


DK_NBA = {
    "pts": 1.0, "pt3m": 0.5, "reb": 1.25, "ast": 1.5,
    "stl": 2.0, "blk": 2.0, "tov": -0.5,
}

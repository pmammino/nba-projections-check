"""
Convert raw counting stats (preseason season-totals OR single-game logs) into
the canonical rate vector defined in config.

Two entry points:
    derive_components(df)  -- add fg2a/fg2m/pts, normalize column names
    to_rates(df)           -- produce the per-minute / per-attempt rate vector

The same rate definitions are used for the prior and for the evidence. That is
the whole point: if the preseason sheet and the game logs ever disagree about
what "fg2a" means, every posterior is silently wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Canonical raw columns the rest of the package expects.
RAW_COLS = [
    "minutes", "fga", "fgm", "pt3a", "pt3m", "fta", "ftm",
    "oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
]

# nba_api PlayerGameLogs -> canonical. Preseason sheet already mostly matches.
NBA_API_RENAME = {
    "MIN": "minutes", "FGA": "fga", "FGM": "fgm",
    "FG3A": "pt3a", "FG3M": "pt3m", "FTA": "fta", "FTM": "ftm",
    "OREB": "oreb", "DREB": "dreb", "AST": "ast", "STL": "stl",
    "BLK": "blk", "TOV": "tov", "PF": "pf",
    "PLAYER_ID": "nba_player_id", "GAME_DATE": "game_date",
    "GAME_ID": "game_id", "TEAM_ABBREVIATION": "team",
}

PRESEASON_RENAME = {
    "assists": "ast", "steals": "stl", "blocks": "blk", "turnovers": "tov",
}


def normalize_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Rename source-specific columns to the canonical set."""
    df = df.copy()
    if source == "nba_api":
        df = df.rename(columns=NBA_API_RENAME)
    elif source == "preseason":
        df = df.rename(columns=PRESEASON_RENAME)
        df.columns = [c.strip() for c in df.columns]
        if "position" in df:
            df["position"] = df["position"].astype(str).str.strip()
    else:
        raise ValueError(f"unknown source: {source!r}")
    return df


def derive_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the derived raw columns the rate vector needs.

    The preseason sheet has no points column and no 2P split; both are exact
    identities, not estimates.
    """
    df = df.copy()
    df["fg2a"] = df["fga"] - df["pt3a"]
    df["fg2m"] = df["fgm"] - df["pt3m"]
    df["pts"] = 2 * df["fgm"] + df["pt3m"] + df["ftm"]
    df["reb"] = df["oreb"] + df["dreb"]

    # Data integrity. These DO fire on the real sheet -- run validate.audit()
    # first and validate.repair() if you want to proceed anyway. Failing loudly
    # here is deliberate: a negative 2P attempt count silently produces a
    # nonsense 2P% that then propagates into every points projection.
    bad2a = int((df["fg2a"] < 0).sum())
    bad2m = int((df["fg2m"] < 0).sum())
    if bad2a or bad2m:
        raise ValueError(
            f"{bad2a} rows with pt3a > fga and {bad2m} rows with pt3m > fgm. "
            "Run validate.audit() to see which, then validate.repair() to "
            "clamp them."
        )
    return df


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Rate with a zero denominator is undefined, not zero."""
    out = pd.Series(np.nan, index=num.index, dtype="float64")
    ok = den > 0
    out.loc[ok] = num.loc[ok] / den.loc[ok]
    return out


def to_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce the canonical rate vector.

    Per-minute rates come back as NaN when minutes == 0, and conversion rates
    as NaN when the player took no attempts. NaN means "no information", which
    is exactly what the Bayesian update needs to see -- it must contribute zero
    evidence, not pull the posterior toward zero.
    """
    df = derive_components(df)
    out = pd.DataFrame(index=df.index)

    mins = df["minutes"].astype("float64")
    for stat in config.VOLUME_RATES:
        raw = stat[:-3]  # strip "_pm"
        out[stat] = _safe_div(df[raw].astype("float64"), mins)

    for stat, (num_col, den_col) in config.SHARE_RATES.items():
        out[stat] = _safe_div(
            df[num_col].astype("float64"), df[den_col].astype("float64")
        )

    return out


def evidence_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape raw counts into (successes, exposure) pairs -- the form the
    conjugate update actually consumes.

    Returns one column pair per stat: "<stat>__x" and "<stat>__n".
    For per-minute stats exposure is minutes; for conversion rates it is
    attempts.
    """
    df = derive_components(df)
    out = pd.DataFrame(index=df.index)

    for stat in config.VOLUME_RATES:
        raw = stat[:-3]
        out[f"{stat}__x"] = df[raw].astype("float64")
        out[f"{stat}__n"] = df["minutes"].astype("float64")

    for stat, (num_col, den_col) in config.SHARE_RATES.items():
        out[f"{stat}__x"] = df[num_col].astype("float64")
        out[f"{stat}__n"] = df[den_col].astype("float64")

    return out


def points_from_rates(rates: pd.DataFrame, minutes: pd.Series) -> pd.Series:
    """
    Reconstruct points from the component rates.

    Points is never modeled directly -- it falls out of attempts x conversion.
    That keeps the scoring projection internally consistent and makes it
    possible to attribute a projection change to volume vs efficiency.
    """
    fga = rates["fga_pm"] * minutes
    fta = rates["fta_pm"] * minutes
    pt3a = fga * rates["pt3a_share"]
    fg2a = fga - pt3a
    return (
        2 * fg2a * rates["fg2_pct"]
        + 3 * pt3a * rates["fg3_pct"]
        + 1 * fta * rates["ft_pct"]
    )

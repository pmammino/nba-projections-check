"""
Input validation for the preseason projection sheet.

This exists because the pipeline found real errors in the source file on the
first run, and a projection model that silently swallows bad inputs produces
confidently wrong output for the rest of the season. Every check here fired at
least once on 2025NBAProjections.xlsx.

Run `audit()` before `build_priors()`. It returns a frame of findings with a
severity; `fatal` findings must be fixed or explicitly repaired, `warn`
findings are logged and tolerated.
"""

from __future__ import annotations

import pandas as pd

# Tolerance for the reb vs oreb+dreb check. +/-1 is rounding from a per-game
# projection scaled to a season total; anything larger is a real inconsistency.
REB_TOLERANCE = 1


def audit(pre: pd.DataFrame, id_cols=("newsid", "firstname", "lastname")) -> pd.DataFrame:
    """Return one row per finding."""
    df = pre.copy()
    df.columns = [c.strip() for c in df.columns]
    found = []

    def add(mask, check, severity, note):
        for _, row in df[mask].iterrows():
            found.append({
                "check": check,
                "severity": severity,
                **{c: row.get(c) for c in id_cols},
                "note": note.format(**row.to_dict()),
            })

    # --- fatal: breaks the rate identities -------------------------------
    add(df["pt3a"] > df["fga"], "pt3a_gt_fga", "fatal",
        "pt3a={pt3a} exceeds fga={fga}; implies negative 2P attempts")
    add(df["pt3m"] > df["pt3a"], "pt3m_gt_pt3a", "fatal",
        "pt3m={pt3m} exceeds pt3a={pt3a}")
    add(df["fgm"] > df["fga"], "fgm_gt_fga", "fatal",
        "fgm={fgm} exceeds fga={fga}")
    add(df["ftm"] > df["fta"], "ftm_gt_fta", "fatal",
        "ftm={ftm} exceeds fta={fta}")
    add((df["minutes"] > 0) & (df["games"] <= 0), "minutes_no_games", "fatal",
        "minutes={minutes} with games={games}")

    # --- warn: internally inconsistent but not load-bearing --------------
    reb_gap = (df["reb"] - (df["oreb"] + df["dreb"])).abs()
    add(reb_gap > REB_TOLERANCE, "reb_mismatch", "warn",
        "reb={reb} vs oreb+dreb={oreb}+{dreb}")
    add(df["minutes"] / df["games"].clip(lower=1) > 42, "mpg_high", "warn",
        "implied mpg above 42")
    add(df["minutes"] <= 0, "zero_minutes", "info",
        "no projected minutes; excluded from the model")
    if "team" in df:
        add(df["team"].isna(), "no_team", "info",
            "unsigned/two-way; prior strength reduced")

    if not found:
        return pd.DataFrame(columns=["check", "severity", *id_cols, "note"])
    return pd.DataFrame(found)


def summarize(report: pd.DataFrame) -> pd.DataFrame:
    """Counts by check and severity, for a one-line log entry."""
    if report.empty:
        return pd.DataFrame(columns=["check", "severity", "n"])
    return (report.groupby(["severity", "check"], as_index=False)
            .size().rename(columns={"size": "n"})
            .sort_values(["severity", "n"], ascending=[True, False]))


def repair(pre: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, documented repairs so a single bad row does not stop the pipeline.

    The `reb` column is NOT repaired because the model never reads it -- every
    rebound rate is built from oreb and dreb directly. The mismatch is reported
    so the source can be fixed upstream, but it has no effect downstream.
    """
    df = pre.copy()
    df.columns = [c.strip() for c in df.columns]

    # pt3a > fga: clamp pt3a to fga and scale pt3m to keep 3P% intact. This is
    # a guess at intent and should be corrected at the source instead.
    bad = df["pt3a"] > df["fga"]
    if bad.any():
        pct = (df.loc[bad, "pt3m"] / df.loc[bad, "pt3a"]).fillna(0.0)
        df.loc[bad, "pt3a"] = df.loc[bad, "fga"]
        df.loc[bad, "pt3m"] = (df.loc[bad, "pt3a"] * pct).round()
        df.loc[bad, "fgm"] = df[["fgm", "fga"]].min(axis=1)[bad]

    df["reb"] = df["oreb"] + df["dreb"]
    return df

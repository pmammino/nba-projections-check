"""
The minutes model.

Deliberately NOT part of the Bayesian rate machinery. Minutes are rotation
decisions, injury news, and blowout risk -- none of which is a slowly-drifting
player attribute, which is the only thing the conjugate update handles well.

Worth stating the stakes: once rates are decomposed properly, per-minute rates
are far more stable than minutes, so nearly all remaining error in a per-game
projection lives here. This module being crude is the single largest known
weakness of the whole pipeline.

Pipeline:
    1. blend decayed recent min/game with the preseason prior  (shrinkage)
    2. assign a depth role from the blended estimate
    3. apply blowout exposure from |spread|, by role
    4. apply back-to-back adjustment, by role
    5. renormalize each team to 240 minutes
    6. clip to [floor, ceiling]

Step 5 is what makes step 3 self-consistent: minutes the starters give up in a
blowout have to land somewhere, and renormalization puts them on the bench
rather than deleting them from the game.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import availability, config

M = config.MINUTES


def baseline_minutes(preseason: pd.DataFrame, logs: pd.DataFrame | None,
                     as_of: pd.Timestamp,
                     player_id_col: str = "newsid") -> pd.DataFrame:
    """
    Blend preseason min/game with decayed observed min/game.

    Preseason min/game = minutes / games, which is NOT the same as minutes per
    game played if the projection already discounts for missed games. It does
    here (games < 82 for nearly everyone), so minutes/games is the right
    denominator: it is minutes per game *played*.

    Shrinkage is the same form as the rate model, in games rather than minutes:
        mpg = (prior_games * prior_mpg + sum_w * obs_mpg) / (prior_games + sum_w)
    """
    pre = preseason.copy()
    pre.columns = [c.strip() for c in pre.columns]
    pid = pre[player_id_col].astype("int64")
    prior_mpg = np.where(
        pre["games"].to_numpy() > 0,
        pre["minutes"].to_numpy() / np.maximum(pre["games"].to_numpy(), 1),
        0.0,
    )
    # Availability: the projection already discounts for missed games, so
    # games/82 is the implied probability the player is active on any given
    # night. This matters more than it looks -- see the note on the two modes
    # in project_minutes().
    avail = np.clip(pre["games"].to_numpy() / 82.0, 0.0, 1.0)

    base = pd.DataFrame({
        "player_id": pid.to_numpy(),
        "team": pre["team"].to_numpy() if "team" in pre else None,
        "position": (pre["position"].astype(str).str.strip().to_numpy()
                     if "position" in pre else "F"),
        "prior_mpg": prior_mpg,
        "avail": avail,
        "obs_w": 0.0,
        "obs_mpg": np.nan,
    })

    if logs is not None and len(logs):
        lg = logs.copy()
        lg["game_date"] = pd.to_datetime(lg["game_date"])
        # `logs` holds PLAYED games only (every real caller filters out DNP /
        # zero-minute rows before this is called -- see pipeline.py and
        # backtest_slates.py). A DNP therefore contributes no row at all, so
        # it is excluded from the decay entirely rather than counted as
        # evidence of zero minutes. That is the correct behaviour: a missed
        # game (rest, injury, a healthy scratch) says nothing about how many
        # minutes a player would get if he DID play, which is exactly what
        # this average is trying to estimate.
        #
        # GAP-CAPPED DECAY. Plain calendar-day decay (age = as_of - game_date)
        # is right for the steady state, where games come every 1-4 days and
        # the gap IS the signal -- but it breaks on a real injury absence.
        # Confirmed on a real player (Jerami Grant, POR): an established
        # 28-game, ~32 mpg starter missed 28 days, then played one
        # minutes-restricted comeback game at 23 minutes. Under plain decay,
        # that one game's weight (age~0) swamped 28 games of history whose
        # weight had decayed through the entire gap (age~28-33 days against
        # tau=10), so his modeled mpg snapped 34.3 -> 25.2 and his assigned
        # role flipped starter -> rotation after a SINGLE game, rather than
        # treating that game as one noisy data point in an established role.
        #
        # Fix: cap how much any single inter-game gap can age a game's
        # weight, at `gap_cap_days` (a normal rest interval -- back-to-backs
        # up through an All-Star-break-length gap). Time beyond that, within
        # one gap, does not make the games on either side of it any more
        # "stale" relative to each other -- an injury absence is not the kind
        # of drift calendar decay is meant to capture. This only changes
        # anything across a gap LONGER than the cap; normal week-to-week play
        # (gaps of a few days) decays exactly as before.
        #
        # Concretely: each game's effective age is the sum of CAPPED forward
        # gaps from that game to every later game (or to `as_of`, for a
        # player who has not returned yet), not the raw elapsed calendar
        # time. That keeps pre-injury history meaningfully weighted through
        # and just after the return, while a return game's own, ordinary gap
        # to the next game still decays normally -- so sustained new evidence
        # (several real games back) still overrides stale history, same as
        # it always did; only the one-time artificial cliff from the gap
        # itself is removed.
        lg = lg.sort_values(["player_id", "game_date"]).reset_index(drop=True)
        next_date = lg.groupby("player_id")["game_date"].shift(-1)
        next_date = next_date.fillna(pd.Timestamp(as_of))
        gap_days = (next_date - lg["game_date"]).dt.total_seconds() / 86400.0
        lg["capped_gap"] = np.minimum(gap_days.to_numpy(), M["gap_cap_days"])
        eff_age = (lg.iloc[::-1].groupby("player_id")["capped_gap"]
                   .cumsum().sort_index())
        w = np.exp(-eff_age.to_numpy() / M["tau_days"])
        agg = pd.DataFrame({
            "player_id": lg["player_id"].to_numpy(),
            "w": w,
            "wm": (w * lg["minutes"].astype("float64").to_numpy()),
        }).groupby("player_id", as_index=False).sum()
        agg["obs_mpg_new"] = agg["wm"] / agg["w"]
        base = base.merge(
            agg[["player_id", "w", "obs_mpg_new"]], on="player_id", how="outer"
        )
        base["obs_w"] = base["w"].fillna(0.0)
        base["obs_mpg"] = base["obs_mpg_new"]
        base = base.drop(columns=["w", "obs_mpg_new"])

    # TEAM COMES FROM THE MOST RECENT GAME, NOT FROM THE PRESEASON SHEET.
    #
    # This matters twice over.
    #
    # Players in the logs but not the sheet (rookies, two-way call-ups) have no
    # team at all. Without a backfill they are dropped by project_minutes on
    # its merge with the slate, and the 240 normalisation silently reallocates
    # their minutes to sheet players. Late in a season that is ~24% of minutes.
    #
    # TRADED PLAYERS are the worse case, and an earlier version got it wrong by
    # only filling NULLS. A player traded in February still carried his October
    # team, so he was dropped from his NEW team's roster. Measured on a real
    # post-deadline slate: Chicago had 9 active players and only 2 were
    # projectable -- the other 7 were still filed under NY, CHR, BOS, PHO, DET
    # and MIN. Those 2 then absorbed all 240 minutes and pinned at the
    # 44-minute ceiling. Preferring the observed team fixes it; the sheet is
    # only the fallback for players who have not played yet.
    if logs is not None and len(logs) and "team" in logs.columns:
        recent = (logs.sort_values("game_date")
                      .groupby("player_id")["team"].last())
        seen = base["player_id"].map(recent)
        base["team"] = seen.where(seen.notna(), base["team"])

    base["prior_mpg"] = base["prior_mpg"].fillna(0.0)
    if "position" in base:
        base["position"] = base["position"].fillna("F")

    # Shrinkage toward the preseason mpg. prior_games is 0.0 by default --
    # see config.MINUTES for the measurement, but in short: the preseason
    # minutes projection makes the forecast strictly WORSE once any observed
    # minutes exist, monotonically at every decay setting tested.
    #
    # prior_games = 0 therefore needs an explicit fallback rather than a
    # division by zero. A player with no observed minutes at all (opening
    # night, a player yet to debut) still needs a number, and the preseason
    # projection is the only thing available for him.
    have_obs = base["obs_mpg"].notna() & (base["obs_w"] > 0)
    num = M["prior_games"] * base["prior_mpg"]
    den = pd.Series(M["prior_games"], index=base.index, dtype="float64")
    num = num.where(~have_obs, num + base["obs_w"] * base["obs_mpg"].fillna(0.0))
    den = den.where(~have_obs, den + base["obs_w"])
    base["mpg"] = np.where(den > 0, num / den.replace(0, np.nan),
                           base["prior_mpg"])
    base["mpg"] = pd.Series(base["mpg"], index=base.index).fillna(
        base["prior_mpg"]).fillna(0.0)
    base["avail"] = base["avail"].fillna(base["avail"].median())
    # Expected minutes per NIGHT, not per game played.
    base["exp_mpn"] = base["mpg"] * base["avail"]
    return base


def assign_role(mpg: pd.Series) -> pd.Series:
    """Depth role from blended mpg. Drives the blowout response."""
    return pd.Series(
        np.select(
            [mpg >= M["role_cut_starter"], mpg >= M["role_cut_rotation"]],
            ["starter", "rotation"],
            default="bench",
        ),
        index=mpg.index,
    )


def blowout_exposure(spread: pd.Series) -> pd.Series:
    """
    Garbage-time exposure from |spread|.

    Zero until the spread clears the threshold, then linear, then capped.
    This shape is a guess. It is also the single piece of this model most worth
    fitting for real: regress observed minutes on |spread| within depth role and
    the curve falls straight out.
    """
    excess = (spread.abs() - M["blowout_threshold"]).clip(lower=0.0)
    return (excess / M["blowout_scale"]).clip(upper=M["blowout_cap"])


def project_minutes(base: pd.DataFrame, env: pd.DataFrame,
                    b2b_teams: set[str] | None = None,
                    actives: dict[str, set[int]] | None = None,
                    mode: str = "auto",
                    absent: dict[str, set[int]] | None = None,
                    post: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Project minutes for the games in `env`.

    TWO MODES, and picking the wrong one is the commonest way to get minutes
    badly wrong:

    mode="expected" (no actives known)
        Uses expected minutes per NIGHT = mpg x P(active). This is the right
        basis for season-long / rest-of-season value, where you do not know who
        will be out on a given date.

    mode="actives" (an inactive list is available)
        Filters to the active players and uses mpg (minutes per game PLAYED),
        then normalizes to 240 across the actives only. This is the right basis
        for DFS and any single-slate projection.

    mode="auto" picks "actives" when `actives` is supplied, else "expected".

    WHY THIS MATTERS -- measured on the 2025 sheet: summing mpg across each
    team's projected roster gives ~343 minutes, not 240, because ~16 players
    per team are projected but only ~9-10 play on a given night. Normalizing
    raw mpg to 240 therefore compresses every starter by roughly 30%. Using
    availability-weighted minutes instead removes most of that bias: total
    projected minutes / 82 comes to 262 per team, still 9% over 240, so the
    sheet genuinely over-allocates team minutes and the normalization below is
    doing real work rather than just rounding.

    `actives`: team -> set of player_ids active tonight.
    `b2b_teams`: teams on the second night of a back-to-back.
    `absent`:   team -> set of player_ids ruled out. Drives the absence
                response; pass `post` alongside it so usage tiers can be
                computed. When `actives` is given, `absent` is derived from it
                if not supplied explicitly.
    `post`:     posterior rate frame, needed only for the absence response.

    The absence response redistributes minutes UNEVENLY -- bench players gain
    ~1.46 min per 10 points of absent rotation usage while high-usage players
    gain only ~0.85, because stars are already near their ceiling. The 240
    normalization then sets the level, so this only decides who gains.
    """
    b2b_teams = b2b_teams or set()
    if mode == "auto":
        mode = "actives" if actives else "expected"
    if mode not in ("actives", "expected"):
        raise ValueError(f"mode must be 'actives' or 'expected', got {mode!r}")

    base = base.copy()

    # `base["team"]` is backfilled from each player's most recently PLAYED
    # game (see baseline_minutes), which cannot know about a trade until the
    # player has actually suited up for the new team once. On the trade
    # itself -- or a two-way call-up's first game for a team other than the
    # preseason sheet's -- that leaves him filed under his old team for
    # exactly one night: his first game for the new one. `actives` already
    # says who is playing for whom TONIGHT, so when it is supplied it is the
    # more current source and should win.
    #
    # Measured on a real trade-deadline slate (CHI, 2026-02-05, three players
    # traded in that day): without this override those three players stayed
    # filed under their old teams, vanished from CHI's projection entirely,
    # and left CHI's 5 remaining known players to cover all 240 minutes --
    # the team came up more than 20 minutes short even after fixing the
    # ceiling-convergence bug above, because there just weren't enough known
    # players to begin with.
    if mode == "actives" and actives:
        reassign = pd.Series(
            {pid: t for t, ids in actives.items() for pid in ids}
        )
        if len(reassign):
            mapped = base["player_id"].map(reassign)
            base["team"] = mapped.where(mapped.notna(), base["team"])

    minute_col = "mpg" if mode == "actives" else "exp_mpn"
    base["minute_basis"] = base[minute_col]
    # Role is always assigned from mpg (minutes when playing), never from the
    # availability-weighted figure -- a 32-mpg starter who misses a third of the
    # season is still a starter and should still shed minutes in a blowout.
    base["role"] = assign_role(base["mpg"])

    slate = env[["game_id", "team", "spread"]].copy()
    df = base.merge(slate, on="team", how="inner")

    if mode == "actives":
        keep = df.apply(
            lambda r: r["player_id"] in actives.get(r["team"], set()), axis=1
        )
        df = df[keep]

    df = df[df["minute_basis"] > 0].copy()
    if df.empty:
        return df.assign(minutes_proj=[], absent_share=[])

    # --- absence response, before the blowout curve and the 240 normalization
    df["absent_share"] = 0.0
    # Tier is a property of the player, not of tonight's inactive list, so it
    # is computed whenever `post` is available -- including at full strength.
    # (An earlier version computed it only inside the `if absent` branch, which
    # left every full-strength projection tagged with the placeholder tier and
    # made before/after comparisons unreadable.)
    df["usage_tier"] = "mid-low"
    if post is not None:
        tiers = availability.usage_tier(post, post)
        tier_map = dict(zip(post["player_id"], tiers))
        df["usage_tier"] = df["player_id"].map(tier_map).fillna("mid-low")

    POS_SHARE_COLS = ["absent_C", "absent_F", "absent_G", "absent_same"]
    for c in POS_SHARE_COLS:
        df[c] = 0.0
    if absent and post is not None:
        loads = base.merge(post, on="player_id", how="left")
        loads["usg_load"] = availability.usage_load(loads, loads["mpg"])

        for team, out_ids in absent.items():
            tm = loads[loads["team"] == team]
            if tm.empty or not availability.rotation_flag(tm).any():
                continue
            split = availability.absent_share_by_position(tm, out_ids)
            maps = {c: dict(zip(split["player_id"], split[c]))
                    for c in POS_SHARE_COLS}

            m = df["team"] == team
            for c in POS_SHARE_COLS:
                df.loc[m, c] = df.loc[m, "player_id"].map(maps[c]).fillna(0.0)
            df.loc[m, "absent_share"] = df.loc[m, ["absent_C", "absent_F",
                                                   "absent_G"]].sum(axis=1)

            pos_col = (df.loc[m, "position"] if "position" in df.columns
                       else pd.Series("F", index=df.index[m]))
            df.loc[m, "minute_basis"] = availability.minutes_response(
                df.loc[m, "minute_basis"], df.loc[m, POS_SHARE_COLS],
                df.loc[m, "usage_tier"], pos_col,
            )
        # Never let the response push a projection below zero.
        df["minute_basis"] = df["minute_basis"].clip(lower=0.0)

    exposure = blowout_exposure(df["spread"])
    role_beta = df["role"].map(M["blowout_role_beta"]).astype("float64")
    df["min_raw"] = df["minute_basis"] * (1.0 + role_beta * exposure)

    is_b2b = df["team"].isin(b2b_teams)
    b2b_mult = np.where(
        df["role"].eq("bench"), M["b2b_bench_mult"], M["b2b_starter_mult"]
    )
    df["min_raw"] = (df["min_raw"] * np.where(is_b2b, b2b_mult, 1.0)).clip(lower=0.0)

    # Renormalize each team-game to `team_minutes`, converging on the ceiling.
    #
    # A single clip-then-rescale pass breaks the 240 constraint the moment ANY
    # player gets capped: the uncapped players have to absorb what the capped
    # ones give up, but capping removes those capped players from the pool
    # being rescaled, so the absorption falls short. A second fixed pass
    # caught the common case, but it still falls short whenever SEVERAL
    # players are capped AT ONCE -- which is exactly what happens when a team
    # has several rotation players out simultaneously and the absence
    # response pushes multiple remaining players toward the ceiling together.
    # Measured on the full season with the real absence wiring: 3 of 1,896
    # team-games missed 240 by more than 2 minutes under the old 2-pass
    # scheme, one by a full 20 minutes (CHI, 2026-02-05: 8 rotation players
    # out, 5 players pinned at the ceiling at once).
    #
    # Fixed with water-filling: repeatedly rescale only the still-uncapped
    # players to make up whatever the capped players are NOT contributing,
    # re-checking the ceiling each pass, until no new player gets capped. Real
    # rosters converge in 2-4 passes; the loop is bounded generously (20
    # passes -- more than any real roster size) purely as a safety stop.
    gid, team = df["game_id"], df["team"]
    capped = pd.Series(False, index=df.index)
    minutes_proj = pd.Series(0.0, index=df.index)
    for _ in range(20):
        capped_total = (minutes_proj.where(capped, 0.0)
                        .groupby([gid, team]).transform("sum"))
        free_raw_total = (df["min_raw"].where(~capped, 0.0)
                          .groupby([gid, team]).transform("sum"))
        target_free = M["team_minutes"] - capped_total
        # np.where evaluates both branches eagerly, so a bare
        # target_free / free_raw_total would still divide by zero when
        # every player in the group is capped (free_raw_total == 0) --
        # guard the denominator itself, not just which branch is kept.
        safe_den = free_raw_total.where(free_raw_total > 0, 1.0)
        scale = np.where(free_raw_total > 0, target_free / safe_den, 0.0)
        new_vals = (df["min_raw"] * scale).where(~capped, minutes_proj)
        newly_capped = (~capped) & (new_vals > M["min_ceiling"])
        minutes_proj = new_vals.where(~newly_capped, M["min_ceiling"])
        if not newly_capped.any():
            break
        capped = capped | newly_capped

    # Degenerate fallback: if more players are genuinely driven above the
    # ceiling than 240 minutes can fit even with everyone else at zero, the
    # ceiling and the 240 constraint cannot both hold (this needs 6+ players
    # independently above 44 projected minutes -- not a real depth chart, but
    # possible if an upstream adjustment misbehaves). Prefer hitting 240:
    # fall back to an unconstrained proportional split for that team-game
    # rather than silently leaving it short.
    team_sum = minutes_proj.groupby([gid, team]).transform("sum")
    raw_total = df["min_raw"].groupby([gid, team]).transform("sum")
    infeasible = (raw_total > 0) & ((M["team_minutes"] - team_sum).abs() > 1e-6)
    if infeasible.any():
        safe_raw = raw_total.where(raw_total > 0, 1.0)
        fallback = df["min_raw"] * np.where(
            raw_total > 0, M["team_minutes"] / safe_raw, 0.0
        )
        minutes_proj = minutes_proj.where(~infeasible, fallback)

    df["minutes_proj"] = minutes_proj.clip(M["min_floor"], M["min_ceiling"])

    keep = ["game_id", "player_id", "team", "role", "mpg", "avail",
            "minute_basis", "spread", "minutes_proj", "absent_share",
            "absent_C", "absent_F", "absent_G", "absent_same", "usage_tier"]
    if "position" in df.columns:
        keep.insert(3, "position")
    return df[keep]


def team_minutes_audit(pre: pd.DataFrame) -> pd.DataFrame:
    """
    How far the preseason sheet is from minute-balanced at team level.

    Two views:
        sum_mpg       sum of minutes/games across the roster. Will greatly
                      exceed 240 -- that is expected, not an error, because it
                      ignores availability.
        mpn           sum of total minutes / 82. SHOULD be ~240. Anything much
                      above means the sheet allocates more team-minutes than
                      exist, and the normalization in project_minutes will be
                      silently rescaling that team's whole roster down.
    """
    df = pre.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["team"])
    df["mpg"] = df["minutes"] / df["games"].clip(lower=1)
    out = df.groupby("team").agg(
        players=("mpg", "size"),
        sum_mpg=("mpg", "sum"),
        mpn=("minutes", lambda x: x.sum() / 82.0),
    )
    out["mpn_vs_240"] = (out["mpn"] / M["team_minutes"] - 1.0)
    return out.sort_values("mpn_vs_240", ascending=False).round(3)

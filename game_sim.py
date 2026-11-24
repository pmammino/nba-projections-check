"""
Game-level simulation, and direction-aware floors and ceilings.

TWO THINGS THIS DOES THAT simulate.py DOES NOT.

1. GAME AND TEAM COHERENCE. simulate.py draws every player's minutes
   independently, so within a single draw a team's minutes do not add up --
   measured at a mean of 218-234 against a target of 240, with a standard
   deviation of 27. That is fine for a marginal per-player band and wrong for
   anything that depends on players in the same game being consistent with
   each other. Here each draw is a GAME: one shared pace shock, then minutes
   allocated within each team and renormalized to 240 (plus overtime), so
   every draw is an internally consistent box score.

   It matters beyond tidiness. A fast game should lift everyone on both teams
   together; independent draws wash that correlation out and understate the
   spread of any team-level or stacked quantity.

2. DIRECTION-AWARE CEILING AND FLOOR. "Ceiling" means the GOOD outcome, not
   the high number. For turnovers and personal fouls a player's ceiling game
   is a LOW one, so ceiling is the 25th percentile and floor is the 75th.
   Getting this backwards would label a six-turnover night a player's ceiling.

   The percentile pair stays 25/75 throughout; only which end is called
   ceiling flips. Fantasy points use the scoring weights, so they are
   higher-is-better even though turnovers inside them are not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, simulate

# Which direction is good. False means a LOWER value is the better outcome, so
# ceiling is the 25th percentile and floor the 75th.
HIGHER_IS_BETTER = {
    "minutes": True, "pts": True, "reb": True, "oreb": True, "dreb": True,
    "ast": True, "stl": True, "blk": True, "fga": True, "fta": True,
    "pt3m": True, "fgm": True, "ftm": True, "dd": True, "td": True,
    "fp": True,
    # the negative stats
    "tov": False,
    "pf": False,
}

# Game-level pace shock. Possessions land about 4.3% away from the market-based
# estimate in standard-deviation terms (possession MAE 3.45 against a mean of
# 101.5, so a Gaussian sd of 4.32). Applied as a shared lognormal multiplier so
# both teams in a game move together.
PACE_SHOCK_SD = 0.043

# Overtime, measured on 2,345 team-games: 4.4% of games run past 250 team
# minutes, and those average 268.7 -- close to one extra period (265). Mean
# team minutes are 241.3 rather than 240 for exactly this reason.
OT_PROB = 0.044
OT_MINUTES = 25.0


def _direction(stat: str) -> bool:
    return HIGHER_IS_BETTER.get(stat, True)


def simulate_games(proj: pd.DataFrame, n_sims: int = 4000, seed: int = 0,
                   scoring: dict | None = None,
                   minutes_table: dict | None = None,
                   spread: float = config.SIM_MINUTES_SPREAD,
                   rate_spread: float = config.SIM_RATE_SPREAD,
                   tail_gamma: float = config.SIM_TAIL_GAMMA,
                   pace_shock_sd: float = PACE_SHOCK_SD,
                   enforce_team_minutes: bool = True,
                   include_sampling: bool = True,
                   return_draws: bool = False) -> tuple:
    """
    Simulate a slate as whole games.

    `proj` is project.project_slate output: one row per player, carrying
    game_id, team, minutes_proj, the per-stat rate columns and their
    `<stat>__sd` posterior standard deviations.

    Returns (players, teams) by default:
        players  one row per player with p25/p50/p75/p05/p95 and mean for each
                 stat, plus direction-aware `<stat>_floor` and
                 `<stat>_ceiling`
        teams    one row per team-game with the same quantiles on team totals,
                 which only mean anything because the draws are coherent

    With `return_draws=True`, returns (players, teams, fp_draws): fp_draws is
    the raw simulated fantasy-point draw, shape (n_players, n_sims) float32,
    row order matching `players["player_id"]` -- the same (n_players, nsim)
    convention a raw-draw consumer (e.g. a DFS lineup-scoring pipeline that
    sums simulated FP across a lineup's players) expects. Nothing about the
    simulation itself changes; this only exposes the array that was already
    being computed and collapsed into quantiles.

    The pace shock reaches the rates through the SAME elasticities the point
    projection uses (config.ELASTICITY), so a fast draw raises shot attempts
    and rebounds by the amounts the model already believes, rather than by a
    second set of numbers invented here.

    Counts are drawn Binomial(minutes, rate), and `spread`/`rate_spread`/
    `tail_gamma` default to the backtested values in config.py -- see its
    "Simulator calibration" section for how they were tuned and the
    documented, still-open ast/stl/blk/tov/reb over-coverage.
    """
    rng = np.random.default_rng(seed)
    scoring = scoring or simulate.DK_NBA_SIM
    table = minutes_table or simulate.MINUTES_RESID_Q
    n = len(proj)
    if n == 0:
        return pd.DataFrame(), pd.DataFrame()

    proj = proj.reset_index(drop=True)
    gid = proj["game_id"].to_numpy() if "game_id" in proj else np.zeros(n, int)
    team = (proj["team"].to_numpy() if "team" in proj
            else np.zeros(n, dtype=object))

    # ---- game-level pace shock, shared by both teams in a game -----------
    games = pd.unique(gid)
    gidx = {g: i for i, g in enumerate(games)}
    gcol = np.array([gidx[g] for g in gid])
    pace = np.exp(rng.normal(0.0, pace_shock_sd, size=(n_sims, len(games))))
    pace_p = pace[:, gcol]                        # per player per draw

    # ---- overtime, also a game-level event -------------------------------
    ot = rng.random((n_sims, len(games))) < OT_PROB
    team_target = config.MINUTES["team_minutes"] + ot * OT_MINUTES
    target_p = team_target[:, gcol]

    # ---- minutes: calibrated marginal draw, then renormalized to target --
    m = simulate._draw_minutes(
        proj["minutes_proj"].to_numpy(dtype="float64"), n_sims, rng,
        table, spread, tail_gamma)

    if enforce_team_minutes:
        # Renormalize within each team-game so every draw is a legal box
        # score. Net, the per-player band comes out about 2.9% WIDER than the
        # independent simulator's, with coverage holding at 0.519 on minutes
        # and 0.525 on fantasy points.
        #
        # That width does NOT come from the pace shock, which is shared and
        # therefore swamped per-player by minutes and sampling noise -- its
        # per-player effect flips sign even at 12,000 draws. The overtime draw
        # is the plausible source, since it adds real minute variance. The
        # shock earns its place at TEAM level, where a common factor does not
        # average out: it widens the team points sd from 19.18 to 19.49.
        keys = pd.Series([f"{a}|{b}" for a, b in zip(gid, team)])
        for k in keys.unique():
            cols = np.where(keys.to_numpy() == k)[0]
            tot = m[:, cols].sum(axis=1, keepdims=True)
            tgt = target_p[:, cols[0]][:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.where(tot > 0, tgt / tot, 0.0)
            m[:, cols] = np.clip(m[:, cols] * scale, 0.0,
                                 config.MINUTES["min_ceiling"])

    # ---- rate draws, then the shared pace shock --------------------------
    rate = {}
    for stat in config.VOLUME_RATES:
        mu = proj[stat].to_numpy(dtype="float64")
        mu = np.where(np.isfinite(mu) & (mu > 0), mu, 1e-6)
        sd = proj.get(f"{stat}__sd")
        if sd is None:
            draws = np.tile(mu, (n_sims, 1))
        else:
            sd = np.nan_to_num(sd.to_numpy(dtype="float64"), nan=0.0) * rate_spread
            sd = np.maximum(sd, 1e-9)
            shape = np.square(mu / sd)
            draws = rng.gamma(shape, mu / shape, size=(n_sims, n))
        b_pace = config.ELASTICITY.get(stat, (0.0, 0, 0, 0))[0]
        rate[stat] = draws * np.power(pace_p, b_pace)

    for stat in config.SHARE_RATES:
        mu = np.clip(proj[stat].to_numpy(dtype="float64"), 1e-4, 0.999)
        sd = proj.get(f"{stat}__sd")
        if sd is None:
            draws = np.tile(mu, (n_sims, 1))
        else:
            sd = np.nan_to_num(sd.to_numpy(dtype="float64"), nan=0.0) * rate_spread
            var = np.clip(np.square(sd), 1e-12, mu * (1 - mu) * 0.999)
            k = np.maximum(mu * (1 - mu) / var - 1.0, 1.0)
            draws = rng.beta(mu * k, (1 - mu) * k, size=(n_sims, n))
        b_pace = config.ELASTICITY.get(stat, (0.0, 0, 0, 0))[0]
        rate[stat] = np.clip(draws * np.power(pace_p, b_pace), 1e-4, 0.999)

    # ---- counts: Binomial(minutes, rate), not Poisson(rate*minutes) ------
    # Poisson's Var=lambda is a hard floor; Binomial's Var=n*p*(1-p) sits
    # strictly below it whenever p>0, and minutes is a natural trial count.
    # See config.py's "Simulator calibration" section for the backtest and
    # its documented limits (small per-minute rates like ast/stl/blk barely
    # benefit, since Binomial(n,p) is numerically close to Poisson there).
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

    def _n(x):
        return np.nan_to_num(x, nan=0.0, posinf=0.0).clip(0).astype("int64")

    def _p(stat, default):
        v = np.nan_to_num(rate[stat], nan=default, posinf=default)
        return np.clip(v, 0.0, 1.0)

    fga, fta = draws["fga"], draws["fta"]
    if include_sampling:
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
        "oreb": draws["oreb"], "dreb": draws["dreb"],
        "ast": draws["ast"], "stl": draws["stl"], "blk": draws["blk"],
        "tov": draws["tov"], "pf": draws["pf"],
        "fga": fga, "fta": fta, "pt3m": pt3m,
        "fgm": fg2m + pt3m, "ftm": ftm,
    }
    sim["fp"] = sum(sim[k] * v for k, v in scoring.items() if k in sim)
    cats = np.stack([sim["pts"], sim["reb"], sim["ast"], sim["stl"], sim["blk"]])
    sim["dd"] = ((cats >= 10).sum(axis=0) >= 2).astype("float64")
    sim["td"] = ((cats >= 10).sum(axis=0) >= 3).astype("float64")

    players = _quantiles(proj, sim, gid, team)
    teams = _team_quantiles(proj, sim, gid, team, scoring)
    if return_draws:
        # sim["fp"] is (n_sims, n_players); transpose to (n_players, n_sims)
        # to match the raw-draw convention a lineup-scoring consumer expects.
        return players, teams, sim["fp"].T.astype(np.float32)
    return players, teams


def _quantiles(proj, sim, gid, team) -> pd.DataFrame:
    out = {"player_id": proj["player_id"].to_numpy(),
           "game_id": gid, "team": team}
    for c in ("position", "usage_tier", "minutes_proj"):
        if c in proj.columns:
            out[c] = proj[c].to_numpy()

    for k, v in sim.items():
        q = np.quantile(v, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
        out[f"{k}_p05"], out[f"{k}_p25"] = q[0], q[1]
        out[f"{k}_p50"], out[f"{k}_p75"] = q[2], q[3]
        out[f"{k}_p95"] = q[4]
        out[f"{k}_mean"] = v.mean(axis=0)
        # Direction-aware. For turnovers and fouls the ceiling is the LOW tail.
        good_high = _direction(k)
        out[f"{k}_ceiling"] = q[3] if good_high else q[1]
        out[f"{k}_floor"] = q[1] if good_high else q[3]

    df = pd.DataFrame(out)
    df["fp_iqr"] = df["fp_p75"] - df["fp_p25"]
    df["fp_iqr_rel"] = df["fp_iqr"] / df["fp_p50"].replace(0, np.nan)
    return df


def _team_quantiles(proj, sim, gid, team, scoring) -> pd.DataFrame:
    """Team totals per draw -- only meaningful because the draws are coherent."""
    keys = np.array([f"{a}|{b}" for a, b in zip(gid, team)])
    rows = []
    for k in pd.unique(keys):
        cols = np.where(keys == k)[0]
        g, t = k.split("|", 1)
        rec = {"game_id": g, "team": t, "n_players": len(cols)}
        for stat in ("minutes", "pts", "reb", "ast", "tov", "fga", "fp"):
            if stat not in sim:
                continue
            tot = sim[stat][:, cols].sum(axis=1)
            q = np.quantile(tot, [0.25, 0.50, 0.75])
            rec[f"{stat}_p25"], rec[f"{stat}_p50"], rec[f"{stat}_p75"] = q
            rec[f"{stat}_mean"] = float(tot.mean())
            rec[f"{stat}_sd"] = float(tot.std())
        rows.append(rec)
    return pd.DataFrame(rows)


def ceiling_floor_table(players: pd.DataFrame,
                        stats: list[str] | None = None) -> pd.DataFrame:
    """
    Tidy long-format floor / median / ceiling per player per stat, with the
    direction recorded so a consumer cannot mix the conventions up.
    """
    stats = stats or ["minutes", "pts", "reb", "ast", "stl", "blk", "tov",
                      "pf", "fga", "pt3m", "fp"]
    rows = []
    for s in stats:
        if f"{s}_p50" not in players.columns:
            continue
        rows.append(pd.DataFrame({
            "player_id": players["player_id"],
            "team": players.get("team"),
            "stat": s,
            "floor": players[f"{s}_floor"],
            "median": players[f"{s}_p50"],
            "ceiling": players[f"{s}_ceiling"],
            "higher_is_better": _direction(s),
        }))
    out = pd.concat(rows, ignore_index=True)
    # A ceiling must be at least as good as the median in the stat's own
    # direction. Cheap invariant, and it catches a flipped sign immediately.
    good = np.where(out["higher_is_better"],
                    out["ceiling"] >= out["median"] - 1e-9,
                    out["ceiling"] <= out["median"] + 1e-9)
    bad = int((~good).sum())
    if bad:
        raise ValueError(f"{bad} rows where the ceiling is worse than the "
                         "median in its own direction")
    return out

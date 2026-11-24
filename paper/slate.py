"""Run the projection + game simulation for one slate date, from this repo.

    preseason sheet + game logs -> posterior rates -> minutes -> environment
    (spread/total from OpticOdds) -> project_slate -> simulate_games(draws)

Draws are saved with the capture (store.save_sim). They are NOT re-derived
later: re-running the model next month would use a different state and quietly
break point-in-time honesty -- a paper trade must be priced from what the model
knew that morning.

STATUS: exercised end-to-end only on synthetic data (see tests/test_slate.py).
Not run against a real preseason sheet / live lines from the build session.
"""

from __future__ import annotations

import pandas as pd

from . import bridge, store
from .teams import canon


def games_from_lines(fixtures_payload, team_map=None):
    """OpticOdds spread/total payload -> one row per TEAM-game for
    market.game_environment (spread from that team's side, negative = favourite).

    Team labels use the competitors' abbreviations, mapped through ingest's
    ESPN->RotoWire table so they join to the preseason sheet. Selection on a
    spread odd is matched on team name OR abbreviation."""
    rows = []
    payloads = fixtures_payload if isinstance(fixtures_payload, list) else [fixtures_payload]
    fixtures = []
    for pl in payloads:     # a payload envelope {"data": [...]}, or a bare fixture
        fixtures.extend(pl.get("data", []) if isinstance(pl, dict) and "data" in pl else [pl])
    for fx in fixtures:
        def comp(side):
            c = (fx.get(side) or [{}])[0]
            return c.get("abbreviation"), c.get("name")
        (h_ab, h_nm), (a_ab, a_nm) = comp("home_competitors"), comp("away_competitors")
        spreads, total = {}, None
        for odd in fx.get("odds", []):
            mk = (odd.get("market_id") or odd.get("market") or "").lower()
            if "spread" in mk and odd.get("points") is not None:
                spreads[str(odd.get("selection"))] = float(odd["points"])
            elif "total" in mk and total is None and odd.get("points") is not None:
                total = float(odd["points"])
        if total is None or not spreads:
            continue
        for ab, nm, opp_ab, is_home in ((h_ab, h_nm, a_ab, 1), (a_ab, a_nm, h_ab, 0)):
            sp = spreads.get(str(nm), spreads.get(str(ab)))
            if sp is None:
                continue
            rows.append({"game_id": fx.get("id"), "team": canon(ab), "opp": canon(opp_ab),
                         "is_home": is_home, "spread": sp, "total": total,
                         "game_date": fx.get("start_date"), "fixture_id": fx.get("id")})
    return pd.DataFrame(rows)


def build_draws(date, preseason, games, logs=None, actives=None, absent=None, b2b_teams=None,
                n_sims=4000, seed=0, pace_prior=None):
    """Returns (players_df, sim_dict, proj_df). `sim_dict` is game_sim's
    return_draws="all" dict; columns align with players_df["player_id"]."""
    m = bridge.model()
    as_of = pd.Timestamp(date)
    st = m.state.build_priors(preseason)
    if logs is not None and len(logs):
        st = m.state.update(st, logs, as_of)
    post = m.state.posterior_rates(st, m.state.population_rates(preseason))
    base = m.minutes.baseline_minutes(preseason, logs, as_of)
    env = m.market.game_environment(games, pace_prior)
    mins = m.minutes.project_minutes(base, env, b2b_teams=b2b_teams, actives=actives,
                                     absent=absent, post=post)
    proj = m.project.project_slate(post, mins, env)
    players, teams, sim = m.game_sim.simulate_games(proj, n_sims=n_sims, seed=seed, return_draws="all")
    return players, sim, proj


def slate_players(preseason, proj=None):
    """Frame for crosswalk.resolve: player_id, name, team."""
    pre = preseason.copy()
    pre.columns = [c.strip() for c in pre.columns]
    name = (pre["firstname"].astype(str) + " " + pre["lastname"].astype(str)).str.strip()
    out = pd.DataFrame({"player_id": pre["newsid"].astype("int64"), "name": name, "team": pre["team"]})
    if proj is not None and "team" in proj:
        # tonight's team (post-trade) wins over the preseason sheet
        tonight = proj.drop_duplicates("player_id").set_index("player_id")["team"]
        out["team"] = out["player_id"].map(tonight).fillna(out["team"])
        out = out[out["player_id"].isin(tonight.index)]
    return out.to_dict("records")


def save_slate(root, date, players, sim):
    store.save_sim(root, date, players["player_id"].to_numpy(), sim)

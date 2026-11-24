"""The slate runner against the REAL model modules, on a synthetic preseason sheet.
Proves the wiring (state -> minutes -> environment -> project -> simulate -> draws
-> prices); says nothing about real-data quality."""
import numpy as np
import pandas as pd
import pytest

from paper import slate, store
from paper.candidates import DrawBook, candidates_for_board


def preseason(teams=("BOS", "NYK"), per_team=10, seed=0):
    rng = np.random.default_rng(seed)
    rows, nid = [], 1000
    for t in teams:
        for i in range(per_team):
            mpg = 34 - i * 2.8
            g = 70
            mins = mpg * g
            fga = mins * rng.uniform(.38, .5) * (1.2 if i < 3 else 0.8) / 1.0 * 0.45
            rows.append({"newsid": nid, "firstname": f"{t}", "lastname": f"Player{'ABCDEFGHIJ'[i]}", "team": t,
                         "position": ["G", "F", "C"][i % 3], "games": g, "minutes": mins,
                         "fga": fga, "fgm": fga * .47, "pt3a": fga * .38, "pt3m": fga * .38 * .36,
                         "fta": fga * .25, "ftm": fga * .25 * .78, "oreb": mins * .03, "dreb": mins * .12,
                         "ast": mins * (.16 if i < 2 else .08), "stl": mins * .03, "blk": mins * .02,
                         "tov": mins * .06, "pf": mins * .08})
            nid += 1
    return pd.DataFrame(rows)


def lines_payload():
    def odd(mk, sel, pts):
        return {"market_id": mk, "selection": sel, "points": pts}
    return {"data": [{"id": "FX1", "start_date": "2026-11-25T00:30:00Z",
                      "home_competitors": [{"abbreviation": "BOS", "name": "Boston Celtics"}],
                      "away_competitors": [{"abbreviation": "NY", "name": "New York Knicks"}],
                      "odds": [odd("point_spread", "Boston Celtics", -4.5), odd("point_spread", "New York Knicks", 4.5),
                               odd("total_points", "Over", 224.5)]}]}


def test_games_from_lines_orients_spread_per_team():
    g = slate.games_from_lines(lines_payload())
    assert len(g) == 2
    bos = g[g["is_home"] == 1].iloc[0]
    assert bos["spread"] == -4.5 and bos["total"] == 224.5          # favourite is negative
    assert g[g["is_home"] == 0].iloc[0]["spread"] == 4.5


def test_build_draws_runs_the_real_model_and_prices_a_line(tmp_path):
    pre = preseason()
    games = slate.games_from_lines(lines_payload())
    # team abbreviations must join to the preseason sheet (RotoWire NYK vs ESPN NY)
    games["team"] = games["team"].replace({"NY": "NYK"}); games["opp"] = games["opp"].replace({"NY": "NYK"})
    players, sim, proj = slate.build_draws("2026-11-24", pre, games, logs=None, n_sims=500, seed=3)
    assert len(players) > 10
    n_players = len(players)
    for k in ("pts", "reb", "ast", "pt3m", "stl", "blk", "tov", "fgm", "fga", "ftm", "fta"):
        assert sim[k].shape == (500, n_players)
    # every simulated game is a legal box: points = 2*fg2m + 3*3pm + ftm  => pts >= pt3m*3 and integral
    assert np.all(sim["pts"] >= 3 * sim["pt3m"]) and np.allclose(sim["pts"], np.round(sim["pts"]))

    store.save_sim(tmp_path, "2026-11-24", players["player_id"], sim)
    ids, loaded = store.load_sim(tmp_path, "2026-11-24")
    assert list(ids) == list(players["player_id"]) and loaded["pts"].shape == (500, n_players)

    # price a PRA line for the top starter straight off the draws
    db = DrawBook(loaded, ids)
    pid = int(players.sort_values("minutes_p50", ascending=False)["player_id"].iloc[0])
    pra = db.draws(pid, "pra")
    assert np.allclose(pra, loaded["pts"][:, db.col[pid]] + loaded["reb"][:, db.col[pid]] + loaded["ast"][:, db.col[pid]])
    line = float(np.median(pra)) + 0.5
    row = {"fixture_id": "FX1", "sportsbook": "fanduel", "stat": "pra", "player_id": pid, "player_name": "x",
           "team": "BOS", "opp": "NYK", "line": line, "over_odds": -110, "under_odds": -110,
           "max_stake": None, "timestamp": None, "start_date": None}
    rows = [{**row, "sportsbook": b} for b in ("fanduel", "draftkings", "betmgm")]
    out = candidates_for_board({"date": "2026-11-24", "slot": "morning", "rows": rows}, db, min_edge=-1)
    over = [c for c in out if c["side"] == "over"][0]
    assert 0.3 < over["p_proj"] < 0.7                    # a median-ish line prices near a coin flip


def test_slate_players_uses_tonights_team_over_the_sheet():
    pre = preseason()
    proj = pd.DataFrame({"player_id": [1000, 1001], "team": ["MIA", "BOS"]})
    sp = {p["player_id"]: p for p in slate.slate_players(pre, proj)}
    assert set(sp) == {1000, 1001} and sp[1000]["team"] == "MIA"      # traded; only tonight's players kept

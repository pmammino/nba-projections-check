"""Results parsing against the real /fixtures/player-results shape (James Harden's
51-point line is copied from a live response; the rest is built to be consistent)."""
import pytest

from paper import results as R

CLE, IND = "D5348BDFEBCC", "F342C875E571"


def fixture(status="completed"):
    return {"id": "202611212B1024B8", "start_date": "2026-11-21T00:30:00Z", "status": status,
            "home_competitors": [{"id": CLE, "name": "Cleveland Cavaliers", "abbreviation": "CLE"}],
            "away_competitors": [{"id": IND, "name": "Indiana Pacers", "abbreviation": "IND"}]}


def player(pid, name, team_id, secs, **s):
    base = dict(assists=0, blocks=0, defensive_rebounds=0, offensive_rebounds=0, field_goals_attempted=0,
                field_goals_made=0, fouls=0, free_throws_attempted=0, free_throws_made=0, steals=0,
                three_point_field_goals_attempted=0, three_point_field_goals_made=0, turnovers=0)
    base.update(s)
    fgm, tpm, ftm = base["field_goals_made"], base["three_point_field_goals_made"], base["free_throws_made"]
    base["points"] = 2 * fgm + tpm + ftm
    base["seconds"], base["minutes"] = secs, round(secs / 60)
    reb = base["offensive_rebounds"] + base["defensive_rebounds"]
    ms = {"player_points": base["points"], "player_rebounds": reb, "player_assists": base["assists"],
          "player_made_threes": tpm, "player_steals": base["steals"], "player_blocks": base["blocks"],
          "player_turnovers": base["turnovers"], "player_double_double": 1.2070100002}
    return {"player": {"id": pid, "name": name}, "team": {"id": team_id, "name": "x"}, "status": "completed",
            "stats": [{"period": "all", "stats": base}, {"period": "period_1", "stats": {"points": 0}}],
            "market_stats": ms, "is_starter": True}


HARDEN = dict(assists=6, blocks=0, defensive_rebounds=6, offensive_rebounds=0, field_goals_attempted=26,
              field_goals_made=15, fouls=3, free_throws_attempted=14, free_throws_made=14, steals=1,
              three_point_field_goals_attempted=11, three_point_field_goals_made=7, turnovers=2)


def payload(players, status="completed"):
    return [{"fixture": fixture(status), "results": players}]


def test_real_line_parses_and_reconciles_with_optics_own_grading():
    g = R.parse_player_results(payload([player("CC707B1EADE5", "James Harden", CLE, 2400, **HARDEN)]))[0]
    r = g["rows"][0]
    assert g["completed"] and not g["errors"] and (g["home"], g["away"]) == ("CLE", "IND")
    assert (r["team"], r["minutes"], r["fgm"], r["pt3m"], r["ftm"]) == ("CLE", 40.0, 15, 7, 14)
    assert 2 * r["fgm"] + r["pt3m"] + r["ftm"] == 51          # Harden's 51


def test_team_comes_from_competitor_id_not_the_team_name():
    p = player("A", "Aaron Nesmith", IND, 1740, assists=7)
    p["team"]["name"] = "something unhelpful"
    assert R.parse_player_results(payload([p]))[0]["rows"][0]["team"] == "IND"


def test_dnp_has_no_row_so_bets_void():
    g = R.parse_player_results(payload([player("A", "Bench Guy", IND, 0), player("B", "Jay Huff", IND, 60)]))[0]
    assert [r["player_name"] for r in g["rows"]] == ["Jay Huff"] and g["rows"][0]["minutes"] == 1.0


def test_box_that_disagrees_with_optics_market_stats_is_flagged():
    p = player("A", "Jarrett Allen", CLE, 1860, field_goals_made=3, field_goals_attempted=7)
    p["market_stats"]["player_points"] = 99
    g = R.parse_player_results(payload([p]))[0]
    assert g["errors"] and "player_points" in g["errors"][0]


def test_unfinished_or_postponed_fixtures_are_not_complete():
    assert not R.parse_player_results(payload([player("A", "x", CLE, 600)], status="inprogress"))[0]["completed"]
    assert R.parse_player_results(payload([], status="postponed"))[0]["dead"]


def test_double_double_encoding_is_never_trusted():
    # Optic's player_double_double is e.g. 1.2070100002, not 0/1: dd/td come from the stat line.
    from paper.markets import stat_actual
    box = dict(fgm=5, pt3m=0, ftm=0, oreb=2, dreb=8, ast=7, stl=0, blk=2, tov=0, fga=6, fta=1)
    assert stat_actual(box, "dd") == 1.0 and stat_actual(box, "td") == 0.0


def team_results(rows, home="CLE", away="IND"):
    def tot(team):
        mine = [r for r in rows if r["team"] == team]
        return {"pts": sum(2 * r["fgm"] + r["pt3m"] + r["ftm"] for r in mine),
                "stats": {"assists": sum(r["ast"] for r in mine), "blocks": sum(r["blk"] for r in mine),
                          "steals": sum(r["stl"] for r in mine),
                          "field_goals_attempted": sum(r["fga"] for r in mine),
                          "free_throws_attempted": sum(r["fta"] for r in mine),
                          "three_point_field_goals_made": sum(r["pt3m"] for r in mine)}}
    h, a = tot(home), tot(away)
    return [{"fixture": fixture(), "scores": {"home": {"total": h["pts"]}, "away": {"total": a["pts"]}},
             "stats": {"home": [{"period": "all", "stats": h["stats"]}], "away": [{"period": "all", "stats": a["stats"]}]}}]


def test_verify_box_passes_when_complete_and_catches_a_missing_player():
    players = [player("A", "a", CLE, 1800, field_goals_made=5, field_goals_attempted=9, assists=3),
               player("B", "b", CLE, 1200, field_goals_made=2, field_goals_attempted=4, assists=1),
               player("C", "c", IND, 1800, field_goals_made=4, field_goals_attempted=8, assists=2)]
    g = R.parse_player_results(payload(players))[0]
    t = R.parse_optic_results(team_results(g["rows"]))[0]
    assert R.verify_box(g["rows"], t) == [] and t["completed"]
    short = [r for r in g["rows"] if r["player_name"] != "b"]
    assert any("points" in m or "assists" in m for m in R.verify_box(short, t))


def test_attach_by_optic_id_then_name_then_traded_player():
    players = [{"player_id": 1, "name": "James Harden", "team": "LAC"},       # sheet says LAC; he plays CLE
               {"player_id": 2, "name": "Jarrett Allen", "team": "CLE"},
               {"player_id": 3, "name": "Obi Toppin", "team": "IND"}]
    rows = R.parse_player_results(payload([player("H", "James Harden", CLE, 2400, **HARDEN),
                                           player("J", "Jarrett Allen", CLE, 1800, assists=4),
                                           player("T", "Obi Toppin", IND, 1500, assists=2),
                                           player("Z", "Nobody Known", IND, 600)]))[0]["rows"]
    m, un = R.attach_player_ids(rows, players, id_map={"T": 3})
    got = {r["player_name"]: (r["player_id"], r["team"]) for r in m}
    assert got["Obi Toppin"] == (3, "IND")                       # by Optic id
    assert got["Jarrett Allen"] == (2, "CLE")                    # by name + team
    assert got["James Harden"] == (1, "CLE")                     # traded: name alone, box team wins
    assert [u["name"] for u in un] == ["Nobody Known"]


def test_id_map_round_trips_and_logs_never_double_count(tmp_path):
    p = tmp_path / "ids.csv"
    R.remember_ids(p, [{"optic_player_id": "AB12", "player_id": 5, "player_name": "x"}])
    R.remember_ids(p, [{"optic_player_id": "CD34", "player_id": 6, "player_name": "y"}])
    assert R.read_id_map(p) == {"AB12": 5, "CD34": 6}
    row = {c: 0 for c in R.LOG_COLS} | {"player_id": 5, "game_id": "g1", "game_date": "2026-11-20", "team": "CLE",
                                          "player_name": "x", "optic_player_id": "AB12"}
    lp = tmp_path / "logs.csv"
    assert R.append_logs(lp, [row]) == 1 and R.append_logs(lp, [row]) == 0
    assert len(R.read_logs(lp)) == 1

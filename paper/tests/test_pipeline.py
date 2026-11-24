"""Integration: Optic-shaped payload -> normalize -> crosswalk -> candidates ->
personas -> settle -> CLV -> filter report, all on synthetic data."""
import numpy as np
import pytest

from paper import candidates, capture, crosswalk, normalize, optic, replay, store
from paper.candidates import DrawBook
from paper.paper_filter import apply_filters, preset_filters


def odd(book, market, player, side, price, line, fx="F1"):
    return {"id": f"{fx}:{book}:{market}:{player}:{side}", "sportsbook": book, "market": market.replace("_", " ").title(),
            "market_id": market, "name": f"{player} {side} {line}", "price": price, "points": line,
            "selection": player, "selection_line": side, "player_id": "HEX" + player[:3], "team_id": None,
            "timestamp": 1789468082.3, "is_main": True, "limits": {"max": 500}}


def fixture(odds, fx="F1"):
    return {"id": fx, "start_date": "2026-11-25T00:30:00Z", "status": "unplayed", "is_live": False,
            "home_competitors": [{"id": "h", "name": "Boston Celtics", "abbreviation": "BOS"}],
            "away_competitors": [{"id": "a", "name": "New York Knicks", "abbreviation": "NY"}],
            "odds": odds}


def pts_market(book, o, u, line=24.5, player="Jaylen Brown"):
    return [odd(book, "player_points", player, "over", o, line), odd(book, "player_points", player, "under", u, line)]


# ---- normalize ---------------------------------------------------------------
def test_pairs_both_sides_and_keys_on_line():
    payload = {"data": [fixture(pts_market("DraftKings", -115, -105, 24.5) + pts_market("DraftKings", -110, -110, 25.5))]}
    rows, diag = normalize.normalize_payloads([payload])
    assert len(rows) == 2 and {r["line"] for r in rows} == {24.5, 25.5}
    assert all(not r["one_sided"] for r in rows)
    assert rows[0]["home"] == "BOS" and rows[0]["player_name"] == "Jaylen Brown"


def test_one_sided_unmatched_markets_and_team_entries_are_diagnosed():
    team_entry = {**odd("DraftKings", "player_points", "Boston Celtics", "over", -110, 24.5), "player_id": None, "team_id": "h"}
    payload = {"data": [fixture([odd("FanDuel", "player_points", "Jaylen Brown", "over", -120, 24.5),
                                 odd("FanDuel", "player_points_combo_", "Jaylen Brown", "over", -120, 24.5),
                                 odd("FanDuel", "1st_half_player_points", "Jaylen Brown", "over", -120, 12.5),
                                 team_entry])]}
    rows, diag = normalize.normalize_payloads([payload])
    assert len(rows) == 1 and rows[0]["one_sided"] and rows[0]["under_odds"] is None
    assert diag["team_entries"] == 1
    assert set(diag["unmatched_markets"]) == {"player_points_combo_", "1st_half_player_points"}


def test_yes_no_markets_use_implicit_half_line():
    payload = {"data": [fixture([{**odd("DraftKings", "player_double_double", "Jaylen Brown", "yes", 300, None),
                                  "selection_line": "yes", "points": None}])]}
    rows, _ = normalize.normalize_payloads([payload], stats=["dd"])
    assert rows[0]["line"] == 0.5 and rows[0]["over_odds"] == 300


# ---- crosswalk ---------------------------------------------------------------
PLAYERS = [{"player_id": 1, "name": "Jaylen Brown", "team": "BOS"},
           {"player_id": 2, "name": "Nikola Jokić", "team": "DEN"},
           {"player_id": 3, "name": "Jalen Williams", "team": "OKC"},
           {"player_id": 4, "name": "Jalen Williams", "team": "PHX"},
           {"player_id": 5, "name": "Jabari Smith Jr.", "team": "HOU"}]


def test_crosswalk_accents_suffixes_and_team_disambiguation():
    rows = [{"player_name": "Nikola Jokic", "fixture_id": "x", "home": "DEN", "away": "LAL"},
            {"player_name": "Jabari Smith", "fixture_id": "x", "home": "HOU", "away": "SAS"},
            {"player_name": "Jalen Williams", "fixture_id": "y", "home": "OKC", "away": "PHX"},   # both teams: ambiguous
            {"player_name": "Jalen Williams", "fixture_id": "z", "home": "OKC", "away": "LAL"},
            {"player_name": "Who Dat", "fixture_id": "z", "home": "OKC", "away": "LAL"}]
    m, un = crosswalk.resolve(rows, PLAYERS)
    assert [r["player_id"] for r in m] == [2, 5, 3]
    assert {(u["name"], u["reason"]) for u in un} == {("Jalen Williams", "ambiguous"), ("Who Dat", "no match")}
    assert m[2]["opp"] == "LAL"


def test_crosswalk_override_wins():
    m, un = crosswalk.resolve([{"player_name": "Jalen Williams", "fixture_id": "y", "home": "OKC", "away": "PHX"}],
                              PLAYERS, overrides={"Jalen Williams": 4})
    assert m[0]["player_id"] == 4 and m[0]["team"] == "PHX" and not un


# ---- optic client ------------------------------------------------------------
class FakeResp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self._b, self.headers, self.text = status, body or {}, headers or {}, "x"

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        return self._b


class FakeSession:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, headers))
        return self.script.pop(0)


def test_query_repeats_array_params_and_key_goes_in_header_only(monkeypatch):
    assert optic.build_query({"sportsbook": ["a", "b"], "x": None, "y": ""}) == "sportsbook=a&sportsbook=b"
    s = FakeSession([FakeResp(200, {"data": []})])
    c = optic.OpticOddsClient(api_key="SECRET", session=s, logger=lambda *_: None)
    c.get("/fixtures/odds", {"sportsbook": ["a", "b"]})
    url, headers = s.calls[0]
    assert "SECRET" not in url and headers["X-Api-Key"] == "SECRET"


def test_retries_429_then_succeeds_and_4xx_fails_fast(monkeypatch):
    monkeypatch.setattr(optic.time, "sleep", lambda *_: None)
    s = FakeSession([FakeResp(429, headers={"retry-after": "1"}), FakeResp(200, {"data": [1]})])
    c = optic.OpticOddsClient(api_key="k", session=s, logger=lambda *_: None)
    assert c.get("/x") == {"data": [1]} and len(s.calls) == 2
    s2 = FakeSession([FakeResp(401)])
    with pytest.raises(optic.OpticOddsError):
        optic.OpticOddsClient(api_key="k", session=s2, logger=lambda *_: None).get("/x")
    assert len(s2.calls) == 1


def test_fan_out_respects_five_book_five_fixture_caps():
    s = FakeSession([FakeResp(200, {"data": []}) for _ in range(20)])
    c = optic.OpticOddsClient(api_key="k", session=s, rate=(1000, 1), logger=lambda *_: None)
    c.get_odds_for_fixtures([f"f{i}" for i in range(11)], [f"b{i}" for i in range(7)], ["m"])
    assert len(s.calls) == 3 * 2                      # ceil(11/5) x ceil(7/5)
    s = FakeSession([FakeResp(200, {"data": []}) for _ in range(20)])
    c = optic.OpticOddsClient(api_key="k", session=s, rate=(1000, 1), logger=lambda *_: None)
    c.get_historical_odds(["f1", "f2", "f3"], ["b1"], ["m"])
    assert len(s.calls) == 3                          # one fixture per historical request


def test_missing_key_is_an_error(monkeypatch):
    monkeypatch.delenv("OPTICODDS_API_KEY", raising=False)
    with pytest.raises(optic.OpticOddsError):
        optic.OpticOddsClient(api_key=None, session=object())


def test_slate_window_covers_us_evening_tips():
    s, e = capture.slate_window("2026-11-24")
    assert s == "2026-11-24T10:00:00Z" and e == "2026-11-25T10:00:00Z"


# ---- end to end -----------------------------------------------------------------
def make_sim(n_sims=4000, seed=0):
    """Two players. Brown's points ~ N(30,6): the books say 24.5 (a huge edge on the over)."""
    rng = np.random.default_rng(seed)
    pts = np.round(np.clip(rng.normal([30, 18], [6, 5], size=(n_sims, 2)), 0, None))
    reb = np.round(np.clip(rng.normal([6, 8], [2, 3], size=(n_sims, 2)), 0, None))
    ast = np.round(np.clip(rng.normal([5, 3], [2, 1.5], size=(n_sims, 2)), 0, None))
    z = np.zeros((n_sims, 2))
    return {"pts": pts, "reb": reb, "ast": ast, "pt3m": z + 2, "stl": z, "blk": z, "tov": z + 2,
            "fgm": z + 8, "fga": z + 18, "ftm": z + 3, "fta": z + 4, "oreb": z + 1, "dreb": z + 5}


def quote_row(book, o, u, line, pid=1, name="Jaylen Brown", stat="pts", fx="F1"):
    return {"fixture_id": fx, "sportsbook": book, "stat": stat, "optic_player_id": "H", "player_id": pid,
            "player_name": name, "team": "BOS", "opp": "NY", "home": "BOS", "away": "NY", "line": line,
            "over_odds": o, "under_odds": u, "one_sided": o is None or u is None, "max_stake": 500,
            "timestamp": 1.0, "start_date": "2026-11-25T00:30:00Z"}


def seed_day(root, date, closing_line=25.5, actual_pts=33):
    store.save_sim(root, date, [1, 2], make_sim())
    morning = [quote_row(b, -110, -110, 24.5) for b in ("draftkings", "fanduel", "betmgm", "caesars")]
    morning += [quote_row("pinnacle", -108, -108, 24.5), quote_row("circa_sports", -110, -110, 24.5)]
    # DraftKings is the stale soft price: Brown over 24.5 at +100
    morning[0] = quote_row("draftkings", 100, -120, 24.5)
    store.save_board(root, date, "morning", morning)
    store.save_board(root, date, "pregame", morning)
    closing = [quote_row(b, -125, 105, closing_line) for b in ("draftkings", "fanduel", "betmgm", "pinnacle")]
    store.save_board(root, date, "closing", closing)
    capture.rebuild_candidates(root, date)
    store.save_actuals(root, date, [
        {"date": date, "player_id": 1, "team": "BOS", "minutes": 36, "pts": actual_pts, "reb": 6, "oreb": 1,
         "dreb": 5, "ast": 5, "stl": 1, "blk": 0, "tov": 2, "pf": 2, "fga": 20, "fgm": 13, "fta": 6, "ftm": 5, "pt3m": 2}])


def test_end_to_end(tmp_path):
    root = tmp_path
    seed_day(root, "2026-11-24")
    cands = store.load_candidates(root, "2026-11-24")
    over = [c for c in cands if c["player_id"] == 1 and c["stat"] == "pts" and c["side"] == "over"
            and c["book"] == "draftkings" and c["slot"] == "morning"]
    assert len(over) == 1 and over[0]["p_proj"] > 0.8 and over[0]["bettable"]
    # reference books are priced but flagged unbettable
    assert any(c["book"] == "pinnacle" and not c["bettable"] for c in cands)
    # the closing board is never a candidate that gets bet
    res = replay.replay(root, write=True)
    fh = res["firehose"]["rows"]
    assert fh and all(r["slot"] in ("morning", "pregame") and r["book"] != "pinnacle" for r in fh)
    top = [r for r in fh if r["player_id"] == 1 and r["stat"] == "pts"][0]
    assert top["status"] == "won" and top["actual"] == 33
    assert top["pnl_units"] == pytest.approx(top["stake_units"] * (odds_dec(top["odds"]) - 1))
    # The close moved to 25.5, so it is a different bet: no numeric CLV is invented for it.
    assert top["close_fair"] is None and top["clv"] is None
    assert (store.ledger_path(root, "firehose", "2026-11-24")).exists()


def odds_dec(o):
    return 1 + o / 100 if o > 0 else 1 + 100 / -o


def test_dnp_is_void_and_unfinished_game_is_pending(tmp_path):
    root = tmp_path
    seed_day(root, "2026-11-24")
    # Brown absent from the box score of a FINAL BOS game => void (book refunds)
    store.save_actuals(root, "2026-11-24", [
        {"date": "2026-11-24", "player_id": 99, "team": "BOS", "minutes": 20, "pts": 5, "reb": 1, "oreb": 0,
         "dreb": 1, "ast": 0, "stl": 0, "blk": 0, "tov": 0, "pf": 0, "fga": 3, "fgm": 2, "fta": 1, "ftm": 1, "pt3m": 0}])
    rows = replay.replay(root, write=False)["firehose"]["rows"]
    assert rows and all(r["status"] == "void" and r["pnl_units"] == 0.0 for r in rows)
    # no box score at all yet => pending, never a loss
    (root / "actuals" / "2026-11-24.csv").unlink()
    rows = replay.replay(root, write=False)["firehose"]["rows"]
    assert all(r["status"] == "pending" and r["pnl_units"] is None for r in rows)


def test_replay_is_deterministic_and_rebuilds_from_scratch(tmp_path):
    seed_day(tmp_path, "2026-11-24")
    a = replay.replay(tmp_path, write=True)
    b = replay.replay(tmp_path, write=True)
    assert [r["roi"] for r in a.values()] == [r["roi"] for r in b.values()]
    assert a["firehose"]["rows"] == b["firehose"]["rows"]


def test_filter_report_and_sharp_preset(tmp_path):
    seed_day(tmp_path, "2026-11-24")
    r = replay.filter_report(tmp_path, preset_filters("realistic"))
    assert r["summary"]["bets"] >= 1 and r["verdict"] == "too few bets to call"
    pool, complete = replay.settled_pool(tmp_path)
    assert complete == ["2026-11-24"]
    sharp = apply_filters(pool, preset_filters("sharp"))
    # sharp preset: every bet is at a listed book and the sharp books do not contradict it
    assert all(b["p_sharp"] is not None and b["p_sharp"] - b["implied"] >= 0 for b in sharp)


def test_point_in_time_pricing_ignores_same_and_later_days(tmp_path):
    from paper.model_fit import fit_model
    # closing line == morning line so each market has an early/late pair to assess
    seed_day(tmp_path, "2026-11-24", closing_line=24.5)
    seed_day(tmp_path, "2026-11-25", closing_line=24.5)
    m, shares = fit_model(tmp_path, "2026-11-24")
    assert m["dates_used"] == []                       # first day prices on the prior alone
    m2, _ = fit_model(tmp_path, "2026-11-25")
    assert m2["dates_used"] == ["2026-11-24"]


def test_clv_is_measured_against_the_close_without_the_book_we_bet(tmp_path):
    seed_day(tmp_path, "2026-11-24", closing_line=24.5)       # the line held
    rows = replay.replay(tmp_path, write=False)["firehose"]["rows"]
    top = [r for r in rows if r["player_id"] == 1 and r["stat"] == "pts" and r["side"] == "over"][0]
    # we took the over at +100 (implied 50%); the other books closed it at -125/105-ish => fair > 50%
    assert top["close_fair"] is not None and top["clv"] > 0
    assert top["clv"] == pytest.approx(top["close_fair"] - top["implied"], abs=1e-4)

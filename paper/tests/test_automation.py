"""The whole scheduler against fakes: RotoWire JSON -> slate (real model) -> morning /
pregame / closing captures per fixture -> grading from Optic player-results -> ledgers
+ report. Everything external is faked; everything internal is real."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from paper import automation as A, store
from test_results import fixture as results_fixture, player as box_player, team_results
from test_slate import preseason

UTC = timezone.utc
TIP = datetime(2026, 11, 25, 0, 30, tzinfo=UTC)          # 7:30pm ET on slate day 2026-11-24
DAY = "2026-11-24"
FX = "FXBOSNYK"


def at(h, m=0, day=24):
    return datetime(2026, 11, day, h, m, tzinfo=UTC)


# ---- scheduling (pure) -------------------------------------------------------------
def fx(fid, tip):
    return {"id": fid, "start": tip}


def test_slate_day_spans_ten_utc_to_ten_utc():
    assert A.slate_date(at(9, 59, 25)) == "2026-11-24"      # still last night's slate
    assert A.slate_date(at(10, 0, 25)) == "2026-11-25"
    assert A.slate_date(at(0, 10, 25)) == "2026-11-24"      # after UTC midnight, still the 24th's tips


def test_nothing_due_before_the_morning_window_or_after_tip():
    due, missed = A.plan_captures(at(12), [fx("a", TIP)], {})
    assert not any(due.values())
    due, _ = A.plan_captures(TIP + timedelta(minutes=1), [fx("a", TIP)], {})
    assert not any(due.values())                              # tipped: nothing can be bet


def test_each_slot_opens_relative_to_its_own_tip():
    now = at(22, 40)
    early = fx("early", now + timedelta(hours=1, minutes=30))    # tips in 90 min
    late = fx("late", now + timedelta(hours=4, minutes=30))      # tips in 4.5 h
    due, missed = A.plan_captures(now, [early, late], {})
    assert due["morning"] == ["early", "late"]
    assert due["pregame"] == ["early"]                            # inside ITS 2h window; late is still too far out
    assert due["closing"] == [] and missed == []
    # 20 minutes before the late game: only it is closing, and the early one has tipped
    now2 = late["start"] - timedelta(minutes=20)
    due, _ = A.plan_captures(now2, [early, late], {"morning": {"early", "late"}, "pregame": {"early", "late"}})
    assert due["closing"] == ["late"]


def test_captured_fixtures_are_not_captured_twice_and_missed_windows_are_reported():
    f = fx("a", TIP)
    have = {"morning": {"a"}, "pregame": {"a"}}
    due, missed = A.plan_captures(TIP - timedelta(minutes=20), [f], have)
    assert due["closing"] == ["a"] and not due["morning"] and not due["pregame"]
    due, missed = A.plan_captures(TIP - timedelta(minutes=20), [f], {"morning": {"a"}})
    assert ("pregame", "a") in missed and not due["pregame"]  # never backfilled with a later price


def test_merge_board_replaces_only_the_named_fixtures():
    from paper.capture import merge_board
    old = [{"fixture_id": "a", "v": 1}, {"fixture_id": "b", "v": 1}]
    assert merge_board(old, [{"fixture_id": "a", "v": 2}], ["a"]) == [{"fixture_id": "b", "v": 1}, {"fixture_id": "a", "v": 2}]


# ---- fakes --------------------------------------------------------------------------
def _quartiles(mean):
    from paper import dist
    c = np.cumsum(dist.nb_pmf(mean, 8.0, 80))
    return int(np.searchsorted(c, 0.25)), int(np.searchsorted(c, 0.75))


# feed column -> (preseason total column, divided by games)
_FEED_COLS = {"PTS": None, "REB": None, "AST": "ast", "THREEPM": "pt3m", "STL": "stl", "BLK": "blk", "TO": "tov",
              "PF": "pf", "FGA": "fga", "FGM": "fgm", "FTA": "fta", "FTM": "ftm", "OREB": "oreb", "DREB": "dreb",
              "THREEPA": "pt3a"}


def rw_rows(pre, kind):
    """RotoWire-shaped rows. kind: 'pre' | 'mean' | 'C' | 'F'. The ceiling feed is
    direction-aware like game_sim's: for TO and PF the 'ceiling' is the LOW quartile."""
    if kind == "pre":
        return [{**{k: v for k, v in r.items() if k not in ("ast", "stl", "blk", "tov")},
                 "assists": r["ast"], "steals": r["stl"], "blocks": r["blk"], "turnovers": r["tov"]}
                for r in pre.to_dict("records")]
    out = []
    for r in pre.to_dict("records"):
        g = r["games"]
        row = {"id": str(r["newsid"]), "firstName": r["firstname"], "lastName": r["lastname"],
               "player": f"{r['firstname']} {r['lastname']}", "team": r["team"], "pos": r["position"], "G": "1",
               "opp": "NYK" if r["team"] == "BOS" else "BOS", "DBLDBL": "0.01000", "TRPDBL": "0.00000",
               "MIN": f"{r['minutes'] / g:.5f}"}
        per = {"PTS": (2 * r["fgm"] + r["pt3m"] + r["ftm"]) / g, "REB": (r["oreb"] + r["dreb"]) / g}
        for col, src in _FEED_COLS.items():
            if src:
                per[col] = r[src] / g
        for col, mean in per.items():
            lo, hi = _quartiles(mean)
            if kind == "mean":
                row[col] = f"{mean:.5f}"
            else:
                low_tail_is_ceiling = col in ("TO", "PF")
                row[col] = f"{float(lo if (kind == 'C') == low_tail_is_ceiling else hi):.5f}"
        out.append(row)
    return out


class FakeSession:
    """RotoWire: the preseason sheet and the three daily feeds, by URL."""
    def __init__(self, pre):
        self.pre, self.urls = pre, []

    def get(self, url, headers=None, timeout=None):
        self.urls.append(url)
        kind = "pre" if "type=pre" in url else "C" if "ceilFloor=C" in url else "F" if "ceilFloor=F" in url else "mean"
        rows = rw_rows(self.pre, kind)

        class R:
            ok = True
            status_code = 200
            text = ""
            def json(self_):
                return rows
        return R()


def hexid(name):
    return "HX" + "".join(c for c in name if c.isalnum())[:10].upper()


class FakeOptic:
    def __init__(self, pre, state):
        self.pre, self.state = pre, state          # state: {"results": bool}
        self.calls = []

    # fixtures: one BOS (home) vs NYK game
    def get_fixtures(self, start_date=None, end_date=None, status=None, **_):
        self.calls.append(("fixtures", status))
        done = self.state.get("final")
        if status == "completed":
            return [{"id": FX, "status": "completed"}] if done else []
        return [{"id": FX, "start_date": "2026-11-25T00:30:00Z", "status": "unplayed", "is_live": False}]

    def get_sportsbooks(self):
        return [{"id": b, "name": b} for b in
                ("draftkings", "fanduel", "betmgm", "caesars", "betrivers", "hardrock", "thescore", "pinnacle", "circa_sports")]

    def _fixture(self, odds):
        return {"id": FX, "start_date": "2026-11-25T00:30:00Z", "status": "unplayed", "is_live": False,
                "home_competitors": [{"id": "HOME", "name": "Boston Celtics", "abbreviation": "BOS"}],
                "away_competitors": [{"id": "AWAY", "name": "New York Knicks", "abbreviation": "NY"}], "odds": odds}

    def get_odds_for_fixtures(self, fixture_ids, sportsbooks, markets, **_):
        if "point_spread" in markets:
            return [{"data": [self._fixture([
                {"market_id": "point_spread", "selection": "Boston Celtics", "points": -4.5},
                {"market_id": "point_spread", "selection": "New York Knicks", "points": 4.5},
                {"market_id": "total_points", "selection": "Over", "points": 224.5}])]}]
        odds = []
        for _, p in self.pre.iterrows():
            if p["team"] not in ("BOS", "NYK") or p["minutes"] / p["games"] < 20:
                continue
            name = f"{p['firstname']} {p['lastname']}"
            line = max(1.5, round(p["minutes"] / p["games"] * (p["fgm"] * 2 + p["pt3m"] + p["ftm"]) / p["minutes"]) - 0.5)
            for book, (o, u) in {"draftkings": (+105, -125), "fanduel": (-120, 100), "betmgm": (-115, -105),
                                 "caesars": (-118, -102), "pinnacle": (-112, -108)}.items():
                for side, price in (("over", o), ("under", u)):
                    odds.append({"sportsbook": book, "market": "Player Points", "market_id": "player_points",
                                 "selection": name, "selection_line": side, "price": price, "points": line,
                                 "player_id": hexid(name), "team_id": None, "timestamp": 1.0,
                                 "limits": {"max": 500}})
        return [{"data": [self._fixture(odds)]}]

    # results
    def _box(self):
        rng = np.random.default_rng(7)
        out = []
        for _, p in self.pre.iterrows():
            if p["team"] not in ("BOS", "NYK") or p["minutes"] / p["games"] < 20:
                continue
            name = f"{p['firstname']} {p['lastname']}"
            fgm = int(rng.integers(2, 12))
            out.append((name, "HOME" if p["team"] == "BOS" else "AWAY", dict(
                field_goals_made=fgm, field_goals_attempted=fgm * 2, three_point_field_goals_made=1,
                three_point_field_goals_attempted=3, free_throws_made=2, free_throws_attempted=3,
                offensive_rebounds=1, defensive_rebounds=int(rng.integers(1, 9)), assists=int(rng.integers(0, 9)),
                steals=1, blocks=0, turnovers=2, fouls=2)))
        return out

    def _rows(self):
        return [box_player(hexid(n), n, tid, 1800, **s) for n, tid, s in self._box()]

    def get_fixture_results(self, fixture_ids):
        from paper import results as R
        rows = []
        for r in R.parse_player_results([{"fixture": results_fixture(), "results": self._rows()}])[0]["rows"]:
            rows.append({**r, "team": "CLE" if r["team"] == "CLE" else r["team"]})
        # the shared fixture helper uses CLE/IND ids; re-key players onto BOS/NYK for this game
        return self._team_totals()

    def _team_totals(self):
        def tot(tid):
            s = [x for n, t, x in self._box() if t == tid]
            return ({"total": sum(2 * x["field_goals_made"] + x["three_point_field_goals_made"] + x["free_throws_made"] for x in s)},
                    {"assists": sum(x["assists"] for x in s), "blocks": 0, "steals": sum(x["steals"] for x in s),
                     "field_goals_attempted": sum(x["field_goals_attempted"] for x in s),
                     "free_throws_attempted": sum(x["free_throws_attempted"] for x in s),
                     "three_point_field_goals_made": sum(x["three_point_field_goals_made"] for x in s)})
        (hs, hst), (as_, ast) = tot("HOME"), tot("AWAY")
        fx_ = self._res_fixture()
        return [{"fixture": fx_, "scores": {"home": hs, "away": as_},
                 "stats": {"home": [{"period": "all", "stats": hst}], "away": [{"period": "all", "stats": ast}]}}]

    def _res_fixture(self):
        done = self.state.get("final")
        return {"id": FX, "start_date": "2026-11-25T00:30:00Z", "status": "completed" if done else "inprogress",
                "home_competitors": [{"id": "HOME", "name": "Boston Celtics", "abbreviation": "BOS"}],
                "away_competitors": [{"id": "AWAY", "name": "New York Knicks", "abbreviation": "NY"}]}

    def get_player_results(self, fixture_ids):
        rows = []
        for n, tid, s in self._box():
            rows.append(box_player(hexid(n), n, tid, 1800, **s))
        return [{"fixture": self._res_fixture(), "results": rows}]


def test_a_full_slate_day_runs_itself(tmp_path):
    pre = preseason()
    state = {"final": False}
    client, sess = FakeOptic(pre, state), FakeSession(pre)
    logs = []
    log = logs.append

    # 10:30 ET: morning window opens -> slate built from RotoWire, morning captured
    assert A.auto(tmp_path, client, now=at(15, 30), log=log, n_sims=400, session=sess) == []
    assert store.board_slots(tmp_path, DAY) == ["morning"]
    assert store.load_sim(tmp_path, DAY)[1] is not None              # `both`: draws exist for the combos
    from paper import feeds
    snap = feeds.load_snapshot(tmp_path, DAY, "morning")
    assert snap and {r["team"] for r in snap.values()} == {"BOS", "NYK"}
    ids = {r["fixture_id"] for r in store.load_board(tmp_path, DAY, "morning")}
    assert ids == {FX}
    assert (tmp_path / "inputs" / "optic_ids.csv").exists()   # Optic id -> our player_id, persisted

    # running again in the same window does nothing new (no double capture)
    assert A.auto(tmp_path, client, now=at(15, 45), log=log, n_sims=400, session=sess) == []
    assert store.board_slots(tmp_path, DAY) == ["morning"]

    # ~2h before tip -> pregame; ~20 min before -> closing
    assert A.auto(tmp_path, client, now=at(22, 40), log=log, n_sims=400, session=sess) == []
    assert A.auto(tmp_path, client, now=at(0, 12, 25), log=log, n_sims=400, session=sess) == []
    assert store.board_slots(tmp_path, DAY) == ["closing", "morning", "pregame"]
    assert store.load_candidates(tmp_path, DAY)

    # game over -> graded from Optic player-results, ledgers + report written
    state["final"] = True
    assert A.auto(tmp_path, client, now=at(12, 0, 25), log=log, n_sims=400, session=sess) == []
    act = store.load_actuals(tmp_path, DAY)
    assert act and {a["team"] for a in act} == {"BOS", "NYK"}
    ledger = tmp_path / "ledgers" / "firehose" / f"{DAY}.csv"
    assert ledger.exists()
    import csv
    rows = list(csv.DictReader(open(ledger)))
    assert rows and all(r["status"] in ("won", "lost", "push") for r in rows)     # settled by Optic id, none pending
    assert all(r["book"] != "pinnacle" for r in rows)
    report = (tmp_path / "report.md").read_text()
    assert "Personas" in report and "Filtered paper trading" in report
    assert (tmp_path / "inputs" / "logs.csv").exists()                            # game logs grew from the same results
    # grading is idempotent
    n = len(store.load_actuals(tmp_path, DAY))
    A.auto(tmp_path, client, now=at(13, 0, 25), log=log, n_sims=400, session=sess)
    assert len(store.load_actuals(tmp_path, DAY)) == n


def test_a_box_that_disagrees_with_optic_is_not_graded(tmp_path):
    pre = preseason()
    state = {"final": False}
    client, sess = FakeOptic(pre, state), FakeSession(pre)
    A.auto(tmp_path, client, now=at(15, 30), log=lambda *_: None, n_sims=400, session=sess)
    state["final"] = True
    orig = client.get_player_results

    def tampered(ids):
        data = orig(ids)
        data[0]["results"][0]["market_stats"]["player_points"] += 5   # Optic's own grade disagrees with its box
        return data
    client.get_player_results = tampered
    logs = []
    A.auto(tmp_path, client, now=at(12, 0, 25), log=logs.append, n_sims=400, session=sess)
    assert store.load_actuals(tmp_path, DAY) is None
    assert any("disagrees with Optic" in l for l in logs)


def test_one_failing_step_does_not_stop_the_others(tmp_path):
    pre = preseason()
    client = FakeOptic(pre, {"final": False})

    class Boom(Exception):
        pass
    client.get_sportsbooks = lambda: (_ for _ in ()).throw(Boom("books down"))
    logs = []
    errs = A.auto(tmp_path, client, now=at(15, 30), log=logs.append, n_sims=400, session=FakeSession(pre))
    assert any("capture morning" in e for e in errs)
    assert (tmp_path / "report.md").exists()                      # the report step still ran


def test_each_capture_prices_off_the_feed_as_it_stood_then(tmp_path):
    """A pregame bet must use the pregame feed (late news), never the morning's."""
    from paper import feeds
    pre = preseason()
    client, sess = FakeOptic(pre, {"final": False}), FakeSession(pre)
    A.auto(tmp_path, client, now=at(15, 30), log=lambda *_: None, session=sess)
    n_morning = len(sess.urls)
    A.auto(tmp_path, client, now=at(22, 40), log=lambda *_: None, session=sess)
    assert len(sess.urls) == n_morning + 3                           # 3 feeds re-fetched at the pregame capture
    assert feeds.load_snapshot(tmp_path, DAY, "pregame") and feeds.load_snapshot(tmp_path, DAY, "morning")


def test_feeds_only_mode_runs_no_model(tmp_path):
    pre = preseason()
    client, sess = FakeOptic(pre, {"final": False}), FakeSession(pre)
    assert A.auto(tmp_path, client, now=at(15, 30), log=lambda *_: None, session=sess, price_model="feeds") == []
    assert store.load_sim(tmp_path, DAY)[1] is None and not (tmp_path / "inputs" / "preseason.csv").exists()
    assert store.load_candidates(tmp_path, DAY) is not None


def test_a_simulator_failure_does_not_block_the_feed_priced_captures(tmp_path):
    pre = preseason()
    client = FakeOptic(pre, {"final": False})
    real = client.get_odds_for_fixtures
    client.get_odds_for_fixtures = lambda ids, books, markets, **k: (
        (_ for _ in ()).throw(RuntimeError("lines down")) if "point_spread" in markets else real(ids, books, markets, **k))
    logs = []
    errs = A.auto(tmp_path, client, now=at(15, 30), log=logs.append, session=FakeSession(pre))   # default: both
    assert any("build slate" in e for e in errs)                      # the failure is surfaced (job goes red) ...
    assert store.board_slots(tmp_path, DAY) == ["morning"]            # ... but the morning capture still happened
    assert store.load_candidates(tmp_path, DAY)


def test_sim_mode_still_works_for_combos_and_comparison(tmp_path):
    pre = preseason()
    client, sess = FakeOptic(pre, {"final": False}), FakeSession(pre)
    errs = A.auto(tmp_path, client, now=at(15, 30), log=lambda *_: None, n_sims=400, session=sess,
                  price_model="sim")
    assert errs == [] and store.load_sim(tmp_path, DAY)[1] is not None
    assert (tmp_path / "inputs" / "preseason.csv").exists() and store.load_candidates(tmp_path, DAY)


def test_an_empty_or_filtered_slate_explains_itself(tmp_path):
    pre = preseason()
    client = FakeOptic(pre, {"final": False})
    client.get_fixtures = lambda **k: [{"id": "A", "start_date": "2026-11-25T00:30:00Z", "status": "postponed"},
                                       {"id": "B", "start_date": "garbage", "status": "unplayed"},
                                       {"id": "C", "start_date": "2026-11-25T00:30:00Z", "status": "unplayed", "is_live": True}]
    logs = []
    A.auto(tmp_path, client, now=at(18, 45), log=logs.append, session=FakeSession(pre))
    text = "\n".join(logs)
    assert "3 returned" in text and "0 in slate window" in text and "status=postponed" in text and "unparseable" in text and "live" in text
    assert "due: nothing" in text
    client.get_fixtures = lambda **k: []
    logs.clear()
    A.auto(tmp_path / "fresh", client, now=at(18, 45), log=logs.append, session=FakeSession(pre))   # a fresh root: not throttled
    assert "0 returned" in "\n".join(logs)


def test_the_exact_window_is_applied_locally_whatever_the_api_returns(tmp_path):
    """Live evidence: the API handed back 10 finished games for a time-of-day window. It is asked for a
    wider, date-level range and the slate window is enforced here."""
    pre = preseason()
    client = FakeOptic(pre, {"final": False})
    seen = {}
    def wide(start_date=None, end_date=None, status=None, **k):
        seen.update(start=start_date, end=end_date, status=status)
        return [{"id": "LASTNIGHT", "start_date": "2026-11-24T01:00:00Z", "status": "unplayed"},   # before 10:00Z: last night's slate
                {"id": "TONIGHT", "start_date": "2026-11-25T00:30:00Z", "status": "unplayed"},
                {"id": "AFTERNOON", "start_date": "2026-11-24T22:00:00Z", "status": "unplayed"},
                {"id": "TOMORROW", "start_date": "2026-11-26T00:30:00Z", "status": "unplayed"}]
    client.get_fixtures = wide
    from paper.capture import list_fixtures
    got = list_fixtures(client, DAY)
    assert [f["id"] for f in got] == ["AFTERNOON", "TONIGHT"]            # sorted, window-filtered
    assert seen == {"start": "2026-11-24", "end": "2026-11-26", "status": "unplayed"}


def test_if_the_status_filter_returns_nothing_the_other_routes_are_tried(tmp_path):
    """Live evidence: ?status=unplayed returned 0 while the unfiltered call returned games."""
    pre = preseason()
    client = FakeOptic(pre, {"final": False})
    calls = []
    def get_fixtures(start_date=None, end_date=None, status=None, **k):
        calls.append(("fixtures", status))
        if status is not None:
            return []                                                     # silently nothing for this status
        return [{"id": "G1", "start_date": "2026-11-25T00:30:00Z", "status": "scheduled"},     # a status we did not send
                {"id": "OLD", "start_date": "2026-11-25T00:30:00Z", "status": "completed"}]
    client.get_fixtures = get_fixtures
    client.get_active_fixtures = lambda **k: calls.append(("active", None)) or []
    logs = []
    from paper.capture import list_fixtures
    assert [f["id"] for f in list_fixtures(client, DAY, logs.append)] == ["G1"]
    assert calls == [("fixtures", "unplayed"), ("active", None), ("fixtures", None)]
    text = "\n".join(logs)
    assert "via fixtures?status=unplayed" in text and "via active" in text and "dropped {'status=completed': 1}" in text


def test_the_active_endpoint_is_used_when_it_has_the_games(tmp_path):
    pre = preseason()
    client = FakeOptic(pre, {"final": False})
    client.get_fixtures = lambda **k: []
    client.get_active_fixtures = lambda **k: [{"id": "A1", "start_date": "2026-11-25T00:30:00Z", "status": "unplayed"}]
    from paper.capture import list_fixtures
    assert [f["id"] for f in list_fixtures(client, DAY)] == ["A1"]


def test_when_every_route_is_empty_the_log_says_what_each_returned(tmp_path):
    pre = preseason()
    client = FakeOptic(pre, {"final": False})
    client.get_fixtures = lambda **k: ([{"id": "X", "start_date": "2026-11-23T01:00:00Z", "status": "completed"}] if "status" not in k else [])
    client.get_active_fixtures = lambda **k: (_ for _ in ()).throw(RuntimeError("HTTP 404"))
    logs = []
    from paper.capture import list_fixtures
    assert list_fixtures(client, DAY, logs.append) == []
    text = "\n".join(logs)
    assert "via active: failed (HTTP 404)" in text
    assert "1 returned, statuses {'completed': 1}, starts 2026-11-23T01:00Z..2026-11-23T01:00Z" in text
    assert "outside the slate window" in text


def test_a_status_the_api_rejects_does_not_stop_the_other_routes(tmp_path):
    pre = preseason()
    client = FakeOptic(pre, {"final": False})
    calls = []
    def picky(start_date=None, end_date=None, status=None, **k):
        calls.append(status)
        if status is not None:
            raise RuntimeError("HTTP 400 bad status")
        return [{"id": "A", "start_date": "2026-11-25T00:30:00Z", "status": "unplayed"}]
    client.get_fixtures = picky
    logs = []
    from paper.capture import list_fixtures
    assert [f["id"] for f in list_fixtures(client, DAY, logs.append)] == ["A"]
    assert calls == ["unplayed", None] and any("failed (HTTP 400" in l for l in logs)


def test_an_empty_answer_is_not_asked_again_for_30_minutes(tmp_path):
    pre = preseason()
    client = FakeOptic(pre, {"final": False})
    calls = []
    client.get_fixtures = lambda **k: calls.append(1) or []
    client.get_active_fixtures = lambda **k: []
    logs = []
    A.auto(tmp_path, client, now=at(19, 0), log=logs.append, session=FakeSession(pre))
    n = len(calls)
    assert n >= 2                                                           # several routes on the first, empty, probe
    A.auto(tmp_path, client, now=at(19, 15), log=logs.append, session=FakeSession(pre))
    assert len(calls) == n and any("asking again after 19:30Z" in l for l in logs)     # throttled: no new requests
    A.auto(tmp_path, client, now=at(19, 31), log=logs.append, session=FakeSession(pre))
    assert len(calls) > n                                                   # window passed: asks again
    # as soon as games exist the throttle clears
    client.get_fixtures = lambda **k: [{"id": "A", "start_date": "2026-11-25T00:30:00Z", "status": "unplayed"}]
    A.auto(tmp_path, client, now=at(20, 5), log=logs.append, session=FakeSession(pre))
    assert not (tmp_path / "inputs" / "fixtures_probe.json").exists()


def test_the_probe_with_no_date_filter_is_the_last_resort_and_is_logged(tmp_path):
    pre = preseason()
    client = FakeOptic(pre, {"final": False})
    client.get_fixtures = lambda **k: []
    seen = []
    def active(start_date=None, end_date=None):
        seen.append((start_date, end_date))
        return [] if start_date else [{"id": "NEXT", "start_date": "2026-11-27T00:30:00Z", "status": "unplayed"}]
    client.get_active_fixtures = active
    logs = []
    from paper.capture import list_fixtures
    assert list_fixtures(client, DAY, logs.append) == []                    # not in the slate window ...
    assert seen[-1] == (None, None)                                         # ... but the probe asked for everything
    assert any("active-any-date" in l and "starts 2026-11-27T00:30Z" in l for l in logs)   # ... and the log shows when the next game is


def test_unchanged_outputs_are_not_rewritten(tmp_path):
    import time
    from paper import export
    seed = __import__("test_pipeline").seed_day
    seed(tmp_path, "2026-11-24", closing_line=24.5)
    A.write_report(tmp_path)
    out = export.export_site(tmp_path)
    before = {p: p.read_bytes() for p in list(out.rglob("*.json")) + [tmp_path / "report.md"]}
    time.sleep(1.1)                                                         # generatedAt / "Updated" would differ
    A.write_report(tmp_path)
    export.export_site(tmp_path)
    after = {p: p.read_bytes() for p in before}
    assert before == after                                                  # byte-identical: no commit on the data branch


# ---- the live /fixtures?status=unplayed response (trimmed from the real payload) ----------------------
def real_fx(fid, start, home, away, has_odds=True):
    return {"id": fid, "start_date": start, "status": "unplayed", "is_live": False, "has_odds": has_odds,
            "home_competitors": [{"abbreviation": home}], "away_competitors": [{"abbreviation": away}]}


LIVE_PAGE_1 = [real_fx("20261125398B391D", "2026-11-25T00:00:00Z", "ORL", "DET"),
               real_fx("20261125EF3D60A2", "2026-11-25T01:00:00Z", "WAS", "BOS"),
               real_fx("202611257A3D5A9D", "2026-11-25T03:00:00Z", "DEN", "DAL"),
               real_fx("202611255BC9A1BC", "2026-11-25T22:00:00Z", "PHI", "MIA"),     # tomorrow's slate (after 10:00Z)
               real_fx("2026112637F9D8D2", "2026-11-26T00:00:00Z", "CHA", "CHI")]
LIVE_PAGE_2 = [real_fx("20261128889FADF6", "2026-11-28T23:00:00Z", "CHA", "BKN"),
               real_fx("20261212C78603E8", "2026-12-12T22:00:00Z", "MEM", "MIL", has_odds=False)]


def test_upcoming_is_asked_for_the_way_the_optic_docs_do_with_no_dates_and_stops_early():
    from test_pipeline import FakeResp, FakeSession
    from paper import optic
    from datetime import datetime, timezone
    pages = [FakeResp(200, {"data": LIVE_PAGE_1, "has_more": True, "page": 1}),
             FakeResp(200, {"data": LIVE_PAGE_2, "has_more": True, "page": 2})]
    s = FakeSession(pages)
    c = optic.OpticOddsClient(api_key="k", session=s, rate=(1000, 1), logger=lambda *_: None)
    got = c.get_upcoming_fixtures(until=datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc))
    assert len(got) == 5 and len(s.calls) == 1                       # page 1 already reaches past the window: no page 2
    url = s.calls[0][0]
    assert "status=unplayed" in url and "sport=basketball" in url and "league=nba" in url
    assert "start_date" not in url and "end_date" not in url          # the date params are what broke it


def test_tonights_slate_comes_out_of_the_live_response(tmp_path):
    pre = preseason()
    client = FakeOptic(pre, {"final": False})
    calls = []
    client.get_upcoming_fixtures = lambda until: calls.append("upcoming") or LIVE_PAGE_1 + LIVE_PAGE_2
    client.get_fixtures = lambda **k: calls.append("dated") or []
    logs = []
    from paper.capture import list_fixtures
    got = list_fixtures(client, DAY, logs.append)
    assert [f["id"] for f in got] == ["20261125398B391D", "20261125EF3D60A2", "202611257A3D5A9D"]
    assert calls == ["upcoming"]                                       # the first route worked: nothing else asked
    assert "dropped" in "\n".join(logs) and "outside the slate window" in "\n".join(logs)


def test_a_game_with_no_odds_posted_yet_is_not_captured(tmp_path):
    pre = preseason()
    client = FakeOptic(pre, {"final": False})
    client.get_upcoming_fixtures = lambda until: [real_fx("A", "2026-11-25T00:00:00Z", "ORL", "DET", has_odds=False),
                                                  real_fx("B", "2026-11-25T01:00:00Z", "WAS", "BOS")]
    logs = []
    from paper.capture import list_fixtures
    assert [f["id"] for f in list_fixtures(client, DAY, logs.append)] == ["B"]
    assert "no odds posted yet" in "\n".join(logs)


def test_if_upcoming_has_nothing_the_dated_routes_still_run(tmp_path):
    pre = preseason()
    client = FakeOptic(pre, {"final": False})
    client.get_upcoming_fixtures = lambda until: []
    from paper.capture import list_fixtures
    got = list_fixtures(client, DAY)                                   # FakeOptic.get_fixtures still serves its game
    assert got and got[0]["id"] == FX

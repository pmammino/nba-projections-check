import math

import numpy as np
import pytest

from paper import books, consensus as cs, markets, odds, paper_filter as pf, personas as ps, pricing


# ---- odds / devig ------------------------------------------------------------
def test_american_conversions():
    assert odds.american_to_decimal(100) == 2.0
    assert odds.american_to_decimal(-200) == pytest.approx(1.5)
    assert odds.american_to_decimal(50) is None          # not a valid American price
    assert odds.american_to_prob(-110) == pytest.approx(0.5238, abs=1e-4)


def test_devig_symmetric_methods_agree():
    for m in odds.DEVIG_METHODS:
        d = odds.devig_two_way(-110, -110, m)
        assert d["fair_over"] == pytest.approx(0.5)
        assert d["hold"] == pytest.approx(2 * 0.5238 - 1, abs=1e-3)


def test_devig_power_shrinks_longshot_more():
    mult = odds.devig_two_way(400, -600, "multiplicative")["fair_over"]
    power = odds.devig_two_way(400, -600, "power")["fair_over"]
    assert power < mult


def test_grade_outcome_push():
    assert odds.grade_outcome(25, 25, "over") == "push"
    assert odds.grade_outcome(26, 25.5, "over") == "won"
    assert odds.grade_outcome(26, 25.5, "under") == "lost"


# ---- books -------------------------------------------------------------------
def test_reference_books_never_bettable():
    assert not books.is_bettable("pinnacle")
    assert not books.is_bettable("circa_sports")
    assert books.is_bettable("DraftKings") and books.is_bettable("Hard Rock Bet")
    assert not books.is_bettable("bet99")                # unknown => not bettable


def test_resolve_ambiguous_is_reported_not_guessed():
    live = [{"id": "circa_sports", "name": "Circa Sports"}, {"id": "circa_vegas", "name": "Circa Vegas"},
            {"id": "pinnacle", "name": "Pinnacle"}]
    r = books.resolve_book_ids(["circa", "pinnacle", "nope"], live)
    assert r["ambiguous"] == {"circa": ["circa_sports", "circa_vegas"]}
    assert r["matched"] == {"pinnacle": "pinnacle"} and r["missing"] == ["nope"]


# ---- markets -----------------------------------------------------------------
def test_market_matching_is_exact():
    assert markets.match_stat_key("player_points_+_rebounds_+_assists") == "pra"
    assert markets.match_stat_key("Player Points + Rebounds") == "pr"
    for variant in ("player_points_combo_", "player_points_either_", "1st_half_player_points",
                    "player_points_h2h_spread", "1st_quarter_player_points"):
        assert markets.match_stat_key(variant) is None


def test_stat_actual_combos_and_double_double():
    box = dict(fgm=10, pt3m=2, ftm=4, oreb=2, dreb=8, ast=7, stl=1, blk=0, tov=3, fga=20, fta=5)
    assert markets.stat_actual(box, "pts") == 26
    assert markets.stat_actual(box, "pra") == 26 + 10 + 7
    assert markets.stat_actual(box, "dd") == 1.0 and markets.stat_actual(box, "td") == 0.0


# ---- pricing -----------------------------------------------------------------
def test_win_prob_conditions_on_no_push():
    d = np.array([20, 25, 25, 25, 30] * 200, dtype=float)
    po = pricing.win_prob(d, 25, "over")
    pu = pricing.win_prob(d, 25, "under")
    assert po == pytest.approx(0.5, abs=0.01) and po + pu == pytest.approx(1.0, abs=0.01)


def test_win_prob_never_certain():
    d = np.full(4000, 30.0)
    assert 0 < pricing.win_prob(d, 10.5, "under") < 0.01
    assert pricing.win_prob(d, 10.5, "over") < 1.0


# ---- consensus ---------------------------------------------------------------
def votes_for(prices):
    rows = [{"book": b, "over_odds": o, "under_odds": u} for b, o, u in prices]
    return cs.book_votes(rows)


def test_one_vote_per_book_and_one_sided_flag():
    v = votes_for([("DraftKings", -110, -110), ("draftkings", -150, 130), ("FanDuel", -115, None)])
    assert [x["book"] for x in v] == ["draftkings", "fanduel"]
    assert v[0]["two_sided"] and not v[1]["two_sided"]
    assert v[0]["key"] == "draftkings:2" and v[1]["key"] == "fanduel:1"


def test_under_only_quote_is_mirrored():
    # An Under favourite at -130 means P(over) < 0.5: a NEGATIVE logit, not dropped.
    v = votes_for([("Pinnacle", None, -130)])
    assert len(v) == 1 and v[0]["logit"] < 0 and not v[0]["two_sided"]


def test_pool_is_bounded_by_the_market_and_needs_books():
    v = votes_for([("DraftKings", -110, -110), ("FanDuel", -110, -110), ("BetMGM", -110, -110)])
    assert cs.pool_votes([], projection_prob=0.9, projection_share=0.75) is None
    pooled = cs.pool_votes(v, projection_prob=0.99, projection_share=0.75)
    # a wildly confident projection is drawn in to MAX_DEVIATION of the market
    assert pooled["prob"] < cs.sigmoid(cs.MAX_DEVIATION * 0.75) + 1e-9
    assert cs.pool_votes(v, projection_prob=0.99, projection_share=0.0)["prob"] == pytest.approx(0.5, abs=0.01)


def test_pool_outlier_cannot_drag_further_than_max_deviation():
    v = votes_for([("DraftKings", -110, -110), ("FanDuel", -110, -110), ("BetMGM", -110, -110),
                   ("Caesars", 900, -1500)])
    p = cs.pool_votes(v)["prob"]
    lim = cs.sigmoid(cs.MAX_DEVIATION)
    assert 0.3 < p < lim


def test_lead_regression_recovers_planted_lead():
    rng = np.random.default_rng(0)
    pairs = []
    for i in range(400):
        truth = rng.normal(0, 0.4)
        # "sharp" leads: its early logit already holds the truth; the others lag it
        early_lo = rng.normal(0, 0.05)
        sharp = {"book": "pinnacle", "key": "pinnacle:2", "two_sided": True, "logit": early_lo + truth, "hold": .04, "assumed_hold": .07}
        others = [{"book": b, "key": f"{b}:2", "two_sided": True, "logit": early_lo + rng.normal(0, .05), "hold": .05, "assumed_hold": .07}
                  for b in ("draftkings", "fanduel", "betmgm")]
        late = [{"book": b, "key": f"{b}:2", "two_sided": True, "logit": early_lo + 0.8 * truth + rng.normal(0, .05), "hold": .05, "assumed_hold": .07}
                for b in ("draftkings", "fanduel", "betmgm", "pinnacle")]
        pairs.append({"date": f"d{i // 4}", "slot": "morning", "stat": "pts", "player_id": i, "line": 20.5,
                      "cluster": f"d{i // 4}|{i}", "early": [sharp] + others, "late": late, "projection": None})
    a = cs.assess_sources(pairs)
    assert a["pinnacle:2"]["lead"] > 0.5 > a["draftkings:2"]["lead"]
    w = cs.fit_source_weights(a)["weights"]
    assert w["pinnacle:2"] > w["draftkings:2"]


def test_weights_as_of_never_sees_its_own_date():
    pairs = [{"date": "2026-01-02", "slot": "m", "stat": "pts", "player_id": 1, "line": 1, "cluster": "c",
              "early": [], "late": [], "projection": None}]
    assert cs.weights_as_of(pairs, "2026-01-02")["dates_used"] == []
    assert cs.weights_as_of(pairs, "2026-01-03")["dates_used"] == ["2026-01-02"]


def test_projection_share_zero_without_evidence_and_positive_with():
    rng = np.random.default_rng(1)
    def obs(informative, n=600):
        out = []
        for i in range(n):
            lb = rng.normal(0, 0.3)
            lp = lb + rng.normal(0, 0.4)
            p = cs.sigmoid(lb + (0.6 * (lp - lb) if informative else 0.0))
            out.append({"l_books": lb, "l_proj": lp, "y": float(rng.random() < p), "cluster": f"c{i}"})
        return out
    assert cs.fit_projection_share(obs(False))["share"] < 0.15
    assert cs.fit_projection_share(obs(True, 3000))["share"] > 0.2
    assert cs.fit_projection_share([])["share"] == 0.0


# ---- personas ----------------------------------------------------------------
def edge_row(pid, stat="pts", book="draftkings", p=0.60, odds_=-110, **kw):
    implied = odds.american_to_prob(odds_)
    r = dict(date="d", slot="morning", fixture_id="g1", player_id=pid, name=f"P{pid}", team="AAA", opp="BBB",
             stat=stat, book=book, bettable=True, side="over", line=20.5, odds=odds_, opposite_odds=-110,
             implied=implied, fair=implied - 0.02, hold=0.045, one_sided=False, p_proj=p, p_pool=p - .01,
             p_sharp=p - .02, p_fair=None, n_books=5, n_sharp=1, main_line=20.5, rung=0, max_stake=None)
    r.update(kw)
    return r


def test_reference_book_price_can_never_be_staked():
    rows = ps.with_basis([edge_row(1, book="pinnacle", bettable=False), edge_row(2)])
    out = ps.simulate_day(rows, ps.PERSONA_BY_ID["firehose"])
    assert [b["player_id"] for b in out] == [2]


def test_shopper_beats_single_book_by_best_price():
    rows = ps.with_basis([edge_row(1, book="draftkings", odds_=-130), edge_row(1, book="fanduel", odds_=-105)])
    dk = ps.simulate_day(rows, ps.PERSONA_BY_ID["disciplined"])
    sh = ps.simulate_day(rows, ps.PERSONA_BY_ID["shopper"])
    assert dk[0]["book"] == "draftkings" and sh[0]["book"] == "fanduel"
    assert sh[0]["edge"] > dk[0]["edge"]


def test_one_bet_per_player_across_correlated_stats():
    rows = ps.with_basis([edge_row(1, "pts"), edge_row(1, "pra", p=0.65), edge_row(1, "pa", p=0.62)])
    out = ps.simulate_day(rows, ps.PERSONA_BY_ID["shopper"])
    assert len(out) == 1 and out[0]["stat"] == "pra"


def test_max_bets_and_per_game_limits():
    rows = ps.with_basis([edge_row(i, p=0.70 - i * 0.001, fixture_id=f"g{i % 4}") for i in range(40)])
    out = ps.simulate_day(rows, ps.PERSONA_BY_ID["shopper"])
    assert len(out) == 5
    per_game = {}
    for b in out:
        per_game[b["fixture_id"]] = per_game.get(b["fixture_id"], 0) + 1
    assert max(per_game.values()) <= 2


def test_daily_budget_scales_stakes_down_keeping_selection():
    p = {**ps.PERSONA_BY_ID["budget"], "min_edge": 0.0}
    rows = ps.with_basis([edge_row(i, p=0.60 + i * 0.01) for i in range(5)])
    out = ps.simulate_day(rows, p)
    assert sum(b["stake_units"] for b in out) == pytest.approx(10.0, abs=1e-3)
    assert out[0]["stake_units"] > out[-1]["stake_units"]


def test_kelly_stake_is_fraction_of_compounding_bankroll():
    rows = ps.with_basis([edge_row(1, p=0.60)])
    a = ps.simulate_day(rows, ps.PERSONA_BY_ID["kelly"], 100.0)[0]["stake_units"]
    b = ps.simulate_day(rows, ps.PERSONA_BY_ID["kelly"], 200.0)[0]["stake_units"]
    assert b == pytest.approx(2 * a, rel=1e-3) and a <= 3.0 + 1e-9    # 3% cap of 100


def test_sharp_persona_requires_a_sharp_quote_that_agrees():
    ok = edge_row(1, p_sharp=0.58)
    contradicts = edge_row(2, p_sharp=0.40)
    none = edge_row(3, p_sharp=None)
    rows = ps.with_basis([ok, contradicts, none])
    out = ps.simulate_day(rows, ps.PERSONA_BY_ID["sharp"])
    assert [b["player_id"] for b in out] == [1]


def test_purist_ignores_one_sided_markets():
    rows = ps.with_basis([edge_row(1, one_sided=True), edge_row(2, one_sided=False, p=0.70)])
    assert [b["player_id"] for b in ps.simulate_day(rows, ps.PERSONA_BY_ID["purist"])] == [2]


def test_bankroll_only_moves_for_compounding():
    assert ps.advance_bankroll(ps.PERSONA_BY_ID["shopper"], 100, -5) == 100
    assert ps.advance_bankroll(ps.PERSONA_BY_ID["kelly"], 100, -5) == 95
    assert ps.advance_bankroll(ps.PERSONA_BY_ID["kelly"], 3, -50) == 0


# ---- filter ------------------------------------------------------------------
def frow(pid, slot="morning", book="draftkings", odds_=-110, p=0.60, status="won", **kw):
    r = edge_row(pid, slot=slot, book=book, odds_=odds_, p=p)
    r.update(status=status, actual=30.0, close_fair=None)
    r.update(kw)
    return r


def test_sharp_confirm_drops_unconfirmable_markets():
    rows = [frow(1, p_sharp=0.58), frow(2, p_sharp=0.40), frow(3, p_sharp=None)]
    f = {**pf.DEFAULT_FILTERS, "sharp_confirm": 0.0}
    assert [b["player_id"] for b in pf.apply_filters(rows, f)] == [1]


def test_timing_first_takes_earliest_capture_once():
    rows = [frow(1, slot="pregame", p=0.70), frow(1, slot="morning", p=0.61)]
    bets = pf.apply_filters(rows, {"timing": "first"})
    assert len(bets) == 1 and bets[0]["slot"] == "morning"
    assert len(pf.apply_filters(rows, {"timing": "every"})) == 2


def test_closing_board_is_never_bettable():
    assert pf.apply_filters([frow(1, slot="closing")]) == []


def test_shopping_picks_best_price_and_ladder_best_picks_biggest_edge():
    rows = [frow(1, book="draftkings", odds_=-120), frow(1, book="fanduel", odds_=-105)]
    assert pf.apply_filters(rows, {"shop": True})[0]["book"] == "fanduel"
    ladder = [frow(1, line=20.5, rung=0, p=0.60), frow(1, line=22.5, rung=2, p=0.66)]
    assert pf.apply_filters(ladder, {"ladder": "best"})[0]["line"] == 22.5
    assert len(pf.apply_filters(ladder, {"ladder": "all", "min_books": 1})) == 2


def test_odds_range_in_decimal_space():
    assert pf.odds_in_range(-400, -400, 200) and pf.odds_in_range(200, 200, -400)
    assert not pf.odds_in_range(-500, -400, 200) and not pf.odds_in_range(250, -400, 200)


def test_summarize_void_is_not_a_loss_and_pending_on_complete_day_is_not_a_win():
    bets = pf.apply_filters([frow(1, status="won"), frow(2, status="lost"), frow(3, status="void"),
                             frow(4, status="pending")])
    s = pf.summarize(bets, complete_dates=["d"])
    assert (s["won"], s["lost"], s["void"], s["no_line"]) == (1, 1, 1, 1)
    assert s["settled"] == 2 and s["pnl"] == pytest.approx((100 / 110) - 1)


def test_verdict_needs_volume_and_an_interval_excluding_zero():
    assert pf.verdict({"settled": 10, "roi": 0.5, "roi_band": 0.1}) == "too few bets to call"
    assert pf.verdict({"settled": 500, "roi": 0.05, "roi_band": 0.1}) == "indistinguishable from zero"
    assert pf.verdict({"settled": 500, "roi": 0.15, "roi_band": 0.1}) == "positive"
    assert pf.verdict({"settled": 500, "roi": -0.15, "roi_band": 0.1}) == "negative"

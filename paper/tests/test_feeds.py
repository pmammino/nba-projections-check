"""The RotoWire feeds as the price source. The two rows below are copied from the live
feeds (Santi Aldama, Deandre Ayton): the median feed and the ceiling feed."""
import numpy as np
import pytest

from paper import dist, feeds
from paper.pricing import win_prob

MEAN = [
    {"id": "5428", "player": "Santi Aldama", "team": "DAL", "pos": "F", "G": "1", "DBLDBL": "0.00850", "TRPDBL": "0.00000",
     "THREEPA": "2.07691", "THREEPM": "0.70211", "AST": "0.92831", "BLK": "0.19088", "DREB": "1.80668", "FGA": "3.87455",
     "FGM": "1.73897", "FTA": "0.65489", "FTM": "0.44519", "MIN": "10.00000", "OREB": "0.48052", "PF": "0.80155",
     "PTS": "4.62523", "REB": "2.28720", "STL": "0.29570", "TO": "0.54149", "opp": "DEN"},
    {"id": "4372", "player": "Deandre Ayton", "team": "WAS", "pos": "C", "G": "1", "DBLDBL": "0.00967", "TRPDBL": "0.00000",
     "THREEPA": "0.00018", "THREEPM": "0.00007", "AST": "0.41386", "BLK": "0.18694", "DREB": "1.43848", "FGA": "1.84214",
     "FGM": "1.17039", "FTA": "0.33233", "FTM": "0.18997", "MIN": "9.00000", "OREB": "1.13212", "PF": "0.70761",
     "PTS": "2.53083", "REB": "2.57060", "STL": "0.12311", "TO": "0.27670", "opp": "BOS"},
]
CEIL = [
    {"id": "5428", "player": "Santi Aldama", "team": "DAL", "pos": "F", "G": "1", "DBLDBL": "0.00850", "TRPDBL": "0.00000",
     "THREEPA": "3.00000", "THREEPM": "1.00000", "AST": "1.00000", "BLK": "0.00000", "DREB": "3.00000", "FGA": "6.00000",
     "FGM": "3.00000", "FTA": "1.00000", "FTM": "1.00000", "MIN": "16.54108", "OREB": "1.00000", "PF": "0.00000",
     "PTS": "8.00000", "REB": "4.00000", "STL": "0.00000", "TO": "0.00000", "opp": "DEN"},
    {"id": "4372", "player": "Deandre Ayton", "team": "WAS", "pos": "C", "G": "1", "DBLDBL": "0.00967", "TRPDBL": "0.00000",
     "THREEPA": "0.00000", "THREEPM": "0.00000", "AST": "1.00000", "BLK": "0.00000", "DREB": "3.00000", "FGA": "3.00000",
     "FGM": "2.00000", "FTA": "0.00000", "FTM": "0.00000", "MIN": "16.54417", "OREB": "2.00000", "PF": "0.00000",
     "PTS": "4.00000", "REB": "5.00000", "STL": "0.00000", "TO": "0.00000", "opp": "BOS"},
]


def floor_like(ceil_rows):
    """The floor feed is not in the sample; build one on the other side of each median."""
    out = []
    for r, m in zip(ceil_rows, MEAN):
        f = dict(r)
        for k in ("PTS", "REB", "AST", "FGA", "FGM", "FTA", "FTM", "OREB", "DREB", "STL", "BLK", "THREEPA", "THREEPM"):
            f[k] = "0.00000"
        f["TO"] = "1.00000"          # direction-aware: the low-tail "ceiling" has 0, the "floor" has MORE turnovers
        f["PF"] = "1.00000"
        f["MIN"] = "5.00000"
        out.append(f)
    return out


def snap():
    return feeds.build_snapshot(feeds.parse_feed(MEAN), feeds.parse_feed(floor_like(CEIL)), feeds.parse_feed(CEIL))


def test_real_rows_parse():
    s = feeds.parse_feed(MEAN)
    assert s[5428]["name"] == "Santi Aldama" and (s[5428]["team"], s[5428]["opp"]) == ("DAL", "DEN")
    assert s[5428]["pts"] == pytest.approx(4.62523) and s[5428]["to"] == pytest.approx(0.54149)


def test_quartiles_are_ordered_whichever_way_the_feeds_are_labelled():
    sn = snap()[5428]
    assert (sn["pts_lo"], sn["pts_hi"]) == (0.0, 8.0)
    # turnovers: the "ceiling" feed carries the LOW tail (0) and the "floor" feed the high (1)
    assert (sn["tov_lo"], sn["tov_hi"]) == (0.0, 1.0)
    swapped = feeds.build_snapshot(feeds.parse_feed(MEAN), feeds.parse_feed(CEIL), feeds.parse_feed(floor_like(CEIL)))
    assert (swapped[5428]["pts_lo"], swapped[5428]["pts_hi"], swapped[5428]["tov_lo"]) == (0.0, 8.0, 0.0)


def test_a_player_missing_from_a_feed_is_not_priced():
    s = feeds.build_snapshot(feeds.parse_feed(MEAN), feeds.parse_feed(floor_like(CEIL)[:1]), feeds.parse_feed(CEIL))
    assert 4372 not in s and 5428 in s


# ---- dist -----------------------------------------------------------------------------
def test_fit_reproduces_the_published_quartiles_and_the_mean():
    for mean, lo, hi in [(20.0, 15, 25), (4.6, 1, 7), (0.93, 0, 1), (6.5, 4, 9)]:
        pmf, k, err = dist.fit_count(mean, lo, hi)
        c = np.cumsum(pmf)
        assert err < 0.002, (mean, lo, hi, err)
        assert c[lo - 1 if lo else 0] < 0.25 + 1e-9 or lo == 0
        assert c[lo] >= 0.25 - 1e-9 and c[hi] >= 0.75 - 1e-9 and (hi == 0 or c[hi - 1] < 0.75 + 1e-9)
        assert (pmf * np.arange(len(pmf))).sum() == pytest.approx(mean, rel=0.02)


def test_draws_reproduce_the_pmf_deterministically():
    pmf, _, _ = dist.fit_count(20.0, 15, 25)
    d = dist.quantile_draws(pmf)
    assert d.mean() == pytest.approx(20.0, rel=0.02)
    assert np.array_equal(d, dist.quantile_draws(pmf))               # no RNG
    assert (win_prob(d, 20.5, "over") + win_prob(d, 20.5, "under")) == pytest.approx(1.0, abs=0.01)


def test_an_inconsistent_feed_is_reported_not_silently_priced():
    # a floor of 30 with a mean of 5 cannot come from any negative binomial
    assert dist.fit_count(5.0, 30, 40)[2] > 0


def test_zero_mean_and_probability_markets():
    pmf, _, err = dist.fit_count(0.00007, 0, 0)
    assert pmf[0] > 0.999 and err == 0
    d = dist.bernoulli_draws(0.2)
    assert d.mean() == pytest.approx(0.2, abs=0.001) and win_prob(d, 0.5, "over") == pytest.approx(0.2, abs=0.01)


# ---- FeedBook ---------------------------------------------------------------------------
def test_feedbook_prices_singles_and_refuses_combos():
    b = feeds.FeedBook(snap())
    assert b.priced(5428, "pts") and b.priced(5428, "tov") and b.priced(5428, "dd")
    assert not b.supports("pra") and not b.supports("pr")           # three marginals cannot price a sum
    p_over_half = win_prob(b.draws(5428, "stl"), 0.5, "over")
    assert 0.15 < p_over_half < 0.35                                  # mean 0.30 steals: P(>=1) well under half
    assert win_prob(b.draws(4372, "threes"), 0.5, "over") < 0.01      # 0.00007 expected threes


def test_snapshots_are_per_slot_and_merge_by_team(tmp_path):
    s = snap()
    assert feeds.save_merged(tmp_path, "d", "morning", s, {"DAL"}) == 1
    assert feeds.save_merged(tmp_path, "d", "morning", s, {"WAS"}) == 1       # a later capture keeps DAL
    got = feeds.load_snapshot(tmp_path, "d", "morning")
    assert set(got) == {5428, 4372} and got[5428]["pts_hi"] == 8.0
    assert feeds.load_snapshot(tmp_path, "d", "pregame") is None


def test_calibration_compares_realised_with_what_the_fit_implies(tmp_path):
    from paper import store
    sn = snap()
    feeds.save_merged(tmp_path, "d", "morning", sn, {"DAL", "WAS"})
    box = {"date": "d", "player_id": 5428, "team": "DAL", "minutes": 12, "pts": 20, "reb": 3, "oreb": 1, "dreb": 2,
           "ast": 1, "stl": 0, "blk": 0, "tov": 0, "pf": 1, "fga": 8, "fgm": 8, "fta": 0, "ftm": 4, "pt3m": 0, "pt3a": 0}
    store.save_actuals(tmp_path, "d", [box])
    cal = feeds.calibration(tmp_path, ["d"])
    assert cal["pts"]["above"] == 1.0                                   # 20 points is above the ceiling of 8
    assert 0 <= cal["pts"]["exp_above"] < 0.5 and cal["pts"]["n"] == 1


# The floor feed's first row, copied from the live feed: the high tail for TO/PF is HERE, in the floor.
FLOOR_ALDAMA = {"id": "5428", "player": "Santi Aldama", "team": "DAL", "pos": "F", "G": "1", "DBLDBL": "0.00850",
                "TRPDBL": "0.00000", "THREEPA": "0.00000", "THREEPM": "0.00000", "AST": "0.00000", "BLK": "0.00000",
                "DREB": "0.00000", "FGA": "0.00000", "FGM": "0.00000", "FTA": "0.00000", "FTM": "0.00000",
                "MIN": "2.50592", "OREB": "0.00000", "PF": "1.00000", "PTS": "0.00000", "REB": "0.00000",
                "STL": "0.00000", "TO": "1.00000", "opp": "DEN"}


def test_real_floor_row_confirms_the_direction_convention():
    s = feeds.build_snapshot(feeds.parse_feed(MEAN[:1]), feeds.parse_feed([FLOOR_ALDAMA]), feeds.parse_feed(CEIL[:1]))[5428]
    assert (s["pts_lo"], s["pts_hi"]) == (0.0, 8.0)            # floor 0 / ceiling 8
    assert (s["tov_lo"], s["tov_hi"]) == (0.0, 1.0)            # ceiling feed 0, FLOOR feed 1: the low tail is the "ceiling"
    assert (s["min_lo"], s["min_hi"]) == (pytest.approx(2.50592), pytest.approx(16.54108))
    # and the fits are consistent with the published quartiles
    b = feeds.FeedBook({5428: s})
    assert all(b.fit(5428, st)[2] < 0.01 for st in ("pts", "reb", "ast", "tov", "threes"))


def test_zero_inflation_is_used_only_when_a_plain_nb_cannot_reproduce_the_quartiles():
    # Aldama's real row: mean 4.63 points, quartiles 0 and 8 => a big mass at zero.
    pmf, k, err = dist.fit_count(4.62523, 0, 8)
    assert err < 1e-9 and pmf[0] >= 0.25 and np.cumsum(pmf)[7] < 0.75 + 1e-9 <= np.cumsum(pmf)[8] + 1e-9 + 0.75
    assert (pmf * np.arange(len(pmf))).sum() == pytest.approx(4.62523, rel=0.01)       # the mean survives
    # a symmetric, ordinary case needs no inflation: P(0) is just the NB's own
    plain, _, e2 = dist.fit_count(20.0, 15, 25)
    assert e2 < 0.002 and plain[0] < 1e-3

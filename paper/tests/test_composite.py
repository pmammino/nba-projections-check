"""price_model="both": singles from the feed snapshot, combos from the simulator's draws."""
import numpy as np
import pytest

from paper import capture, feeds, store
from paper.pricebook import CompositeBook, get_book
from test_feeds import snap
from test_pipeline import make_sim, quote_row

D = "2026-11-24"
PID = 5428                      # the feed's id for Santi Aldama; the sim uses the same id space


def seed(root, with_sim=True):
    feeds.save_merged(root, D, "morning", snap(), {"DAL", "WAS"})
    if with_sim:
        store.save_sim(root, D, [PID, 4372], make_sim())
    books = ("draftkings", "fanduel", "betmgm", "caesars")
    rows = []
    for stat, line in (("pts", 24.5), ("pra", 40.5)):
        # soft DraftKings over, so there is a big edge on the OVER if the source thinks it is likely
        rows += [quote_row(b, 130 if b == "draftkings" else -110, -150 if b == "draftkings" else -110, line,
                           pid=PID, name="Santi Aldama", stat=stat) for b in books]
    store.save_board(root, D, "morning", rows)
    return capture.rebuild_candidates(root, D)


def test_singles_come_from_the_feed_and_combos_from_the_sim(tmp_path):
    seed(tmp_path)
    c = {(r["stat"], r["side"], r["book"]): r for r in store.load_candidates(tmp_path, D)}
    book = get_book(tmp_path, D, "morning")
    assert isinstance(book, CompositeBook)
    # points: the FEED says mean 4.6 (quartiles 0/8) so an over 24.5 is hopeless and the UNDER has the edge
    pts_under = c[("pts", "under", "draftkings")]
    assert pts_under["p_proj"] > 0.95
    # PRA: the SIM says 30/6/5 => ~41 expected, so a 40.5 over is a real chance, nothing like the feed's ~7
    pra_over = [r for r in store.load_candidates(tmp_path, D) if r["stat"] == "pra" and r["side"] == "over"]
    assert pra_over and 0.3 < pra_over[0]["p_proj"] < 0.7


def test_without_draws_combos_are_not_priced_but_singles_still_are(tmp_path):
    seed(tmp_path, with_sim=False)
    stats = {r["stat"] for r in store.load_candidates(tmp_path, D)}
    assert "pts" in stats and "pra" not in stats       # never a combo built from three marginals
    assert not feeds.FeedBook(snap()).supports("pra")


def test_a_player_the_feed_dropped_is_not_priced_even_if_the_sim_has_him(tmp_path):
    """Injured players are left out of the feed. The sim still has a column for them, but both
    singles AND combos must follow the feed: it is the later, news-aware word on who plays."""
    s = snap()
    del s[PID]
    feeds.save_merged(tmp_path, D, "morning", s, {"DAL", "WAS"})
    store.save_sim(tmp_path, D, [PID, 4372], make_sim())
    b = get_book(tmp_path, D, "morning")
    assert not b.priced(PID, "pts")                    # single: feed says out
    assert not b.priced(PID, "pra")                    # combo: the sim has a column, but the feed dropped him
    assert b.priced(4372, "pra")                       # a player the feed still lists keeps his combo price


def test_routing_is_per_stat():
    class Fake:
        def __init__(self, stats, tag):
            self.stats, self.tag = stats, tag
        def supports(self, st): return st in self.stats
        def has(self, p): return True
        def priced(self, p, st): return st in self.stats
        def draws(self, p, st): return np.array([self.tag])
    cb = CompositeBook(Fake({"pts"}, 1.0), Fake({"pts", "pra"}, 2.0))
    assert cb.draws(1, "pts")[0] == 1.0 and cb.draws(1, "pra")[0] == 2.0

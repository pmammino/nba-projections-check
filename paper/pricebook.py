"""Which price source a board is priced from.

  feeds   the RotoWire floor/median/ceiling snapshot taken at that slot: singles only
  sim     the stored simulator draws: singles AND the combo markets (PRA etc.)
  both    (default) singles from the feed snapshot, combos from the simulator's draws

Combos need a JOINT distribution (points, rebounds and assists rise and fall together
through minutes); three published marginals cannot give one. Only the simulator's draws
can, so in `both` mode a combo is priced from them and everything else from the feed.
A day with no draws (the simulator failed, or sim was never built) prices singles only and
simply has no combo candidates: it never prices a combo from marginals.

Note the two sources are separate models of the same process, so a combo and its own
legs can disagree slightly; that is a property of mixing sources, not a bug to average
away. The ids must agree too: the feed's `id` and the preseason sheet's `newsid` are
assumed to be the same RotoWire player id (`check-feeds` reports the overlap).
"""

from __future__ import annotations

from . import feeds, store
from .candidates import DrawBook


class CompositeBook:
    def __init__(self, feed_book, sim_book):
        self.feed, self.sim = feed_book, sim_book

    def supports(self, stat) -> bool:
        return self.feed.supports(stat) or self.sim.supports(stat)

    def has(self, player_id) -> bool:
        return self.feed.has(player_id) or self.sim.has(player_id)

    def _source(self, player_id, stat):
        """The book that prices (player, stat), or None. Singles prefer the feed."""
        if self.feed.supports(stat):
            return self.feed if self.feed.priced(player_id, stat) else None   # feed says he is out/unpriced
        # A combo comes from the sim's joint draws, but only for a player the FEED still lists:
        # the feed is the news-aware, later word on who is playing (injured players are dropped
        # from it), and the sim still has a column for a player who has since been ruled out.
        return self.sim if self.feed.has(player_id) and self.sim.priced(player_id, stat) else None

    def priced(self, player_id, stat) -> bool:
        return self._source(player_id, stat) is not None

    def draws(self, player_id, stat):
        src = self._source(player_id, stat)
        if src is None:
            raise KeyError(f"no price source for {stat} / player {player_id}")
        return src.draws(player_id, stat)


def get_book(root, date, slot):
    snap = feeds.load_snapshot(root, date, slot)
    ids, sim = store.load_sim(root, date)
    sim_book = DrawBook(sim, ids) if sim is not None else None
    feed_book = feeds.FeedBook(snap) if snap else None
    if feed_book and sim_book:
        return CompositeBook(feed_book, sim_book)
    return feed_book or sim_book

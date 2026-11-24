"""Which price source a board is priced from. The RotoWire feed snapshot taken at that
slot when there is one (the default; no model to run), else the stored simulator draws
(price_model="sim", still needed for the combo markets)."""

from __future__ import annotations

from . import feeds, store
from .candidates import DrawBook


def get_book(root, date, slot):
    snap = feeds.load_snapshot(root, date, slot)
    if snap:
        return feeds.FeedBook(snap)
    ids, sim = store.load_sim(root, date)
    return DrawBook(sim, ids) if sim is not None else None

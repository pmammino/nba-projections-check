"""The RotoWire daily projection feeds, used as the price source.

  median  .../projections.php?type=today&pos=ALL                 (per-player EXPECTATION)
  ceiling .../projections-ceil-floor.php?type=today&pos=ALL&ceilFloor=C
  floor   .../projections-ceil-floor.php?type=today&pos=ALL&ceilFloor=F

Same process as this repo's game_sim: the "median" feed is the simulation's mean (it is
fractional: 0.93 assists), and floor/ceiling are its 25th/75th percentiles (integers for
counts). That makes them a drop-in for the simulator's draws, minus the joint distribution
(see markets.py: combo markets need the sim).

DIRECTION. game_sim publishes ceiling/floor direction-aware: for turnovers and fouls the
"ceiling" is the LOW tail. The sample confirms it (Aldama: median TO 0.54, ceiling TO 0).
Rather than trust labels, each stat's lower quartile is min(floor, ceiling) and the upper
is max(floor, ceiling): correct either way, and immune to a label flip.

Snapshots are taken at every capture and stored per (date, slot): a pregame bet is priced
off the feed as it stood at pregame (late minutes/injury news included), never off the
morning's. STATUS: written against the sample rows; NOT run against the live site from
the build session. `python -m paper.cli check-feeds` is the first live check.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from . import dist
from .teams import canon

BASE = "https://www.rotowire.com/basketball/tables/"
URLS = {"mean": BASE + "projections.php?type=today&pos=ALL",
        "ceil": BASE + "projections-ceil-floor.php?type=today&pos=ALL&ceilFloor=C",
        "floor": BASE + "projections-ceil-floor.php?type=today&pos=ALL&ceilFloor=F"}
# Same browser-like headers the NFL ingest sends (it works against the same site).
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.rotowire.com/basketball/",
}

# A game has ~25-30 players on the sheet. Fewer than this per team means the feed came back truncated
# (a live run got only the first handful of players alphabetically) and pricing off it would silently
# drop almost every player.
MIN_PLAYERS_PER_TEAM = int(os.environ.get("PAPER_MIN_FEED_PLAYERS_PER_TEAM", "8"))


def request_headers():
    """Browser-like headers, plus the session cookie when ROTOWIRE_COOKIE is set. RotoWire serves only a
    small unauthenticated preview (the NFL repo documents ~top 10) and the full table to a logged-in session."""
    h = dict(HEADERS)
    if os.environ.get("ROTOWIRE_COOKIE"):
        h["Cookie"] = os.environ["ROTOWIRE_COOKIE"]
    return h

# our stat -> feed column (lower-cased). dd/td are probabilities, not counts.
COLUMNS = {"pts": "pts", "reb": "reb", "ast": "ast", "threes": "threepm", "stl": "stl", "blk": "blk",
           "tov": "to", "threesa": "threepa", "fgm": "fgm", "fga": "fga", "ftm": "ftm", "fta": "fta",
           "oreb": "oreb", "dreb": "dreb", "dd": "dbldbl", "td": "trpdbl"}
PROB_STATS = {"dd", "td"}


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def parse_feed(payload):
    """-> {int id: {name, team, opp, min, <column>: float}} with lower-cased columns."""
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("RotoWire feed returned no rows (changed URL, or blocked).")
    out = {}
    for r in rows:
        r = {str(k).lower(): v for k, v in r.items()}
        pid = _f(r.get("id"))
        if pid is None:
            continue
        name = r.get("player") or f"{r.get('firstname', '')} {r.get('lastname', '')}".strip()
        rec = {"name": name, "team": canon(r.get("team")), "opp": canon(r.get("opp")),
               "min": _f(r.get("min"))}
        for col in set(COLUMNS.values()):
            rec[col] = _f(r.get(col))
        out[int(pid)] = rec
    return out


def fetch_raw(session=None):
    """The three feeds, parsed: {mean|floor|ceil: {pid: row}}."""
    if session is None:
        import requests
        session = requests.Session()
    raw = {}
    for key, url in URLS.items():
        res = session.get(url, headers=request_headers(), timeout=60)
        if not res.ok:
            raise RuntimeError(f"RotoWire HTTP {res.status_code} for the {key} feed")
        raw[key] = parse_feed(res.json())
    return raw


def fetch_snapshot(session=None):
    """Pull all three feeds -> {pid: row}, each stat as mean / lo / hi."""
    raw = fetch_raw(session)
    return build_snapshot(raw["mean"], raw["floor"], raw["ceil"])


def check_coverage(snap, teams, minimum=None):
    """Problems (empty list = fine) with a snapshot for the teams about to be priced: a team with
    fewer than `minimum` players means a truncated feed, and the capture must stop rather than save
    a board that prices a handful of players."""
    minimum = MIN_PLAYERS_PER_TEAM if minimum is None else minimum
    per = {}
    for r in snap.values():
        per[r["team"]] = per.get(r["team"], 0) + 1
    return [f"RotoWire feeds list only {per.get(t, 0)} players for {t} (expected at least {minimum}); "
            f"the feed is truncated or needs subscriber access (set ROTOWIRE_COOKIE)"
            for t in sorted(teams) if per.get(t, 0) < minimum]


def build_snapshot(mean, floor, ceil):
    snap = {}
    for pid, m in mean.items():
        f, c = floor.get(pid), ceil.get(pid)
        if f is None or c is None:
            continue                                     # a player missing a feed cannot be priced
        rec = {"name": m["name"], "team": m["team"], "opp": m["opp"], "min": m["min"], "min_hi": None}
        mn_f, mn_c = f.get("min"), c.get("min")
        rec["min_lo"], rec["min_hi"] = (min(mn_f, mn_c), max(mn_f, mn_c)) if None not in (mn_f, mn_c) else (None, None)
        for stat, col in COLUMNS.items():
            a, b = f.get(col), c.get(col)
            rec[f"{stat}_mean"] = m.get(col)
            if stat in PROB_STATS:
                rec[f"{stat}_lo"] = rec[f"{stat}_hi"] = m.get(col)
            else:
                rec[f"{stat}_lo"], rec[f"{stat}_hi"] = (min(a, b), max(a, b)) if None not in (a, b) else (None, None)
        snap[pid] = rec
    return snap


SNAP_FIELDS = (["player_id", "name", "team", "opp", "min", "min_lo", "min_hi"]
               + [f"{s}_{k}" for s in COLUMNS for k in ("mean", "lo", "hi")])


def snap_path(root, date, slot) -> Path:
    return Path(root) / "feeds" / date / f"{slot}.csv"


def save_merged(root, date, slot, snap, teams):
    """Store `snap` rows for `teams` into the slot's snapshot, replacing those teams and
    keeping the rest (slots are per fixture, so a later capture must not overwrite an
    earlier game's feed)."""
    path = snap_path(root, date, slot)
    keep = {pid: r for pid, r in (load_snapshot(root, date, slot) or {}).items() if r["team"] not in teams}
    new = {pid: r for pid, r in snap.items() if r["team"] in teams}
    merged = {**keep, **new}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SNAP_FIELDS)
        w.writeheader()
        for pid, r in sorted(merged.items()):
            w.writerow({"player_id": pid, **{k: ("" if r.get(k) is None else r.get(k)) for k in SNAP_FIELDS[1:]}})
    os.replace(tmp, path)
    return len(new)


def load_snapshot(root, date, slot):
    path = snap_path(root, date, slot)
    if not path.exists():
        return None
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rec = {"name": r["name"], "team": r["team"], "opp": r["opp"]}
            for k in SNAP_FIELDS[4:]:
                rec[k] = _f(r[k])
            out[int(r["player_id"])] = rec
    return out


def players_of(snap):
    """For the Optic-name crosswalk: [{player_id, name, team}]."""
    return [{"player_id": pid, "name": r["name"], "team": r["team"]} for pid, r in snap.items()]


class FeedBook:
    """Same interface as candidates.DrawBook, but the draws are the quantile function of
    a distribution fitted to (mean, floor, ceiling). `supports()` is False for the combo
    markets: three marginals cannot price a sum."""

    def __init__(self, snap, n=dist.N_DRAWS):
        self.snap, self.n, self._cache, self._fit = snap, n, {}, {}
        self.col = {pid: i for i, pid in enumerate(snap)}

    def supports(self, stat) -> bool:
        return stat in COLUMNS

    def has(self, player_id) -> bool:
        try:
            return int(player_id) in self.snap
        except (TypeError, ValueError):
            return False

    def fit(self, player_id, stat):
        """-> (pmf, k, error) or None when the feed lacks this stat for the player."""
        key = (int(player_id), stat)
        if key not in self._fit:
            r = self.snap[int(player_id)]
            m, lo, hi = r.get(f"{stat}_mean"), r.get(f"{stat}_lo"), r.get(f"{stat}_hi")
            if m is None or (stat not in PROB_STATS and (lo is None or hi is None)):
                self._fit[key] = None
            elif stat in PROB_STATS:
                self._fit[key] = (np.array([1 - m, m]), None, 0.0)
            else:
                self._fit[key] = dist.fit_count(m, lo, hi)
        return self._fit[key]

    def draws(self, player_id, stat):
        key = (int(player_id), stat)
        if key not in self._cache:
            f = self.fit(player_id, stat)
            if f is None:
                raise KeyError(f"feed has no {stat} for player {player_id}")
            pmf = f[0]
            self._cache[key] = (dist.bernoulli_draws(pmf[1], self.n) if stat in PROB_STATS
                                else dist.quantile_draws(pmf, self.n))
        return self._cache[key]

    def priced(self, player_id, stat) -> bool:
        return self.has(player_id) and self.fit(player_id, stat) is not None


def calibration(root, dates, slot="morning"):
    """Does the feed's floor/ceiling mean what it says? Per stat: how often the actual
    fell below the floor / above the ceiling, against what the FITTED distribution says
    should happen, and the share inside the band. For integer stats the band holds >= 50%,
    so compare realised with expected, not with 0.50. This is the first thing to read
    before trusting a price: an over-tight band prices every long-shot over as value."""
    from . import store
    from .markets import STAT_DEFS, stat_actual
    acc = {}
    for d in dates:
        snap, act = load_snapshot(root, d, slot), store.load_actuals(root, d)
        if not snap or not act:
            continue
        book = FeedBook(snap)
        by = {a["player_id"]: a for a in act}
        for pid in snap:
            box = by.get(pid)
            if not box or not (box.get("minutes") or 0) > 0:
                continue
            for stat in STAT_DEFS:
                if stat in PROB_STATS or not book.supports(stat) or not book.priced(pid, stat):
                    continue
                x = stat_actual(box, stat)
                pmf = book.fit(pid, stat)[0]
                r = snap[pid]
                lo, hi = r[f"{stat}_lo"], r[f"{stat}_hi"]
                if x is None or lo is None:
                    continue
                c = np.cumsum(pmf)
                a = acc.setdefault(stat, {"n": 0, "below": 0, "above": 0, "inside": 0,
                                          "e_below": 0.0, "e_above": 0.0, "e_inside": 0.0})
                a["n"] += 1
                a["below"] += x < lo
                a["above"] += x > hi
                a["inside"] += lo <= x <= hi
                a["e_below"] += float(c[int(lo) - 1]) if lo >= 1 else 0.0
                a["e_above"] += 1 - float(c[min(int(hi), len(c) - 1)])
                a["e_inside"] += dist.interval_prob(pmf, lo, hi)
    out = {}
    for stat, a in acc.items():
        n = a["n"]
        out[stat] = {"n": n, "below": a["below"] / n, "exp_below": a["e_below"] / n,
                     "above": a["above"] / n, "exp_above": a["e_above"] / n,
                     "inside": a["inside"] / n, "exp_inside": a["e_inside"] / n}
    return out

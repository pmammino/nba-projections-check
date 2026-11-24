"""Every NBA stat we price: its OpticOdds market id, and how to build it from
the simulator's per-stat draws and from a box-score row.

`optic_id` values are EXACT market ids from the live /markets list. Matching is
exact on purpose (as in the NFL repo): the list also carries "(Combo)",
"(Either)", "(Each)", "H2H", per-quarter and per-half variants of every prop,
and substring matching would price full-game projections against them.

Deliberately NOT modelled (the simulator is full-game only):
  * 1st/2nd half, quarter, "1st 3/5 minutes" player markets
  * (Combo)/(Either)/(Each)/H2H markets, which need joint distributions across
    players (the draws could supply them; left out until asked for)
  * team totals / game totals (those are inputs to the model, not outputs)
  * first-basket / most-X-player markets
"""

from __future__ import annotations

import numpy as np

# components: simulator keys summed to form the stat.
# has_line False => yes/no market treated as an implicit over 0.5.
STAT_DEFS = {
    "pts":     {"optic_id": "player_points",                       "parts": ["pts"]},
    "reb":     {"optic_id": "player_rebounds",                     "parts": ["reb"]},
    "ast":     {"optic_id": "player_assists",                      "parts": ["ast"]},
    "threes":  {"optic_id": "player_made_threes",                  "parts": ["pt3m"]},
    "stl":     {"optic_id": "player_steals",                       "parts": ["stl"]},
    "blk":     {"optic_id": "player_blocks",                       "parts": ["blk"]},
    "tov":     {"optic_id": "player_turnovers",                    "parts": ["tov"]},
    "pr":      {"optic_id": "player_points_+_rebounds",            "parts": ["pts", "reb"]},
    "pa":      {"optic_id": "player_points_+_assists",             "parts": ["pts", "ast"]},
    "ra":      {"optic_id": "player_rebounds_+_assists",           "parts": ["reb", "ast"]},
    "pra":     {"optic_id": "player_points_+_rebounds_+_assists",  "parts": ["pts", "reb", "ast"]},
    "stlblk":  {"optic_id": "player_steals_+_blocks",              "parts": ["stl", "blk"]},
    "threesa": {"optic_id": "player_threes_attempted",           "parts": ["pt3a"]},
    "fgm":     {"optic_id": "player_field_goals_made",             "parts": ["fgm"]},
    "fga":     {"optic_id": "player_field_goals_attempted",        "parts": ["fga"]},
    "ftm":     {"optic_id": "player_free_throws_made",             "parts": ["ftm"]},
    "fta":     {"optic_id": "player_free_throws_attempted",        "parts": ["fta"]},
    "oreb":    {"optic_id": "player_offensive_rebounds",           "parts": ["oreb"]},
    "dreb":    {"optic_id": "player_defensive_rebounds",           "parts": ["dreb"]},
    "dd":      {"optic_id": "player_double_double", "parts": ["dd"], "has_line": False},
    "td":      {"optic_id": "player_triple_double", "parts": ["td"], "has_line": False},
}

# Markets the capture requests: the single-stat props, which both price sources can price.
# The combos (pr, pa, ra, pra, stlblk) need a JOINT distribution, which only the simulator
# has (three published marginals cannot price a sum), so they are opt-in via `--stats` and
# only priced when the day has simulator draws (price_model="sim").
DEFAULT_STATS = ["pts", "reb", "ast", "threes", "stl", "blk", "tov", "threesa"]
COMBO_STATS = ["pr", "pa", "ra", "pra", "stlblk"]

_BY_OPTIC = {d["optic_id"]: k for k, d in STAT_DEFS.items()}


def match_stat_key(market) -> str | None:
    """Resolve an OpticOdds market id or name to a stat key. Exact only."""
    if market is None:
        return None
    s = str(market).strip().lower()
    if s in _BY_OPTIC:
        return _BY_OPTIC[s]
    # name -> id form: "Player Points + Rebounds" -> player_points_+_rebounds
    return _BY_OPTIC.get(s.replace(" ", "_"))


def optic_ids(stats=None):
    return [STAT_DEFS[s]["optic_id"] for s in (stats or DEFAULT_STATS)]


def has_line(stat: str) -> bool:
    return STAT_DEFS[stat].get("has_line", True)


def stat_draws(sim: dict, stat: str) -> np.ndarray:
    """(n_sims, n_players) draws of `stat` from the simulator's raw dict."""
    parts = STAT_DEFS[stat]["parts"]
    return sum(np.asarray(sim[p], dtype="float64") for p in parts)


def stat_actual(box: dict, stat: str):
    """The graded value of `stat` from a box-score row (dict-like), or None."""
    try:
        g = lambda k: float(box[k])  # noqa: E731
        pts = 2 * g("fgm") + g("pt3m") + g("ftm")
        reb = g("oreb") + g("dreb")
        vals = {"pts": pts, "reb": reb, "ast": g("ast"), "pt3m": g("pt3m"),
                "stl": g("stl"), "blk": g("blk"), "tov": g("tov"),
                "fgm": g("fgm"), "fga": g("fga"), "ftm": g("ftm"), "fta": g("fta"),
                "pt3a": g("pt3a") if "pt3a" in box and box["pt3a"] not in (None, "") else 0.0,
                "oreb": g("oreb"), "dreb": g("dreb")}
    except (KeyError, TypeError, ValueError):
        return None
    cats = [pts, reb, vals["ast"], vals["stl"], vals["blk"]]
    vals["dd"] = float(sum(c >= 10 for c in cats) >= 2)
    vals["td"] = float(sum(c >= 10 for c in cats) >= 3)
    return sum(vals[p] for p in STAT_DEFS[stat]["parts"])

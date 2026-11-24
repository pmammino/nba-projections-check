"""Box scores for settlement. Source: nba_api game logs (ingest.fetch_player_logs)
-- or any frame with the same columns, e.g. built from OpticOdds results.

A player who does not appear in the box score for a FINAL team-game did not play,
and sportsbooks void a player prop on a DNP. That is `void` (stake returned), not
a loss: the NFL repo leaves it pending because its feed omits zero-stat rows; the
NBA box score lists everyone who logged minutes, so absence is meaningful.
Minutes == 0 is treated the same way.
"""

from __future__ import annotations

from .store import ACTUAL_COLS


def actuals_from_logs(logs, date, id_col="player_id"):
    """Rows for one slate date from a logs DataFrame (canonical raw columns +
    game_date + team). `id_col` must already hold OUR player_id."""
    import pandas as pd
    d = pd.Timestamp(date)
    day = logs[pd.to_datetime(logs["game_date"]).dt.normalize() == d]
    rows = []
    for r in day.to_dict("records"):
        rows.append({"date": str(date), "player_id": int(r[id_col]), "team": r.get("team", ""),
                     "minutes": r.get("minutes"),
                     "pts": 2 * r["fgm"] + r["pt3m"] + r["ftm"], "reb": r["oreb"] + r["dreb"],
                     **{k: r[k] for k in ("oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
                                          "fga", "fgm", "fta", "ftm", "pt3m")}})
    return [{k: r.get(k) for k in ACTUAL_COLS} for r in rows]


def actuals_from_rows(matched, date):
    """Rows for settlement from results.attach_player_ids output (already carries
    our player_id and canonical team)."""
    out = []
    for r in matched:
        out.append({"date": str(date), "player_id": int(r["player_id"]), "team": r["team"],
                    "minutes": r["minutes"], "pts": 2 * r["fgm"] + r["pt3m"] + r["ftm"],
                    "reb": r["oreb"] + r["dreb"],
                    **{k: r[k] for k in ("oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
                                         "fga", "fgm", "fta", "ftm", "pt3m")}})
    return out

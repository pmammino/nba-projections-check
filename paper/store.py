"""On-disk layout. Everything under one root (default <repo>/data/paper):

  boards/{date}/{slot}.csv     captured quotes: the DURABLE input. Never regenerate.
  sims/{date}.npz              simulator draws the day was priced from. Durable too:
                               re-running the model later would price with a
                               different state and break point-in-time honesty.
  candidates/{date}.csv        derived from the two above + the pricing model
  actuals/{date}.csv           box scores for settlement
  ledgers/{persona}/{date}.csv derived: what each persona took and how it settled

Derived files are rebuilt from scratch on every replay (see replay.py), so a
persona-rule change is a re-run, never a migration.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path(os.environ.get("PAPER_DATA_DIR") or Path(__file__).resolve().parent.parent / "data" / "paper")

# Only what the default markets need: sims are committed with the data, so size
# matters (~1 MB/day compressed). Add keys here (e.g. fga, fgm) to price more markets.
SIM_KEYS = ["pts", "reb", "ast", "pt3m", "stl", "blk", "tov"]


def _path(root, *parts) -> Path:
    return Path(root or DEFAULT_ROOT).joinpath(*[str(p) for p in parts])


def write_rows(path: Path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    columns = columns or (list(rows[0].keys()) if rows else [])
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})
    os.replace(tmp, path)   # a crashed run must never leave a half-written capture


def read_rows(path: Path):
    if not Path(path).exists():
        return None
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def num(v):
    if v is None or v == "":
        return None
    try:
        n = float(v)
    except ValueError:
        return None
    return n if n == n else None


# ---- boards -----------------------------------------------------------------
BOARD_COLS = ["fixture_id", "sportsbook", "stat", "optic_player_id", "player_id", "player_name",
              "team", "opp", "home", "away", "line", "over_odds", "under_odds", "one_sided",
              "max_stake", "timestamp", "start_date"]


def board_path(root, date, slot):
    return _path(root, "boards", date, f"{slot}.csv")


def save_board(root, date, slot, rows):
    write_rows(board_path(root, date, slot), rows, BOARD_COLS)


def load_board(root, date, slot):
    raw = read_rows(board_path(root, date, slot))
    if raw is None:
        return None
    out = []
    for r in raw:
        out.append({**r, "line": num(r["line"]), "over_odds": num(r["over_odds"]),
                    "under_odds": num(r["under_odds"]), "max_stake": num(r["max_stake"]),
                    "one_sided": r["one_sided"] in ("1", "True", "true"),
                    "player_id": r["player_id"] or None})
    return out


def board_slots(root, date):
    d = _path(root, "boards", date)
    return sorted(p.stem for p in d.glob("*.csv")) if d.exists() else []


def all_dates(root):
    d = _path(root, "boards")
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


# ---- sims --------------------------------------------------------------------
def save_sim(root, date, player_ids, sim, meta=None):
    p = _path(root, "sims", f"{date}.npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    arrs = {k: np.asarray(sim[k]).astype("uint8") if k not in ("minutes",) else np.asarray(sim[k], dtype="float32")
            for k in SIM_KEYS if k in sim}
    np.savez_compressed(p, player_ids=np.asarray(player_ids, dtype="int64"), **arrs)


def load_sim(root, date):
    p = _path(root, "sims", f"{date}.npz")
    if not p.exists():
        return None, None
    z = np.load(p)
    return z["player_ids"], {k: z[k].astype("float64") for k in z.files if k != "player_ids"}


# ---- candidates / actuals / ledgers --------------------------------------------
CAND_COLS = ["date", "slot", "fixture_id", "player_id", "name", "team", "opp", "stat", "book",
             "bettable", "side", "line", "odds", "opposite_odds", "implied", "fair", "hold",
             "one_sided", "p_proj", "p_pool", "p_sharp", "p_fair", "n_books", "n_sharp",
             "main_line", "rung", "max_stake", "captured_at", "game_start"]
_NUM = {"line", "odds", "opposite_odds", "implied", "fair", "hold", "p_proj", "p_pool", "p_sharp",
        "p_fair", "max_stake", "main_line"}
_INT = {"n_books", "n_sharp", "rung", "player_id"}
_BOOL = {"bettable", "one_sided"}


def save_candidates(root, date, rows):
    write_rows(_path(root, "candidates", f"{date}.csv"), [
        {**r, "bettable": int(bool(r["bettable"])), "one_sided": int(bool(r["one_sided"]))} for r in rows],
        CAND_COLS)


def load_candidates(root, date):
    raw = read_rows(_path(root, "candidates", f"{date}.csv"))
    if raw is None:
        return None
    out = []
    for r in raw:
        o = dict(r)
        for k in _NUM:
            o[k] = num(r.get(k))
        for k in _INT:
            v = num(r.get(k))
            o[k] = None if v is None else int(v)
        for k in _BOOL:
            o[k] = r.get(k) in ("1", "True", "true")
        out.append(o)
    return out


ACTUAL_COLS = ["date", "player_id", "team", "minutes", "pts", "reb", "oreb", "dreb", "ast", "stl",
               "blk", "tov", "pf", "fga", "fgm", "fta", "ftm", "pt3m"]


def save_actuals(root, date, rows):
    write_rows(_path(root, "actuals", f"{date}.csv"), rows, ACTUAL_COLS)


def load_actuals(root, date):
    raw = read_rows(_path(root, "actuals", f"{date}.csv"))
    if raw is None:
        return None
    return [{**r, **{k: num(r[k]) for k in ACTUAL_COLS if k not in ("date", "team", "player_id")},
             "player_id": int(float(r["player_id"]))} for r in raw]


def ledger_path(root, persona, date):
    return _path(root, "ledgers", persona, f"{date}.csv")

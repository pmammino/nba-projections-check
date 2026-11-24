"""Static JSON for the dashboard (web/). Written to <root>/site/ on the paper-data branch,
so a public dashboard reads it straight from raw.githubusercontent.com with no server.

  summary.json  small: personas (totals, equity curve, latest bets), the preset results,
                today's best bets, feed calibration, defaults. Read on the server (ISR).
  pool/index.json        the dates available, row counts, which are graded
  pool/{date}.json       that day's bettable candidates, packed (integers against string tables,
                         probabilities x10000, -1 = missing) so the browser can run the filter
                         engine locally. One file per day: a season in one file would pass GitHub's
                         100 MB limit, and a finished day's file is byte-identical on every later
                         run (no generatedAt inside, stable order), so history does not grow.
                         Read by the browser directly (they exceed Next's 2 MB data-cache limit).
                         Closing-board rows are left out: they are reference-only, never bet, and
                         their CLV is already attached to the rows that were.

Derived, like everything else: rebuilt from boards/candidates/actuals on each run.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from . import feeds, store
from .paper_filter import DEFAULT_FILTERS, PRESETS, apply_filters, bets_to_resolve, preset_filters, summarize, verdict
from .replay import replay, settled_pool

COLUMNS = ["date", "slot", "player", "opp", "stat", "book", "side", "line", "odds", "implied", "pProj",
           "pPool", "pSharp", "nBooks", "nSharp", "mainLine", "rung", "status", "actual", "closeFair",
           "bettable"]
STATUSES = ["pending", "won", "lost", "push", "void"]


def clean(x):
    """JSON-safe: NaN/inf -> None, recursively."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def pack(rows):
    tables = {"dates": [], "slots": [], "opps": [], "stats": [], "books": []}
    players, pidx = [], {}

    def idx(key, v):
        t = tables[key]
        if v not in t:
            t.append(v)
        return t.index(v)

    prob = lambda p: -1 if p is None else int(round(p * 10000))  # noqa: E731
    out = []
    for r in rows:
        if r["player_id"] not in pidx:
            pidx[r["player_id"]] = len(players)
            players.append({"id": r["player_id"], "name": r["name"], "team": r["team"]})
        out.append([idx("dates", r["date"]), idx("slots", r["slot"]), pidx[r["player_id"]],
                    idx("opps", r.get("opp") or ""), idx("stats", r["stat"]), idx("books", r["book"]),
                    1 if r["side"] == "under" else 0, r["line"], r["odds"], prob(r["implied"]),
                    prob(r.get("p_proj")), prob(r.get("p_pool")), prob(r.get("p_sharp")), r["n_books"],
                    r["n_sharp"], r["main_line"], r["rung"], STATUSES.index(r["status"]) if r["status"] in STATUSES else 0,
                    -1 if r.get("actual") is None else r["actual"], prob(r.get("close_fair")),
                    1 if r["bettable"] else 0])
    return {"columns": COLUMNS, "tables": {**tables, "players": players}, "rows": out}


def _bet_row(b):
    keep = ("date", "slot", "name", "team", "opp", "stat", "book", "line", "side", "odds", "implied",
            "p_proj", "p_pool", "p_sharp", "n_books", "status", "actual")
    return {k: b.get(k) for k in keep} | {"edge": b.get("edge"), "pnl": b.get("pnl"), "clv": b.get("clv")}


def build_summary(root, results=None, pool=None):
    dates = store.all_dates(root)
    results = results or replay(root, write=False)
    pool, complete = pool or settled_pool(root)

    personas = []
    for pid, r in results.items():
        by_date = {}
        for row in r["rows"]:
            by_date[row["date"]] = by_date.get(row["date"], 0.0) + (row["pnl_units"] or 0.0)
        cum, equity = 0.0, []
        for d in sorted(by_date):
            cum += by_date[d]
            equity.append({"date": d, "pnl": round(cum, 4), "bankroll": round(100 + cum, 4)})
        p = r["persona"]
        recent = sorted(r["rows"], key=lambda x: (x["date"], -(x.get("edge") or 0)), reverse=True)[:30]
        personas.append({
            "id": pid, "label": p["label"], "description": p["description"],
            "compounding": p["bankroll"]["kind"] == "compounding", **r["totals"], "roi": r["roi"],
            "bankroll": r["bankroll"], "equity": equity,
            "recent": [{k: row.get(k) for k in ("date", "slot", "name", "team", "stat", "book", "line", "side", "odds",
                                                "edge", "stake_units", "status", "actual", "pnl_units", "clv")}
                       for row in recent]})

    presets = []
    for pr in PRESETS:
        f = preset_filters(pr["id"])
        s = summarize(apply_filters(pool, f), complete)
        presets.append({"id": pr["id"], "label": pr["label"], "description": pr["description"], "filters": f,
                        "summary": s, "verdict": verdict(s), "betsToResolve": bets_to_resolve(s["roi"], s["per_bet_sd"])})

    latest = max((r["date"] for r in pool), default=None)
    today = []
    if latest:
        day = [r for r in pool if r["date"] == latest]
        today = [_bet_row(b) for b in apply_filters(day, preset_filters("realistic"))[:40]]

    return clean({
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dates": dates, "completeDates": complete, "candidates": len(pool),
        "personas": personas, "presets": presets, "latestDate": latest, "today": today,
        "calibration": feeds.calibration(root, dates), "defaults": DEFAULT_FILTERS,
    })


def _write(path: Path, obj):
    """Write strict JSON, but leave the file alone when only `generatedAt` differs: an unchanged
    export must not become a commit on the data branch every 15 minutes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, separators=(",", ":"), allow_nan=False)
    if path.exists():
        try:
            old = json.loads(path.read_text())
            strip = lambda o: {k: v for k, v in o.items() if k != "generatedAt"} if isinstance(o, dict) else o  # noqa: E731
            if strip(old) == strip(json.loads(text)):
                return
        except ValueError:
            pass
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def export_site(root, out_dir=None):
    root = Path(root)
    out = Path(out_dir or root / "site")
    pool, complete = settled_pool(root)
    results = replay(root, write=False)
    summary = build_summary(root, results, (pool, complete))

    by_date = {}
    for r in pool:
        if r["slot"] != "closing":
            by_date.setdefault(r["date"], []).append(r)
    shards = []
    for date in sorted(by_date):
        _write(out / "pool" / f"{date}.json", clean(pack(by_date[date])))
        shards.append({"date": date, "rows": len(by_date[date]), "graded": date in complete})
    # a day that no longer has candidates must not leave a stale shard behind
    for f in (out / "pool").glob("*.json"):
        if f.stem != "index" and f.stem not in by_date:
            f.unlink()
    _write(out / "pool" / "index.json", {"generatedAt": summary["generatedAt"], "completeDates": complete, "shards": shards})
    _write(out / "summary.json", summary)
    return out

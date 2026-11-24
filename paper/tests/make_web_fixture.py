"""Regenerates web/lib/__fixtures__/parity.json: a synthetic pool plus what the PYTHON filter engine
returns for a set of filters. web/lib/filter.test.ts runs the TypeScript port on the same pool and fails
if the two ever disagree.  Run from the repo root:  python paper/tests/make_web_fixture.py"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper import export, paper_filter as pf  # noqa: E402
from paper.odds import american_to_prob  # noqa: E402

rng = random.Random(11)
BOOKS = [("draftkings", True), ("fanduel", True), ("betmgm", True), ("caesars", True), ("pinnacle", False), ("circasports", False)]
PRICES = [-190, -150, -125, -115, -110, -105, 100, 110, 125, 150, 220, 380, -450, 500]


def make_rows():
    rows = []
    for date in ("2026-11-20", "2026-11-21", "2026-11-22"):
        for pid in range(1, 26):
            for stat in ("pts", "reb", "ast", "pra"):
                main = rng.choice([8.5, 12.5, 20.5, 24.5])
                for rung, line in ((-1, main - 2), (0, main), (1, main + 2)):
                    n_books = rng.choice([1, 2, 3, 4, 5, 6])
                    sharp = rng.choice([None, 0.4 + rng.random() * 0.2])
                    for slot in ("morning", "pregame", "closing"):
                        if rng.random() < 0.25:
                            continue
                        for book, bettable in BOOKS:
                            if rng.random() < 0.4:
                                continue
                            for side in ("over", "under"):
                                odds = rng.choice(PRICES)
                                implied = round(american_to_prob(odds), 4)
                                noise = lambda: round(min(0.99, max(0.01, implied + rng.uniform(-0.04, 0.12))), 4)  # noqa: E731
                                status = rng.choice(["won", "lost", "lost", "push", "void", "pending"]) if slot != "closing" else "pending"
                                rows.append({
                                    "date": date, "slot": slot, "player_id": pid, "name": f"Player {pid}", "team": "AAA",
                                    "opp": "BBB", "stat": stat, "book": book, "bettable": bettable, "side": side,
                                    "line": line, "odds": odds, "implied": implied, "p_proj": noise(),
                                    "p_pool": rng.choice([None, noise()]), "p_sharp": None if sharp is None else round(min(0.99, sharp + rng.uniform(-.05, .08)), 4),
                                    "n_books": n_books, "n_sharp": 0 if sharp is None else 1, "main_line": main, "rung": rung,
                                    "status": status, "actual": None if status in ("pending", "void") else float(rng.randint(0, 40)),
                                    "close_fair": rng.choice([None, round(rng.uniform(0.3, 0.7), 4)]),
                                })
    return rows


CASES = {
    "default": {}, "realistic": pf.preset_filters("realistic"), "sharp": pf.preset_filters("sharp"),
    "ladder": pf.preset_filters("ladder"), "everything": pf.preset_filters("everything"),
    "pool_basis": {"basis": "pool", "min_edge": 0.02}, "sharp_basis_confirm": {"basis": "sharp", "sharp_confirm": 0.01},
    "unders_only": {"sides": "under", "min_books": 1}, "main_line_only": {"max_rung": 0, "min_books": 1},
    "no_shop": {"shop": False, "ladder": "all", "min_books": 1}, "every_capture": {"timing": "every", "min_books": 1},
    "wide_odds": {"odds_min": -1000, "odds_max": 3000, "min_books": 1}, "books_subset": {"books": ["fanduel", "betmgm"], "min_books": 1},
    "stats_subset": {"stats": ["pts", "pra"]}, "edge_window": {"min_edge": 0.06, "max_edge": 0.15, "min_books": 1},
    "one_date": {"dates": ["2026-11-21"]}, "include_reference_books": {"bettable_only": False, "min_books": 1},
}


def main():
    rows = make_rows()
    complete = ["2026-11-20", "2026-11-21"]
    cases = {}
    for name, f in CASES.items():
        bets = pf.apply_filters(rows, f)
        s = pf.summarize(bets, complete)
        cases[name] = {
            "filters": f,
            "bets": [[b["date"], b["slot"], b["player_id"], b["stat"], b["book"], b["line"], b["side"], b["odds"]] for b in bets],
            "summary": export.clean({k: s[k] for k in ("bets", "settled", "won", "lost", "push", "void", "unplayed", "no_line",
                                                       "pnl", "roi", "roi_band", "per_bet_sd", "win_rate", "avg_edge", "clv")}),
            "verdict": pf.verdict(s),
        }
    out = ROOT / "web" / "lib" / "__fixtures__" / "parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"pool": export.clean(export.pack(rows)), "completeDates": complete, "cases": cases},
                              separators=(",", ":"), allow_nan=False))
    print(f"{len(rows)} rows, {len(cases)} cases -> {out} ({out.stat().st_size // 1024} KB)")
    print({n: len(c["bets"]) for n, c in cases.items()})


if __name__ == "__main__":
    main()

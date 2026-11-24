"""Command line. Run from the repo root:  python -m paper.cli <command> ...

  auto            the one command the scheduler runs (every ~15 min): builds the day's slate,
                  captures whatever is due, grades finished dates, rebuilds ledgers + report.md
  discover        list live NBA markets/books; confirms the ids in markets.py
  check-results   --date D   first live check of the results path (Optic player-results reconcile?)
  backfill-logs   --through D   pull the season's box scores into the model's game logs
  slate / capture / grade   the individual steps of `auto`, for debugging
  replay          [--basis proj|pool|sharp]   replay every persona; print ROI table
  report          [--preset realistic|sharp|ladder|everything]   filtered paper trading
"""

from __future__ import annotations

import argparse
import json
import sys

from . import store


def _read_frame(path):
    import pandas as pd
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)


def _pct(x):
    return "   —  " if x is None else f"{x * 100:+6.1f}%"


def cmd_discover(a):
    from .books import books_to_fetch, resolve_book_ids
    from .markets import STAT_DEFS
    from .optic import OpticOddsClient
    c = OpticOddsClient()
    markets = c.get_markets()
    live = {m.get("id") for m in markets}
    print(f"{len(live)} NBA markets live.")
    for k, d in STAT_DEFS.items():
        print(f"  {'ok     ' if d['optic_id'] in live else 'MISSING'} {k:7s} {d['optic_id']}")
    res = resolve_book_ids(books_to_fetch(), c.get_sportsbooks())
    print("books resolved:", res["matched"], "| missing:", res["missing"], "| ambiguous:", res["ambiguous"])


def _overrides(path):
    import os
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper_overrides.json")
    return json.load(open(path)) if os.path.exists(path) else None


def cmd_auto(a):
    from datetime import datetime, timezone
    from .automation import auto
    from .optic import OpticOddsClient
    now = datetime.fromisoformat(a.now.replace("Z", "+00:00")) if a.now else datetime.now(timezone.utc)
    errs = auto(a.root, OpticOddsClient(), now=now, overrides=_overrides(a.overrides), n_sims=a.sims)
    if errs:
        sys.exit(1)


def cmd_slate(a):
    from datetime import datetime, timezone
    from .automation import ensure_slate
    from .optic import OpticOddsClient
    ok = ensure_slate(a.root, a.date, OpticOddsClient(), datetime.now(timezone.utc), n_sims=a.sims)
    if not ok:
        sys.exit("slate not built (no fixtures or no spread/total lines yet)")


def cmd_capture(a):
    from .automation import cap
    from .capture import list_fixtures
    from .optic import OpticOddsClient
    c = OpticOddsClient()
    ids = [f["id"] for f in list_fixtures(c, a.date)]
    s = cap.capture(a.root, c, a.date, a.slot, ids, stats=a.stats.split(",") if a.stats else None,
                    overrides=_overrides(a.overrides))
    print(f"{a.date} {a.slot}: {s['fixtures']} fixtures, {s['quotes']} quotes, {s['matched']} matched")
    for u in s["unmatched"][:25]:
        print(f"  unmatched: {u['name']} ({u['reason']})")


def cmd_grade(a):
    from .automation import grade_date
    from .optic import OpticOddsClient
    ok = grade_date(a.root, a.date, OpticOddsClient())
    print(f"{a.date}: {'graded' if ok else 'not graded (games unfinished or box score did not reconcile)'}")


def cmd_check_results(a):
    """First live check of the results path: what Optic returned, and whether each box
    score passes both integrity checks."""
    from . import results as R
    from .automation import board_fixture_ids
    from .optic import OpticOddsClient
    c = OpticOddsClient()
    ids = a.fixture or board_fixture_ids(a.root, a.date)
    teams = {g["fixture_id"]: g for g in R.parse_optic_results(c.get_fixture_results(ids))}
    players = R.parse_player_results(c.get_player_results(ids))
    print(f"{len(teams)} fixtures, {sum(g['completed'] for g in teams.values())} completed")
    for g in players:
        t = teams.get(g["fixture_id"])
        bad = R.verify_box(g["rows"], t) if t else ["no team result"]
        print(f"  {g['away']}@{g['home']}: {g['status']}, {len(g['rows'])} players played, "
              f"{'box OK' if not g['errors'] and not bad else 'PROBLEM ' + '; '.join((g['errors'] + bad)[:3])}")


def cmd_backfill_logs(a):
    from datetime import datetime, timezone
    from . import sources
    from .automation import inputs_dir, season_start, update_logs
    from .optic import OpticOddsClient
    from . import slate
    now = datetime.now(timezone.utc)
    pre = sources.load_preseason(inputs_dir(a.root) / "preseason.csv")
    update_logs(a.root, OpticOddsClient(), slate.slate_players(pre), a.through, start=a.start or season_start(now))


def cmd_replay(a):
    from .replay import replay
    res = replay(a.root, basis=a.basis, write=not a.dry_run)
    print(f"{'persona':13s}{'bets':>6s}{'settled':>9s}{'staked':>9s}{'pnl':>9s}{'roi':>9s}  bankroll")
    for pid, r in res.items():
        t = r["totals"]
        line = (f"{pid:13s}{t['bets']:6d}{t['settled']:9d}{t['staked']:9.1f}"
                f"{t['pnl']:+9.2f}{_pct(r['roi']):>9s}")
        if r["persona"]["bankroll"]["kind"] == "compounding":
            line += f"  {r['bankroll']:.1f}u"
        print(line)
    print("\nNote: personas show what a strategy would have FELT like. Only the firehose (every edge)\n"
          "has enough bets to say whether the simulator is real -- read its verdict, not theirs.")


def cmd_report(a):
    from .paper_filter import bets_to_resolve, preset_filters
    from .replay import filter_report
    r = filter_report(a.root, preset_filters(a.preset))
    s = r["summary"]
    print(f"preset={a.preset}: {s['bets']} bets, {s['settled']} settled "
          f"({s['won']}W/{s['lost']}L/{s['push']}P/{s['void']} void, {s['unplayed']} unplayed)")
    print(f"ROI {_pct(s['roi'])} ± {_pct(s['roi_band']).strip() if s['roi_band'] is not None else '—'}  "
          f"verdict: {r['verdict']}")
    if s["clv"]:
        c = s["clv"]
        print(f"CLV avg {c['avg'] * 100:+.2f} pts over {c['n']} bets, beat the close {c['beat_rate'] * 100:.0f}% "
              f"(±{c['beat_band'] * 100:.0f}% is noise)")
    n = bets_to_resolve(s["roi"], s["per_bet_sd"])
    if n:
        print(f"bets needed to resolve an ROI this size at 2σ: ~{n}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="paper", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None, help="data root (default <repo>/data/paper)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("discover").set_defaults(fn=cmd_discover)
    au = sub.add_parser("auto"); au.set_defaults(fn=cmd_auto)
    au.add_argument("--now", help="override the clock (ISO, UTC) for testing"); au.add_argument("--overrides")
    au.add_argument("--sims", type=int, default=2000)
    s = sub.add_parser("slate"); s.set_defaults(fn=cmd_slate)
    s.add_argument("--date", required=True); s.add_argument("--sims", type=int, default=2000)
    c = sub.add_parser("capture"); c.set_defaults(fn=cmd_capture)
    c.add_argument("--date", required=True); c.add_argument("--slot", required=True)
    c.add_argument("--stats"); c.add_argument("--overrides")
    g = sub.add_parser("grade"); g.set_defaults(fn=cmd_grade); g.add_argument("--date", required=True)
    ck = sub.add_parser("check-results"); ck.set_defaults(fn=cmd_check_results)
    ck.add_argument("--date", required=True); ck.add_argument("--fixture", action="append")
    bf = sub.add_parser("backfill-logs"); bf.set_defaults(fn=cmd_backfill_logs)
    bf.add_argument("--through", required=True); bf.add_argument("--start")
    r = sub.add_parser("replay"); r.set_defaults(fn=cmd_replay)
    r.add_argument("--basis", default="proj", choices=["proj", "pool", "sharp"])
    r.add_argument("--dry-run", action="store_true")
    rp = sub.add_parser("report"); rp.set_defaults(fn=cmd_report)
    rp.add_argument("--preset", default="realistic")
    a = p.parse_args(argv)
    a.root = a.root or store.DEFAULT_ROOT
    a.fn(a)


if __name__ == "__main__":
    main()

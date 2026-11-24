"""Command line. Run from the repo root:  python -m paper.cli <command> ...

  discover                         list live NBA markets/books; confirms the ids in markets.py
  slate    --date D --preseason P [--logs L] [--crosswalk X]
                                   run projection+simulation, save draws and the slate's players
  capture  --date D --slot morning|pregame|closing
                                   pull OpticOdds, crosswalk, save the board, rebuild candidates
  grade    --date D --logs L       save box scores for settlement
  replay   [--basis proj|pool|sharp]   replay every persona; print ROI table, write ledgers
  report   [--preset realistic|sharp|ladder|everything]   filtered paper trading + verdict
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


def cmd_slate(a):
    from . import slate
    from .capture import save_slate_players
    from .optic import OpticOddsClient
    from .capture import slate_window
    pre = _read_frame(a.preseason)
    logs = _read_frame(a.logs) if a.logs else None
    if logs is not None and "player_id" not in logs.columns:
        if not a.crosswalk:
            sys.exit("logs have no player_id: pass --crosswalk (newsid,nba_player_id) to map them")
        xw = _read_frame(a.crosswalk)[["newsid", "nba_player_id"]].dropna()
        logs = logs.merge(xw, on="nba_player_id").rename(columns={"newsid": "player_id"})
    c = OpticOddsClient()
    start, end = slate_window(a.date)
    lines = c.get_odds_for_fixtures(
        [f["id"] for f in c.get_fixtures(start_date=start, end_date=end)], ["pinnacle"],
        ["point_spread", "total_points"])
    games = slate.games_from_lines(lines)
    if games.empty:
        sys.exit("no spread/total lines found for that date")
    players, sim, proj = slate.build_draws(a.date, pre, games, logs, n_sims=a.sims, seed=a.seed)
    slate.save_slate(a.root, a.date, players, sim)
    save_slate_players(a.root, a.date, slate.slate_players(pre, proj))
    print(f"{a.date}: {len(players)} players, {len(games) // 2} games, {a.sims} sims saved")


def cmd_capture(a):
    from .capture import capture
    from .optic import OpticOddsClient
    ov = json.load(open(a.overrides)) if a.overrides else None
    s = capture(a.root, OpticOddsClient(), a.date, a.slot,
                stats=a.stats.split(",") if a.stats else None, overrides=ov)
    print(f"{a.date} {a.slot}: {s['fixtures']} fixtures, {s['quotes']} quotes, {s['matched']} matched")
    for u in s["unmatched"][:25]:
        print(f"  unmatched: {u['name']} ({u['reason']})")
    if len(s["unmatched"]) > 25:
        print(f"  ... and {len(s['unmatched']) - 25} more")


def cmd_grade(a):
    from .actuals import actuals_from_logs
    logs = _read_frame(a.logs)
    rows = actuals_from_logs(logs, a.date)
    store.save_actuals(a.root, a.date, rows)
    print(f"{a.date}: {len(rows)} box-score rows saved")


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
    s = sub.add_parser("slate"); s.set_defaults(fn=cmd_slate)
    s.add_argument("--date", required=True); s.add_argument("--preseason", required=True)
    s.add_argument("--logs"); s.add_argument("--crosswalk")
    s.add_argument("--sims", type=int, default=4000); s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("capture"); c.set_defaults(fn=cmd_capture)
    c.add_argument("--date", required=True); c.add_argument("--slot", required=True)
    c.add_argument("--stats"); c.add_argument("--overrides")
    g = sub.add_parser("grade"); g.set_defaults(fn=cmd_grade)
    g.add_argument("--date", required=True); g.add_argument("--logs", required=True)
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

"""Everything the repo does by itself. `python -m paper.cli auto` is run every
~15 minutes by .github/workflows/paper-trading.yml and decides, from the clock and
the fixtures' tip times, what is due:

  build the slate   once per day, before the first capture: pull RotoWire's sheet
                    (cached), bring the game logs up to date from the results feed,
                    pull spread/total, run projection + simulation, save the draws
  morning           every fixture, once, from 10:00 ET (15:00Z) until it tips
  pregame           each fixture ~2 hours before ITS tip
  closing           each fixture in the last ~25 minutes before ITS tip (reference only)
  grade             every earlier date whose games are all final: save the box scores
                    and extend the game logs
  replay + report   rebuild every persona ledger and write report.md

Windows are per fixture because tips are spread over ~8 hours; a single
"closing" board would be hours stale for the late games. A window that was missed
(cron delay, outage) is reported and skipped, never backfilled with a later price:
a paper bet must use a price that existed at the time.

Each step is isolated: a failure in one is printed and the rest still run, then
the exit code is non-zero so the Action goes red.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import capture as cap, store
from .candidates import SLOT_ORDER

MORNING_AT_UTC_HOUR = 15          # 10:00 ET (11:00 during standard time; adjust if you care)
PREGAME_LEAD = timedelta(hours=2)
PREGAME_CUTOFF = timedelta(minutes=30)   # a pregame taken closer than this to the close is pointless
CLOSING_LEAD = timedelta(minutes=25)


def slate_date(now: datetime) -> str:
    """The slate day `now` belongs to: days run 10:00Z to 10:00Z, covering US
    evening tips that cross UTC midnight."""
    return (now - timedelta(hours=10)).strftime("%Y-%m-%d")


def plan_captures(now, fixtures, captured, morning_hour=MORNING_AT_UTC_HOUR):
    """Pure. fixtures: [{id, start}]; captured: {slot: set(fixture_id)}.
    Returns (due: {slot: [fixture_id]}, missed: [(slot, fixture_id)])."""
    due = {s: [] for s in SLOT_ORDER}
    missed = []
    for f in fixtures:
        fid, tip = f["id"], f["start"]
        have = lambda slot: fid in captured.get(slot, set())  # noqa: E731
        if tip <= now:
            continue                                          # tipped: nothing can be bet
        # morning: once the morning window has opened (a fixture first seen late gets
        # whatever the board is then, labelled by the slot it was actually taken in)
        if not have("morning") and now >= _morning_open(now, tip, morning_hour):
            due["morning"].append(fid)
        if not have("pregame"):
            if tip - PREGAME_LEAD <= now < tip - PREGAME_CUTOFF:
                due["pregame"].append(fid)
            elif now >= tip - PREGAME_CUTOFF:
                missed.append(("pregame", fid))
        if not have("closing") and tip - CLOSING_LEAD <= now < tip:
            due["closing"].append(fid)
    return due, missed


def _morning_open(now, tip, hour):
    """15:00Z on the UTC day the slate starts (the slate day's own date)."""
    day = (tip - timedelta(hours=10)).replace(hour=0, minute=0, second=0, microsecond=0)
    return day + timedelta(hours=hour)


def captured_fixtures(root, date):
    return {slot: {str(r["fixture_id"]) for r in (store.load_board(root, date, slot) or [])}
            for slot in store.board_slots(root, date)}


# ---- inputs ------------------------------------------------------------------
def inputs_dir(root) -> Path:
    return Path(root) / "inputs"


def season_start(now: datetime) -> str:
    y = now.year if now.month >= 9 else now.year - 1
    return f"{y}-10-01"


def update_logs(root, results, players, through_date, start=None, sleep=0.15, log=print):
    """Bring logs.csv up to `through_date` (inclusive) from the results feed.
    Idempotent and incremental: only dates after the last logged one are pulled."""
    from . import results as R
    path = inputs_dir(root) / "logs.csv"
    have = R.read_logs(path)
    last = have["game_date"].max().strftime("%Y-%m-%d") if len(have) else None
    first = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d") if last else start
    if first is None or first > through_date:
        return have
    added, unmatched = 0, {}
    for d in R.season_dates(first, through_date):
        _, rows, games = results.day(d)
        if not rows:
            continue
        matched, un = R.attach_player_ids(rows, players)
        added += R.append_logs(path, R.to_logs(matched))
        for u in un:
            unmatched[u["name"]] = u["reason"]
        time.sleep(sleep)
    log(f"  logs: +{added} player-games through {through_date}"
        + (f"; {len(unmatched)} unmatched names (e.g. {', '.join(list(unmatched)[:5])})" if unmatched else ""))
    return R.read_logs(path)


def games_for(client, fixtures_ids, books=("pinnacle", "draftkings")):
    from . import slate
    for book in books:
        payloads = client.get_odds_for_fixtures(fixtures_ids, [book], ["point_spread", "total_points"])
        g = slate.games_from_lines(payloads)
        if not g.empty:
            return g
    return None


def ensure_slate(root, date, client, results, now, n_sims=2000, log=print, session=None):
    """Build the day's draws once. Returns True when a slate exists."""
    if store.load_sim(root, date)[1] is not None and cap.load_slate_players(root, date) is not None:
        return True
    from . import slate, sources
    pre = sources.load_preseason(inputs_dir(root) / "preseason.csv", session=session)
    players = slate.slate_players(pre)
    fixtures = cap.list_fixtures(client, date)
    if not fixtures:
        log(f"  {date}: no fixtures yet")
        return False
    logs = update_logs(root, results, players, (now - timedelta(days=1)).strftime("%Y-%m-%d"),
                       start=season_start(now), log=log)
    games = games_for(client, [f["id"] for f in fixtures])
    if games is None:
        log(f"  {date}: no spread/total lines posted yet; slate not built")
        return False
    log_in = logs if len(logs) else None
    proj_players, sim, proj = slate.build_draws(date, pre, games, log_in, n_sims=n_sims,
                                                seed=int(date.replace("-", "")) % 100000)
    keep = (proj_players["minutes_p50"] >= 8).to_numpy() if "minutes_p50" in proj_players else None
    if keep is not None:
        import numpy as np
        sim = {k: v[:, keep] for k, v in sim.items()}
        proj_players = proj_players[keep].reset_index(drop=True)
    slate.save_slate(root, date, proj_players, sim)
    cap.save_slate_players(root, date, slate.slate_players(pre, proj))
    log(f"  {date}: slate built ({len(proj_players)} players, {len(games) // 2} games)")
    return True


def board_fixture_ids(root, date):
    ids = set()
    for slot in store.board_slots(root, date):
        ids |= {str(r["fixture_id"]) for r in (store.load_board(root, date, slot) or [])}
    return sorted(ids)


def grade_date(root, date, client, results, log=print):
    """Grade one date if, and only if, every fixture we captured is final AND its
    box score reconciles with Optic's team totals. Returns True when saved."""
    from . import actuals, results as R
    fixture_ids = board_fixture_ids(root, date)
    if not fixture_ids:
        return False
    games = R.parse_optic_results(client.get_fixture_results(fixture_ids))
    live = [g for g in games if not g["dead"]]
    if len(games) < len(fixture_ids) or not all(g["completed"] for g in live) or not live:
        return False                                   # not everything has finished yet
    rows = []
    for g in live:
        sched = results.find_game(g["home"], g["away"], date)
        if sched is None:
            log(f"  grade {date}: {g['away']}@{g['home']} not on the nba.com schedule; will retry")
            return False
        final, box = results.box(sched["game_id"], sched["date"])
        if not final:
            return False                               # CDN lags Optic: retry next run
        bad = R.verify_box(box, g)
        if bad:
            log(f"  grade {date}: {g['away']}@{g['home']} box score does not reconcile "
                f"({'; '.join(bad[:3])}); will retry")
            return False
        rows += box
    players = cap.load_slate_players(root, date) or []
    matched, un = R.attach_player_ids(rows, players)
    store.save_actuals(root, date, actuals.actuals_from_rows(matched, date))
    from . import results as _r
    _r.append_logs(inputs_dir(root) / "logs.csv", _r.to_logs(matched))
    if un:
        log(f"  grade {date}: {len(un)} box-score names unmatched, e.g. {', '.join(u['name'] for u in un[:5])}")
    return True


def grade_pending(root, client, results, log=print):
    """Save box scores for every captured date that has newly become fully final."""
    done = [d for d in store.all_dates(root)
            if store.load_actuals(root, d) is None and grade_date(root, d, client, results, log)]
    if done:
        log(f"  graded: {', '.join(done)}")
    return done


def write_report(root, path=None):
    from .paper_filter import PRESETS, bets_to_resolve, preset_filters
    from .replay import filter_report, replay
    res = replay(root, write=True)
    lines = [f"# NBA paper trading report", "", f"_Updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M}Z_", "",
             "## Personas", "", "| persona | bets | settled | staked | pnl | roi |", "|---|---|---|---|---|---|"]
    for pid, r in res.items():
        t = r["totals"]
        roi = "—" if r["roi"] is None else f"{r['roi'] * 100:+.1f}%"
        lines.append(f"| {pid} | {t['bets']} | {t['settled']} | {t['staked']:.1f} | {t['pnl']:+.2f} | {roi} |")
    lines += ["", "Personas show what a strategy would have *felt* like. Only the filter table below, with",
              "far more bets, can say whether the simulator is real.", "", "## Filtered paper trading", "",
              "| preset | bets | settled | roi | ±2σ | verdict | CLV beat rate |", "|---|---|---|---|---|---|---|"]
    for p in PRESETS:
        r = filter_report(root, preset_filters(p["id"]))
        s = r["summary"]
        roi = "—" if s["roi"] is None else f"{s['roi'] * 100:+.1f}%"
        band = "—" if s["roi_band"] is None else f"{s['roi_band'] * 100:.1f}%"
        clv = "—" if not s["clv"] else f"{s['clv']['beat_rate'] * 100:.0f}% (n={s['clv']['n']})"
        lines.append(f"| {p['label']} | {s['bets']} | {s['settled']} | {roi} | {band} | {r['verdict']} | {clv} |")
    out = Path(path or Path(root) / "report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


def auto(root, client, results, now=None, overrides=None, log=print, n_sims=2000, session=None):
    now = now or datetime.now(timezone.utc)
    date = slate_date(now)
    errors = []

    def step(name, fn):
        try:
            return fn()
        except Exception as err:        # one failing step must not stop the others
            errors.append(f"{name}: {err}")
            log(f"  !! {name} failed: {err}")
            return None

    log(f"auto @ {now:%Y-%m-%d %H:%MZ}  slate {date}")
    fixtures = step("list fixtures", lambda: cap.list_fixtures(client, date)) or []
    captured = captured_fixtures(root, date)
    due, missed = plan_captures(now, fixtures, captured)
    for slot, fid in missed:
        log(f"  missed {slot} window for fixture {fid}; skipped (never backfilled with a later price)")
    if any(due.values()):
        if step("build slate", lambda: ensure_slate(root, date, client, results, now, n_sims, log, session)):
            for slot in SLOT_ORDER:
                if due[slot]:
                    s = step(f"capture {slot}", lambda slot=slot: cap.capture(
                        root, client, date, slot, due[slot], overrides=overrides))
                    if s:
                        log(f"  {slot}: {s['fixtures']} fixtures, {s['matched']}/{s['quotes']} quotes matched"
                            + (f", {len(s['unmatched'])} players unmatched" if s["unmatched"] else ""))
    step("grade", lambda: grade_pending(root, client, results, log))
    step("report", lambda: write_report(root))
    return errors

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


MIN_PLAYERS_PER_FIXTURE = 6      # a game's props cover ~10-14 players; fewer means a partial capture
MIN_LOG_PLAYERS = 100            # a season of box scores covers hundreds of players
MIN_SIM_PLAYERS = 100            # so does a day's simulation


def captured_fixtures(root, date):
    """{slot: fixtures captured}. A fixture only counts once it has a real number of players on the
    board: a first run that matched a single player left that game with one row, the scheduler called
    it captured, and it was never redone."""
    out = {}
    for slot in store.board_slots(root, date):
        players = {}
        for r in store.load_board(root, date, slot) or []:
            players.setdefault(str(r["fixture_id"]), set()).add(r.get("player_id"))
        out[slot] = {fx for fx, p in players.items() if len(p) >= MIN_PLAYERS_PER_FIXTURE}
    return out


# ---- inputs ------------------------------------------------------------------
def inputs_dir(root) -> Path:
    return Path(root) / "inputs"


def season_start(now: datetime) -> str:
    y = now.year if now.month >= 9 else now.year - 1
    return f"{y}-10-01"


def completed_fixture_ids(client, date):
    """Completed fixtures on a slate date (same exact window as the odds capture)."""
    return [f["id"] for f in cap.fixtures_in_window(client, date, "completed")]


def update_logs(root, client, players, through_date, start=None, log=print):
    """Bring logs.csv (the model's game logs) up to `through_date` from Optic's
    player-results. Idempotent and incremental: only dates after the last logged
    one are pulled; the first run backfills from `start` (the season's opening)."""
    from . import results as R
    path = inputs_dir(root) / "logs.csv"
    idmap_path = inputs_dir(root) / "optic_ids.csv"
    have = R.read_logs(path)
    if len(have) and have["player_id"].nunique() < MIN_LOG_PLAYERS:
        # Logs matched against a truncated preseason sheet hold a few players and look "up to date"
        # by date, so the incremental pull would never repair them. Start over.
        log(f"  logs: only {have['player_id'].nunique()} players logged (built from an incomplete sheet); rebuilding")
        path.unlink()
        have = R.read_logs(path)
    last = have["game_date"].max().strftime("%Y-%m-%d") if len(have) else None
    first = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d") if last else start
    if first is None or first > through_date:
        return have
    added, unmatched, bad_games = 0, {}, 0
    for d in R.season_dates(first, through_date):
        ids = completed_fixture_ids(client, d)
        if not ids:
            continue
        for g in R.parse_player_results(client.get_player_results(ids)):
            if not g["completed"] or g["errors"]:
                bad_games += 1                      # never log a box that disagrees with Optic's own grading
                continue
            matched, un = R.attach_player_ids(g["rows"], players, R.read_id_map(idmap_path))
            R.remember_ids(idmap_path, matched)
            added += R.append_logs(path, R.to_logs(matched))
            for u in un:
                unmatched[u["name"]] = u["reason"]
    log(f"  logs: +{added} player-games through {through_date}"
        + (f"; {bad_games} games skipped (not final / disagreed with Optic)" if bad_games else "")
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


def ensure_slate(root, date, client, now, n_sims=2000, log=print, session=None, save_players=True):
    """Build the day's draws once. Returns True when a slate exists."""
    ids, sim0 = store.load_sim(root, date)
    if sim0 is not None and cap.load_slate_players(root, date) is not None:
        if len(ids) >= MIN_SIM_PLAYERS:
            return True
        log(f"  {date}: the stored simulation has only {len(ids)} players (built from an incomplete sheet); rebuilding")
    from . import slate, sources
    pre = sources.load_preseason(inputs_dir(root) / "preseason.csv", session=session)
    players = slate.slate_players(pre)
    fixtures = cap.list_fixtures(client, date)
    if not fixtures:
        log(f"  {date}: no fixtures yet")
        return False
    logs = update_logs(root, client, players, (now - timedelta(days=1)).strftime("%Y-%m-%d"),
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
    if save_players:      # in `both` mode the crosswalk uses the feed's players (see capture)
        cap.save_slate_players(root, date, slate.slate_players(pre, proj))
    log(f"  {date}: slate built ({len(proj_players)} players, {len(games) // 2} games)")
    return True


def board_fixture_ids(root, date):
    ids = set()
    for slot in store.board_slots(root, date):
        ids |= {str(r["fixture_id"]) for r in (store.load_board(root, date, slot) or [])}
    return sorted(ids)


def grade_date(root, date, client, log=print):
    """Grade one date if, and only if, every captured fixture is final AND every
    box score passes both integrity checks (see results.py). Returns True if saved."""
    from . import actuals, results as R
    fixture_ids = board_fixture_ids(root, date)
    if not fixture_ids:
        return False
    teams = R.parse_optic_results(client.get_fixture_results(fixture_ids))
    live = [g for g in teams if not g["dead"]]
    if len(teams) < len(fixture_ids) or not live or not all(g["completed"] for g in live):
        return False                                   # something has not finished yet
    by_fx = {g["fixture_id"]: g for g in R.parse_player_results(
        client.get_player_results([g["fixture_id"] for g in live]))}
    rows = []
    for t in live:
        g = by_fx.get(t["fixture_id"])
        if g is None or not g["completed"]:
            return False
        if g["errors"]:
            log(f"  grade {date}: {t['away']}@{t['home']} box disagrees with Optic's grading "
                f"({'; '.join(g['errors'][:3])}); will retry")
            return False
        bad = R.verify_box(g["rows"], t)
        if bad:
            log(f"  grade {date}: {t['away']}@{t['home']} box does not reconcile with team totals "
                f"({'; '.join(bad[:3])}); will retry")
            return False
        rows += g["rows"]
    players = cap.load_slate_players(root, date) or []
    idmap_path = inputs_dir(root) / "optic_ids.csv"
    matched, un = R.attach_player_ids(rows, players, R.read_id_map(idmap_path))
    R.remember_ids(idmap_path, matched)
    store.save_actuals(root, date, actuals.actuals_from_rows(matched, date))
    R.append_logs(inputs_dir(root) / "logs.csv", R.to_logs(matched))
    if un:
        log(f"  grade {date}: {len(un)} box-score names not on the slate, e.g. {', '.join(u['name'] for u in un[:5])}")
    return True


def grade_pending(root, client, log=print):
    """Save box scores for every captured date that has newly become fully final."""
    done = [d for d in store.all_dates(root)
            if store.load_actuals(root, d) is None and grade_date(root, d, client, log)]
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
    from . import feeds
    cal = feeds.calibration(root, store.all_dates(root))
    if cal:
        lines += ["", "## Is the feed's floor/ceiling calibrated?", "",
                  "Realised vs what the fitted distribution implies. Large gaps mean prices are not to be trusted",
                  "for that stat (an over-tight band prices every long-shot over as value).", "",
                  "| stat | n | below floor (real / exp) | above ceiling (real / exp) | inside band (real / exp) |",
                  "|---|---|---|---|---|"]
        for stat, c in sorted(cal.items()):
            lines.append(f"| {stat} | {c['n']} | {c['below'] * 100:.0f}% / {c['exp_below'] * 100:.0f}% | "
                         f"{c['above'] * 100:.0f}% / {c['exp_above'] * 100:.0f}% | "
                         f"{c['inside'] * 100:.0f}% / {c['exp_inside'] * 100:.0f}% |")
    out = Path(path or Path(root) / "report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    stamp = lambda t: "\n".join(l for l in t.splitlines() if not l.startswith("_Updated"))  # noqa: E731
    if not out.exists() or stamp(out.read_text()) != stamp(text):   # an unchanged report is not rewritten:
        out.write_text(text)                                        # otherwise every run is a data commit
    return out


PROBE_EVERY = timedelta(minutes=30)


def _probe_file(root):
    return inputs_dir(root) / "fixtures_probe.json"


def fixtures_with_throttle(root, client, date, now, log):
    """List the slate's fixtures, but after an EMPTY answer wait PROBE_EVERY before asking again.
    A day with no games (or none posted yet) otherwise spends several requests every 15 minutes
    for nothing. Games posted late are picked up within PROBE_EVERY, well inside the 2h pregame window."""
    import json
    path = _probe_file(root)
    if path.exists():
        try:
            until = datetime.fromisoformat(json.loads(path.read_text())["until"])
            if now < until:
                log(f"  no fixtures at the last check; asking again after {until:%H:%MZ}")
                return []
        except (ValueError, KeyError):
            pass
    fixtures = cap.list_fixtures(client, date, log)
    if fixtures:
        path.unlink(missing_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"until": (now + PROBE_EVERY).isoformat()}))
    return fixtures


def auto(root, client, now=None, overrides=None, log=print, n_sims=2000, session=None,
         price_model="both"):
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

    from .markets import COMBO_STATS, DEFAULT_STATS
    stats = DEFAULT_STATS + COMBO_STATS if price_model in ("both", "sim") else DEFAULT_STATS
    log(f"auto @ {now:%Y-%m-%d %H:%MZ}  slate {date}")
    fixtures = step("list fixtures", lambda: fixtures_with_throttle(root, client, date, now, log)) or []
    captured = captured_fixtures(root, date)
    due, missed = plan_captures(now, fixtures, captured)
    log(f"  due: " + (", ".join(f"{k} x{len(v)}" for k, v in due.items() if v) or "nothing")
        + (f" (already captured: {', '.join(f'{k} x{len(v)}' for k, v in captured.items())})" if captured else ""))
    for slot, fid in missed:
        log(f"  missed {slot} window for fixture {fid}; skipped (never backfilled with a later price)")
    if any(due.values()):
        # feeds: nothing to build (the RotoWire feeds are fetched at each capture).
        # sim: the day's draws are required. both: the draws are only for the combo markets, so a
        # simulator failure must NOT block the feed-priced captures; it is logged and combos wait.
        if price_model == "feeds":
            ready = True
        else:
            built = step("build slate", lambda: ensure_slate(
                root, date, client, now, n_sims, log, session, save_players=(price_model == "sim")))
            ready = True if price_model == "both" else built
        if ready:
            for slot in SLOT_ORDER:
                if due[slot]:
                    s = step(f"capture {slot}", lambda slot=slot: cap.capture(
                        root, client, date, slot, due[slot], overrides=overrides, stats=stats,
                        price_model=price_model, session=session))
                    if s:
                        log(f"  {slot}: {s['fixtures']} fixtures, {s['matched']}/{s['quotes']} quotes matched"
                            + (f", {len(s['unmatched'])} players unmatched" if s["unmatched"] else ""))
    step("grade", lambda: grade_pending(root, client, log))
    step("report", lambda: write_report(root))
    step("export site data", lambda: __import__("paper.export", fromlist=["export_site"]).export_site(root))
    return errors

# NBA paper trading

Multi-persona paper trading and sharp-book analysis for the NBA simulator, ported
from the NFL dashboard repo (`personas.mjs`, `paper-filter.mjs`, `consensus.mjs`,
`source-weights.mjs`). Standalone: only `slate.py` touches the model modules.

## Everything runs itself
`.github/workflows/paper-trading.yml` runs `python -m paper.cli auto` every 15 minutes.
Needs one repo secret, `OPTICODDS_API_KEY`. Each run decides from the clock and the
fixtures' tip times what is due:

* **price source** (default, `--price-model both`: singles from the feeds, combos from the simulator; `feeds` = singles only; `sim` = simulator only): RotoWire's daily floor/median/ceiling
  feeds, fetched AT EACH CAPTURE and stored per slot, so a pregame bet is priced off the feed
  as it stood at pregame (late minutes/injury news included). Nothing to train or simulate.
  Each stat becomes a negative binomial with the feed's mean, whose dispersion is the one whose
  quartiles reproduce the published floor/ceiling (`dist.py`). The ceiling feed is direction-aware
  (for turnovers/fouls it is the LOW tail), so lower/upper quartile = min/max of the two feeds.
  `--price-model sim` instead runs this repo's simulator once a day (preseason sheet + logs +
  spread/total) and is the only way to price the combo markets (PRA etc.), which need a joint
  distribution that three published marginals cannot give. In `both` mode a combo needs BOTH the
  day's draws and the player still listed in the feed (injured players are dropped from it), and a
  simulator failure never blocks the feed-priced captures.
* **morning** every fixture from 10:00 ET; **pregame** each fixture ~2h before ITS tip;
  **closing** each fixture in the last ~25 min before ITS tip (reference only, never bet)
* **grade** a date once every captured fixture is `completed`, the box score reproduces
  Optic's own `market_stats` for every player, and per-team sums match Optic's team totals
* **replay** every persona and rewrite `report.md`

A window missed (cron delay, outage) is reported and skipped, never backfilled with a
later price. Data is committed to a separate `paper-data` branch.

Results come from OpticOdds only: `/fixtures/results` (finality, team totals) and
`/fixtures/player-results` (player lines + `market_stats`). A player's Optic id is
the same id the odds carry, so a bet is matched to its result by id; names are used
once, at capture, to map an Optic id to our RotoWire player_id (inputs/optic_ids.csv).

Manual commands, for debugging: `discover`, `check-feeds`, `check-results --date D`, `backfill-logs
--through D`, `slate`, `capture`, `grade`, `replay`, `report --preset ...`.

## Secrets
Two **repository** secrets (Settings > Secrets and variables > Actions; not environment secrets, the job
declares no environment):

* `OPTICODDS_API_KEY`: the odds and results feed.
* `ROTOWIRE_COOKIE`: a logged-in session's `Cookie` header. RotoWire serves only a small preview to an
  anonymous request (the NFL repo documents ~top 10; here it came back as 7 players, all A-names) and the
  full tables only to a logged-in session. To get it: log in to rotowire.com, open DevTools > Network, load a
  projections page, click the `projections.php` request and copy the whole `Cookie` request header.
  Sessions expire: when it does, the capture stops with "RotoWire feeds list only N players for TEAM ...
  (set ROTOWIRE_COOKIE)" and the job goes red. Refresh the secret.

## Dashboard
`web/` is a Next.js app (deploy on Vercel with Root Directory `web`; see `web/README.md`) that reads the
`site/` JSON the Action writes to `paper-data`: persona results, the filtered paper-trading view and
feed calibration.

## What is durable and what is derived
`boards/` (raw quotes) and `sims/` (the draws the day was priced from) are the
record: never regenerate them, because a later model state would break
point-in-time honesty. `candidates/` and `ledgers/` are derived and rebuilt from
scratch on every run, so changing a persona rule is just a re-run.

## Where it differs from the NFL version
* **Price comes from the simulator's draws**, not a fitted two-piece normal:
  `P(over line)` is the share of draws beyond the line, and combo markets (PRA)
  are coherent because every stat in a draw is from one simulated game.
* **Time unit is the slate day** (NFL: the week); persona limits were rescaled.
* **A DNP is `void`** (stake refunded, as sportsbooks do), not pending/lost.
* **Under-only quotes are kept** (mirrored in the consensus); NBA props bet both sides.
* **Circa is a reference book here** (priced, never staked). Move it to
  `BETTABLE_BOOKS` in `books.py` if you hold an account.
* A line that moves between bet and close gets **no numeric CLV** (it is a
  different bet); it is not turned into a made-up price.

## Not modelled (on purpose)
Half/quarter/first-N-minutes markets, (Combo)/(Either)/(Each)/H2H markets, team
totals, first-basket and most-X markets, Player Threes Attempted. See `markets.py`.

## Verified vs not
Verified by `pytest paper/tests` (88 tests, offline): pricing, de-vig, consensus and
weight fitting (recovers a planted lead), personas, filters, normalization, crosswalk,
settlement incl. DNP/pending, CLV, point-in-time fitting, client batching/retry/key
handling, the scheduler's windows, player-results parsing against the real payload
shape, and a full slate day (RotoWire JSON -> real model -> morning/pregame/closing ->
grading -> ledgers) against fakes.

**Not verified against live endpoints** (blocked in the build session): the three RotoWire
feeds (the median/ceiling rows are real samples; the floor feed has not been seen), the
preseason fetch used by `--price-model sim`, the OpticOdds NBA odds/fixtures calls, the spread/total
parsing on a real payload, and that Optic's player id on a prop equals `player.id` in
player-results (expected from the id format; `check-results` shows it). Run
`discover`, `check-feeds` and `check-results` first. `report.md` also carries a calibration table
(realised vs implied share below floor / above ceiling): read it before trusting any price.

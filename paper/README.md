# NBA paper trading

Multi-persona paper trading and sharp-book analysis for the NBA simulator, ported
from the NFL dashboard repo (`personas.mjs`, `paper-filter.mjs`, `consensus.mjs`,
`source-weights.mjs`). Standalone: only `slate.py` touches the model modules.

## The loop (per slate date `D`)

```bash
python -m paper.cli discover                                   # once: confirm market ids + book names
python -m paper.cli slate   --date D --preseason pre.csv --logs logs.parquet --crosswalk xw.csv
python -m paper.cli capture --date D --slot morning            # morning
python -m paper.cli capture --date D --slot pregame            # ~2h before first tip
python -m paper.cli capture --date D --slot closing            # as close to tip as possible
python -m paper.cli grade   --date D --logs logs.parquet       # next morning, once games are final
python -m paper.cli replay                                     # every persona, all dates -> data/paper/ledgers/
python -m paper.cli report  --preset realistic|sharp|ladder|everything
```

`morning` and `pregame` boards are bettable. `closing` is reference only (CLV, and
the market's final word for weighting books); no bet is ever placed at it.

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
Verified by `pytest paper/tests` (52 tests, offline): pricing, de-vig, consensus
and weight fitting (recovers a planted lead), personas, filters, normalization,
crosswalk, settlement incl. DNP/pending, CLV, point-in-time fitting, the client's
batching/retry/key handling, and the real model run on a synthetic preseason sheet.

**Not verified against live data** (blocked in the build session): the OpticOdds
NBA fetch (`optic.py`, `capture.fetch_board_rows`), the spread/total parsing in
`slate.games_from_lines` against a real payload, and the slate run on the real
preseason sheet. `discover` is the first live check. `get_fixture_results` in
`optic.py` is a guess at the results endpoint and is unused.

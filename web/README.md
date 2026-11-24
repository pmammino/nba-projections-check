# NBA paper trading: dashboard

A Next.js app that reads the data the GitHub Action writes to the `paper-data` branch.
No server, no database, no token: the repo is public and the browser fetches the JSON straight
from `raw.githubusercontent.com`.

| page | what it shows |
|---|---|
| Overview | headline ROI ± band and verdict for the Realistic and Sharp presets, CLV, every persona's P/L and bankroll, cumulative P/L chart, the latest day's best bets |
| Filtered paper trading | every candidate bet, filtered in your browser (price basis, edge, odds range, sharp confirmation, ladders, timing, markets, books), with ROI ± band, verdict, CLV, and where the result comes from |
| Calibration | is RotoWire's floor/ceiling band as wide as it claims? Read this before trusting any price |

## Where the data comes from

`python -m paper.cli auto` (every 15 minutes, in `.github/workflows/paper-trading.yml`) writes
to `paper-data`:

```
site/summary.json          personas, presets, today's bets, calibration      (read on the server, cached 5 min)
site/pool/index.json       which days exist and which are graded
site/pool/YYYY-MM-DD.json  that day's candidate bets, packed                 (read by the browser, one file per day)
```

The filter engine in `lib/filter.ts` is a line-for-line port of `paper/paper_filter.py`.
`npm test` runs both on the same 14,000-row pool across 17 filter configurations and fails if
the bet lists, their order, or any summary number differ. Regenerate the fixture after changing
the Python engine:

```bash
python paper/tests/make_web_fixture.py
cd web && npm test
```

## Deploy on Vercel

1. **Vercel → Add New → Project → Import** `pmammino/nba-projections-check`.
2. **Root Directory:** `web`  (Edit next to the path; this is the one setting that matters).
3. Framework Preset: **Next.js** (auto-detected). Build/Install/Output: leave the defaults.
4. **Environment Variables:** none are required. Optional: `NEXT_PUBLIC_DATA_BASE_URL` to read the
   data from somewhere other than this repo's `paper-data` branch. It is inlined at build time, so
   changing it needs a redeploy.
5. **Production Branch:** `main` (Settings → Git).
6. Deploy.

`web/vercel.json` turns off deployments of the `paper-data` branch. Without it, every data commit
(every 15 minutes) would try to build a branch that contains no app.

The site does **not** need redeploying when data changes: pages revalidate every 5 minutes and the
filtered view fetches fresh JSON on load. Redeploy only when the app's code changes.

## Local development

```bash
cd web && npm install && npm run dev      # http://localhost:3000
npm test                                  # filter parity with Python
```

To work against a local export instead of GitHub: `python -m paper.cli --root /some/dir export`,
serve `/some/dir/site` over HTTP with an `Access-Control-Allow-Origin: *` header, and run with
`NEXT_PUBLIC_DATA_BASE_URL=http://localhost:8765 npm run dev`.

## Notes

* `raw.githubusercontent.com` caches for about 5 minutes, so the dashboard can trail the Action by that much.
* A day with no candidate bets has no shard; the pool is empty until the first capture with games on the board.
* Closing-board rows are not in the pool: they are reference-only (never bet); their CLV is already attached.

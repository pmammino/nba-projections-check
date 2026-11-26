# NBA paper trading report

_Updated 2026-11-26 15:35Z_

## Personas

| persona | bets | settled | staked | pnl | roi |
|---|---|---|---|---|---|
| firehose | 884 | 750 | 884.0 | -11.71 | -1.3% |
| disciplined | 10 | 5 | 10.0 | +6.47 | +64.7% |
| shopper | 10 | 5 | 10.0 | +4.79 | +47.9% |
| kelly | 19 | 12 | 53.2 | -22.24 | -41.8% |
| budget | 10 | 5 | 20.0 | +9.51 | +47.6% |
| purist | 13 | 8 | 39.0 | -0.64 | -1.6% |
| sharp | 9 | 5 | 9.0 | +0.73 | +8.1% |

Personas show what a strategy would have *felt* like. Only the filter table below, with
far more bets, can say whether the simulator is real.

## Filtered paper trading

| preset | bets | settled | roi | ±2σ | verdict | CLV beat rate |
|---|---|---|---|---|---|---|
| Realistic | 788 | 667 | -3.3% | 7.9% | indistinguishable from zero | 27% (n=75) |
| Sharp books | 17 | 14 | -6.9% | 51.7% | too few bets to call | 0% (n=2) |
| Ladders | 3184 | 2668 | -10.3% | 9.1% | negative | 48% (n=192) |
| Everything | 15243 | 13698 | -5.3% | 3.8% | negative | 23% (n=738) |

## Is the feed's floor/ceiling calibrated?

Realised vs what the fitted distribution implies. Large gaps mean prices are not to be trusted
for that stat (an over-tight band prices every long-shot over as value).

| stat | n | below floor (real / exp) | above ceiling (real / exp) | inside band (real / exp) |
|---|---|---|---|---|
| ast | 161 | 4% / 8% | 16% / 18% | 80% / 74% |
| blk | 161 | 0% / 0% | 17% / 15% | 83% / 85% |
| dreb | 161 | 9% / 10% | 16% / 19% | 75% / 71% |
| fga | 161 | 7% / 18% | 21% / 22% | 71% / 60% |
| fgm | 161 | 4% / 11% | 19% / 20% | 78% / 70% |
| fta | 161 | 8% / 6% | 22% / 18% | 70% / 77% |
| ftm | 161 | 7% / 5% | 18% / 16% | 75% / 79% |
| oreb | 161 | 1% / 1% | 22% / 16% | 78% / 83% |
| pts | 161 | 2% / 14% | 22% / 23% | 76% / 63% |
| reb | 161 | 4% / 13% | 19% / 21% | 78% / 67% |
| stl | 161 | 0% / 0% | 14% / 14% | 86% / 86% |
| threes | 161 | 7% / 4% | 16% / 15% | 77% / 82% |
| threesa | 161 | 6% / 11% | 19% / 18% | 75% / 71% |
| tov | 161 | 4% / 3% | 15% / 16% | 81% / 81% |

# NBA paper trading report

_Updated 2026-11-28 13:41Z_

## Personas

| persona | bets | settled | staked | pnl | roi |
|---|---|---|---|---|---|
| firehose | 1969 | 1823 | 1969.0 | -106.99 | -5.4% |
| disciplined | 15 | 10 | 15.0 | +3.38 | +22.5% |
| shopper | 15 | 10 | 15.0 | +3.82 | +25.5% |
| kelly | 31 | 24 | 80.4 | -49.40 | -61.5% |
| budget | 15 | 10 | 30.0 | +7.80 | +26.0% |
| purist | 21 | 16 | 62.8 | -6.15 | -9.8% |
| sharp | 14 | 10 | 14.0 | +5.32 | +38.0% |

Personas show what a strategy would have *felt* like. Only the filter table below, with
far more bets, can say whether the simulator is real.

## Filtered paper trading

| preset | bets | settled | roi | ±2σ | verdict | CLV beat rate |
|---|---|---|---|---|---|---|
| Realistic | 1729 | 1599 | +0.1% | 5.0% | indistinguishable from zero | 27% (n=75) |
| Sharp books | 34 | 30 | +21.2% | 34.4% | too few bets to call | 0% (n=2) |
| Ladders | 6999 | 6448 | -20.4% | 5.4% | negative | 48% (n=192) |
| Everything | 32773 | 31098 | -15.3% | 2.3% | negative | 23% (n=738) |

## Is the feed's floor/ceiling calibrated?

Realised vs what the fitted distribution implies. Large gaps mean prices are not to be trusted
for that stat (an over-tight band prices every long-shot over as value).

| stat | n | below floor (real / exp) | above ceiling (real / exp) | inside band (real / exp) |
|---|---|---|---|---|
| ast | 384 | 6% / 7% | 17% / 18% | 77% / 75% |
| blk | 384 | 0% / 0% | 15% / 15% | 85% / 85% |
| dreb | 384 | 7% / 10% | 16% / 19% | 76% / 71% |
| fga | 384 | 5% / 17% | 17% / 22% | 78% / 61% |
| fgm | 384 | 4% / 11% | 16% / 20% | 80% / 69% |
| fta | 384 | 11% / 6% | 18% / 18% | 71% / 76% |
| ftm | 384 | 8% / 5% | 18% / 17% | 74% / 78% |
| oreb | 384 | 1% / 1% | 21% / 15% | 78% / 83% |
| pts | 384 | 3% / 14% | 19% / 23% | 78% / 63% |
| reb | 384 | 4% / 12% | 18% / 21% | 77% / 67% |
| stl | 384 | 0% / 0% | 15% / 15% | 84% / 85% |
| threes | 384 | 6% / 4% | 14% / 15% | 80% / 82% |
| threesa | 384 | 5% / 11% | 19% / 18% | 75% / 72% |
| tov | 384 | 3% / 3% | 17% / 16% | 80% / 81% |

# Data Analysis Findings

## 1. Split (train=Train_CF_7, val=Test_CF_7 day8, test=Test_CF_8 day9)

- train:  254750 (70.3%)
- val:     55478 (15.3%)
- test:    52172 (14.4%)

## 2. Per-day row counts (held-out Test files, day = fold+1)

- day  2:   38397 rows
- day  3:   28535 rows
- day  4:   37023 rows
- day  5:   34785 rows
- day  6:   39152 rows
- day  7:   37346 rows
- day  8:   55478 rows
- day  9:   52172 rows
- day 10:   31937 rows

## 3. Cross-day drift (L2 between per-day mean vectors)

- mean pairwise drift across held-out days: 0.0605
- min / max pairwise drift: 0.0089 / 0.1581
- see figures/cross_day_drift.png

## 4. Adjacent vs random row distance (within one day)

- mean L2 between consecutive rows: 0.04374
- mean L2 between random rows:      0.61661
- ratio adjacent/random: 0.071 (<<1 => a random split would leak near-duplicates -> use temporal)

## 5. Per-feature ranges (train, after DecPre)

- global min=0.0000  max=0.6000
- all non-negative => min-max to [-1,1] + tanh output is appropriate
- see figures/feature_ranges.png

## 6. Cross-level monotonicity in DecPre space (real train data)

- asks ascending across levels: 100.000%
- bids descending across levels: 100.000%
- positive spread: 100.000%
- fully valid books: 100.000%

_~100% => the layout assumption holds and the validity penalty/metrics are meaningful directly in this space._

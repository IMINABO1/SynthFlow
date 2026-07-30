# Problems Encountered

Substantive problems worth remembering — conceptual, data, and design problems and how we
resolved them. For each solved one: what we did, **why**, and **why not the alternatives**.
(Operational/tooling friction from automation is deliberately kept out of this log.)

---

## 1. Didn't read the synopsis well before coding — CF files are cumulative
- **Problem:** While writing `load_all_days`, didn't realize the `CF_{day}` files are **cumulative** (each day's file contains all previous days as a prefix).
- **Impact:** Naively concatenating all 9 files would duplicate early days up to 9×.
- **Verified on real data:** CF_1 = 39,512 samples, CF_2 = 77,909, and CF_2's leading values match CF_1 exactly → the prefix property holds.
- **Resolution:** `load_all_days()` == `load_day(9)` (CF_9 already holds every day de-duplicated). To recover *per-day* rows for the split, `load_day_incremental(k) = CF_k[len(CF_{k-1}):]`.
- **Why this over alternatives:** CF_9 alone gives the full corpus with zero bookkeeping; per-day recovery is only needed for the train/val/test split, and the prefix property makes it a clean slice.
- **Status:** ✅ Resolved / verified.

## 3. The cumulative files are cumulative PER-STOCK, not globally — so day-slicing was wrong
- **Problem:** I assumed `CF_k` = `[all day1 | all day2 | ...]`, so day k = `CF_k[len(CF_{k-1}):]`. The EDA's full-array prefix check failed (`prefix_ok=False` for days 2–9), contradicting the assumption.
- **Investigation:** feature 0 of `CF_1` and `CF_2` matches for exactly the first **3,454** rows (the number my own `test.py` had probed as `cf2[:3455]`), then diverges — `CF_1` drops to a new price level while `CF_2` continues. FI-2010 concatenates **5 stocks**, and each stock has its *own* expanding window: `CF_2 = [stock1: d1+d2][stock2: d1+d2]…`, not `[all d1][all d2]`. So 3,454 = stock-1's day-1 length, and prefix-slicing interleaves stocks instead of isolating a day.
- **Resolution:** Use the benchmark's **official `Testing/` files** for held-out days. The anchored protocol is `Train_CF_k` = cumulative days 1..k (fine for *pooled* training — per-stock order doesn't matter when pooling), and `Test_CF_k` = day k+1 as a **separate self-contained file**. Corrected split: train=`Train_CF_7` (days 1–7), val=`Test_CF_7` (day 8), test=`Test_CF_8` (day 9) → same ~70/15/15, but cleanly disjoint with no interleaving and no adjacency leakage.
- **Why this over alternatives:** Reverse-engineering per-stock/day boundaries from the training file is fragile (stock lengths vary: 3,454 for stock 1's day 1, not an even split). The official Test files are the intended, publishable held-out sets and need no guessing. Confirmed they load in the same 149-row format and are 100% valid in DecPre space.
- **Caught by:** running the EDA on real data instead of trusting the assumption — the whole point of the "verify, don't inshallah" approach.
- **Status:** ✅ Resolved (loader `load_split` rewritten to use Train + Test files).

## 2. WGAN doesn't need a per-day split for *training* — but evaluation does
- **Problem:** Started splitting the dataset by day for training an unconditional snapshot model, where day boundaries don't affect the training distribution.
- **Resolution:** Pool snapshots for training, BUT keep **held-out days** for evaluation, to tell *learning the distribution* apart from *memorizing* it.
- **Why days, not random rows:** adjacent LOB snapshots are near-duplicates, so a random split leaks near-copies into validation and flatters the model. Whole-day holdout is the honest generalization test (markets are regime-y). Empirically the temporal split lands at ~70/15/15 (days 1–7 / 8 / 9) with no tuning, and days 8–9 are larger/more active — a meaningful holdout.
- **Status:** ✅ Resolved.

# SynthFlow — Working Journal

A running log of progress, open questions, and decisions (with rationale). Newest entries at the bottom of each section.

---

## Project in one line
Structure-aware generative model (WGAN-GP) for synthetic limit-order-book snapshots, trained on FI-2010.

---

## Decisions (with rationale)

### 2026-07-30 — Design decisions locked in
- **Generate single 40-dim snapshots (i.i.d.), MLP generator + critic.**
  - Why: simplest correct GAN to build/understand/debug; matches "no need to split by day" for an unconditional model. Sequences (the "Flow" dynamics) become v2 once the baseline works.
  - Rejected: sequence/window generation (conv/recurrent) — harder to train, more failure modes, poor first project.
- **Order-book validity via soft penalty, λ=0 by default.**
  - Why: with λ=0 it's a *pure* WGAN-GP baseline; the penalty path + validity metric already exist so we can turn λ up and literally watch invalid books drop. Best for learning.
  - Rejected: hard constraint (cumulative-softplus) — guarantees validity but bakes in assumptions and is fiddlier for a first GAN. Rejected "none at all" — we still want validity *measured*.
- **Scope: model + training loop + training-time metrics.** Standalone eval module + diffusion comparison deferred to v2.
- **Data split decided empirically first** via a `data_analysis/` phase (user: "not inshallah it"). Leaning temporal train=days 1–7 / val=day 8 / test=day 9.
  - Why: markets are "regime-y" (Jane Street) — holding out whole days is the honest generalization test. Random splitting leaks near-duplicate adjacent rows. But we verify with real numbers before committing.

### 2026-07-30 — Framing against HRT AI Labs
HRT's 2025 intern project generated synthetic **returns** with **diffusion** after **GANs/transformers mode-collapsed**, and judged success by whether financial properties survived (fat tails, volatility, autocorrelation) — not by loss curves. Consequences for us:
- Treat WGAN-GP-vs-collapse as a **measured** question, never a claim. Log diversity / nearest-neighbor distance during training.
- Keep a held-out split to detect memorization.
- Log HRT-style financial metrics from day one.

---

## Open questions
- Which day-split is actually best? → to be answered by `data_analysis/` once the dataset is present.
- Does DecPre per-column scaling preserve cross-level price monotonicity? → EDA question 6; determines whether the validity penalty can run in normalized space.
- Confirm the 40-feature column layout empirically (assumed FI-2010 standard: per level i, `[P_ask_i, V_ask_i, P_bid_i, V_bid_i]`).

---

## Progress log

### 2026-07-30
- Explored repo, read HRT post, locked design decisions (above).
- Wrote plan. Set up `journal.md` + restructured `problems_encountered.MD`.
- Dataset (FI-2010, ~1.86 GB) downloaded and extracted to `dataset/BenchmarkDatasets/` (git-ignored, 30 GB expanded).
- **Verified on real data (not "inshallah"):**
  - Files are 149 feature-rows × N sample-columns; `load_day` transposes to `[N, 40]`. ✓
  - Cumulative structure holds: CF_1=39,512, CF_2=77,909; CF_2 leading values == CF_1. ✓
  - Per-day sample counts: 1:39,512 · 2:38,397 · 3:28,535 · 4:37,023 · 5:34,785 · 6:39,152 · 7:37,346 · 8:55,478 · 9:52,172 (total 362,400).
  - **Split decided from evidence:** temporal 1–7 / 8 / 9 = 254,750 / 55,478 / 52,172 ≈ **70/15/15**, no tuning. Days 8–9 larger → possible regime shift → meaningful holdout.
- **Decision — loader caching:** `np.loadtxt` on the 864 MB CF_9 is very slow, so the loader caches each parsed array to `.npy` on first read (instant thereafter).
  - Rejected: re-parsing text every run (wastes minutes); pandas `read_csv` (still re-parses each run, extra dep). `.npy` cache is simplest and fastest for repeated loads.
- **Note — path:** default `data_path` is `dataset/BenchmarkDatasets/` (data lives inside the repo, git-ignored), not `../`.
- **Note — case quirk:** ZScore subdir is spelled `1.NoAuction_Zscore` (lowercase s) on disk vs my `_ZScore`. Irrelevant for DecPre (default); fix if we ever use ZScore.
- Built loader (with `.npy` caching), `lob_layout.py`, `models/wgan.py` (Generator/Critic/gradient_penalty/lob_violation_penalty), `metrics.py` (validity/spread/depth/imbalance/nn_distance/diversity), `train.py` (WGAN-GP loop, MinMax scaler, wandb, checkpoints), `data_analysis/eda.py`. All files pass `py_compile`.
- **Env / GPU finding (important):** machine has only Python 3.14 (no numpy/torch, no venv). GPU is **RTX 5070 Laptop = Blackwell (sm_120)**. The pinned `torch==2.6.0+cu124` is wrong twice over: (a) no torch wheels for 3.14, (b) cu124 has no Blackwell kernels → runtime failure on the 5070. **Fix:** target **Python 3.12 + torch 2.7.1+cu128** (CUDA 12.8). Updated `requirements.txt` accordingly (`--extra-index-url .../cu128`, torch 2.7.1+cu128, torchvision 0.22.1+cu128).
- Runtime unit-checks + EDA + training are blocked only on creating a compatible env (needs a Python 3.12 interpreter; none installable via uv/conda/pyenv — none present).
- **Open idea (user asked):** where C++ could fit — fast LOB file parser (pybind11) for the np.loadtxt bottleneck; and a C++ limit-order-book matching engine for the v2 sequence/order-flow phase. Existing g++ is MinGW.org 6.3.0 (2017, 32-bit, pre-C++17, broken) — unusable; v2 C++ needs a modern 64-bit toolchain (MSYS2 UCRT64 or MSVC Build Tools).

### 2026-07-30 (later) — environment working, pipeline verified end-to-end
- Set up env with `uv`: Python 3.12.13 venv + torch **2.7.1+cu128**. `torch.cuda.is_available()=True`, GPU = RTX 5070, capability (12,0)=Blackwell sm_120. cu128 kernels run correctly.
- Slimmed `requirements.txt` to top-level packages (old pins were a torch-2.6 freeze; `sympy==1.13.1` etc. conflicted with torch 2.7). Install needs `--index-strategy unsafe-best-match` (documented in README).
- `scripts/unit_check.py`: **12/12 PASS** on GPU (model shapes, gradient penalty, validity penalty on valid vs crossed books, metrics, scaler round-trip).
- `train.py --smoke`: full WGAN-GP loop ran on real day-1 data, logged metrics, saved `checkpoints/{ckpt_e1,ckpt_final}.pt` + `scaler.npz`. `.npy` cache works: cached `load_day(1)` = 0.00s.
- Wrote `README.md` (setup + usage). gitignored `dataset/`, `checkpoints/`, `wandb/`, figures.
- Next: run full `data_analysis/eda.py` (parses+caches all 9 days) to confirm the split evidence, per-feature ranges, and — key — whether DecPre space preserves cross-level monotonicity (decides if the validity penalty/metrics are meaningful in that space).

### 2026-07-30 (later still) — EDA caught a real bug: split was wrong
- **Finding:** EDA's full-array prefix check failed (`prefix_ok=False`). Investigation showed the `CF_k` training files are cumulative **per stock**, not globally: FI-2010 concatenates 5 stocks, each with its own expanding window. Feature 0 of CF_1/CF_2 matches for exactly 3,454 rows (stock-1's day-1 length; my own `test.py` had probed `cf2[:3455]`) then diverges. So `CF_k[len(CF_{k-1}):]` interleaves stocks instead of isolating a day → `load_split` was wrong. (Details in problems_encountered #3.)
- **Fix:** rewrote `loader` to use the benchmark's official files: `Train_CF_k` = pooled days 1..k for training; `Test_CF_k` = clean held-out day k+1. New split: train=`Train_CF_7` (days 1–7, 254,750) / val=`Test_CF_7` (day 8, 55,478) / test=`Test_CF_8` (day 9, 52,172) — same 70/15/15, cleanly disjoint. Dropped `load_day_incremental`/`verify_cumulative`; added `load_file(day, dataset)` + Testing support.
- **Confirmed:** Test files load in the same 149-row format and are 100% valid in DecPre space.
- **Design assumption verified:** cross-level monotonicity is 100% in DecPre space (train pool + a test day) → validity penalty & metrics are meaningful directly in DecPre space; no need to unscale for λ>0.
- **Other EDA results (valid):** per-feature range [0, 0.6], all non-negative → min-max→[-1,1] + tanh output is right. Adjacent/random row-distance ratio ≈ 0.075 → strong empirical case for a temporal (not random) split.
- Rewrote `data_analysis/eda.py` to use Test files as clean per-day sets (per-day drift is now correct); re-running to regenerate FINDINGS.md.
- **Lesson:** verifying on real data (not "inshallah") caught a data-integrity bug *before* any real training run — exactly the intended payoff.

### 2026-07-30 — first full baseline result (100 epochs, λ_valid=0, pure WGAN-GP)
Ran 100 epochs on the corrected split (paused ~4h mid-run because the laptop slept — see [[laptop-sleep-pauses-training]]). Evaluated `ckpt_final.pt` vs real held-out day 8 (`scripts/report.py`, n=8192):
- **Validity (generated):** asks_monotonic 0.012%, bids_monotonic 0.037%, positive_spread 49.8%, **valid_all 0.000%** (real = 100%).
- **Distributions (fake vs real):** depth 0.391(0.281) vs 0.380(0.264) ✓ close; imbalance −0.058(0.225) vs −0.017(0.200) ~ok; spread −0.0001(0.0014) vs +0.0004(0.0003) ✗ (fake spread centered ~0, too wide → explains ~50% positive-spread).
- **Diversity/memorization:** nn_distance(fake→real)=0.069 (no memorization), pairwise diversity fake 0.537 vs real 0.563 → **no mode collapse**.

**Interpretation:** the pure baseline learns marginal *magnitudes* (depth, roughly imbalance) and, per HRT's worry, does **not** mode-collapse (measured, not claimed). But it fails the hard cross-level **ordering** constraints — 0% fully-valid books — because a vanilla MLP matching marginals won't jointly satisfy 20 monotonicity constraints + positive spread. This directly motivates the next experiments: turn on `--lambda_valid` (soft penalty) and/or hard cumulative-softplus construction; and improve spread modeling.

## Open questions (next phase)
- How much does the soft validity penalty (`--lambda_valid`) raise validity, and at what cost to distribution fidelity / diversity (collapse)? Need a baseline-vs-penalty comparison.
- Is spread the weakest marginal because the model has no incentive to keep best-ask > best-bid tightly? Does the penalty fix it?
- If soft penalty plateaus below ~100% valid, do we need hard constraints (cumulative-softplus construction)?

### 2026-07-30 — decision: proceed to validity-penalty phase (planning)
Baseline (λ=0) established: no collapse, decent marginals, 0% valid. Next phase = drive validity up while preserving no-collapse + distribution fidelity, and produce a clean baseline-vs-penalty comparison. Entering /plan to design it.

### 2026-07-30 — Phase-2 results (model {base,wide} × λ {0,0.1,1,10}, 100 epochs each)
Full table in `data_analysis/experiment_comparison.md` (held-out day 8, n=8192). Highlights:

| run | valid_all | asks_mono | spread+ | depth_err | diversity (real 0.56) |
|---|---|---|---|---|---|
| base λ0 | 0.0% | 0.01% | 0.49 | 0.0136 | 0.528 |
| base λ10 | **4.0%** | 16.4% | 0.74 | 0.0239 | 0.540 |
| wide λ0 | 0.0% | 0.07% | 0.84 | **0.0042** | 0.535 |
| wide λ10 | 0.7% | 3.1% | 0.90 | 0.0161 | 0.543 |

**Findings:**
1. **Soft penalty helps only weakly.** Full validity moves from 0% → at best **4%** (base λ=10); component monotonicity improves more (asks 0.01%→16% at base λ=10) but satisfying all ~20 ordering constraints *jointly* stays rare. λ∈{0.1,1} barely move full validity. ⇒ the soft penalty at these strengths is **insufficient**; need much larger λ / a λ-schedule, or **hard constraints** (cumulative-softplus), which is the planned contingency.
2. **No mode collapse or memorization anywhere.** Diversity 0.53–0.54 vs real 0.56, nn-dist ~0.07 across *all* 8 runs — the penalty doesn't destabilize training (measured, per HRT framing).
3. **Wide model** modestly improves depth fidelity (wide λ0 depth_err 0.0042, best overall) but does **not** fix validity — capacity isn't the bottleneck; the missing inductive bias for ordering is.
4. Spread (`spread+`) is noisy/non-monotonic in λ — no clean win; spread remains the weakest marginal.

**Conclusion / next phase (not started):** structural validity needs either a hard constraint (generator emits sorted prices by construction) or a much stronger/scheduled penalty. Capacity and collapse are not the limiting factors. Recommend hard cumulative-softplus construction as the next experiment.

### 2026-07-30 — Phase-3a: constrained (gap-space) generator WORKS
Built `ConstrainedGenerator` (models/constrained.py): base price + cumulative-softplus gaps ⇒ asks↑, bids↓, spread>0, volumes≥0, all differentiable. Wired into training via `GlobalScaler` (shared price/vol scale, order-preserving). Unit test: **untrained** constrained gen is already 100% valid.

100-epoch validation run (`constrained_v1`) vs baseline, held-out day 8:

| metric | baseline mlp λ0 | constrained | real |
|---|---|---|---|
| valid_all | 0.0% | **97.1%** | 100% |
| asks/bids monotonic | ~0% | **100% / 100%** | 100% |
| spread (mean,std) | −0.0001, 0.0014 | **+0.0003, 0.0007** | +0.0004, 0.0002 |
| diversity | 0.528 | 0.538 | 0.564 |
| nn-dist | 0.070 | 0.067 | — |

**Findings:** validity jumped 0% → 97% (vs Phase-2 best of 4%), spread fidelity improved a lot (the parametrization *is* a spread), no collapse, no memorization — construction gives validity *and* better fidelity. The 97% (not 100%) miss was purely `positive_spread`: tiny learned spreads underflow to 0 in float32. Fixed with a `min_spread=1e-4` floor (real books always have ≥1 tick) → strictly positive by construction.

Next: Phase-3b rigor (multi-seed × multi-day) to report mean±std.

### 2026-07-31 — Phase-3b: rigorous comparison (3 seeds × held-out days 8/9/10)
Ran {constrained, mlp λ0, mlp λ10} × seeds {0,1,2}, 100 epochs each; each evaluated on days
8/9/10 and aggregated to mean±std (`data_analysis/experiment_comparison_phase3.md`). No sleep
this run (power settings held).

| config | valid_all | spread_err | depth_err | nn_dist | diversity (real ~0.56) |
|---|---|---|---|---|---|
| **constrained** | **1.0000 ± 0.0000** | **0.0001 ± 0.0000** | 0.0165 ± 0.0089 | 0.0788 ± 0.0118 | 0.539 ± 0.004 |
| mlp λ0 | 0.0001 ± 0.0001 | 0.0003 ± 0.0001 | 0.0160 ± 0.0082 | 0.0789 ± 0.0110 | 0.540 ± 0.004 |
| mlp λ10 | 0.0467 ± 0.0194 | 0.0002 ± 0.0001 | 0.0131 ± 0.0053 | 0.0782 ± 0.0109 | 0.538 ± 0.003 |

**Defensible conclusions:**
1. Constrained = **100.00% ± 0.00% valid** — perfect, zero variance across all seeds and held-out
   days. vs soft-penalty 4.7% ± 1.9% and baseline ~0%. The min_spread floor gives exactly 100%.
2. **No trade-off:** diversity and nn-dist are statistically identical across all three configs
   (overlapping error bars) → validity came free, no collapse or memorization cost.
3. Constrained also has the tightest spread fidelity (0.0001 ± 0.0000); depth comparable across all.

**Phase-3 verdict:** hard construction (gap-space) is the answer to structural validity — it beats
penalty-tuning and capacity, with no fidelity cost, and the result is robust (multi-seed × multi-day).
The gap-space representation is reusable for a future diffusion model (the HRT comparison).

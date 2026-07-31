# SynthFlow

Structure-aware **generative model for synthetic limit-order-book (LOB) data**.
A WGAN-GP learns the distribution of FI-2010 order-book snapshots and synthesizes
new, plausible books. Framed against HRT AI Labs' synthetic-market-data work: we
**measure** mode collapse and financial-property fidelity rather than trusting loss
curves.

## Results (the arc)
1. **Baseline WGAN-GP** generates diverse order books but **0% are structurally valid**
   (prices out of order) — and it does *not* mode-collapse.
2. **Soft validity penalty & bigger models don't fix it** — best was 4.7% ± 1.9% valid.
   Validity can't be reliably *learned* from a penalty.
3. **Gap-space construction fixes it** — emit a base price + cumulative-softplus gaps ⇒
   **100.00% ± 0.00% valid** by construction, robust across seeds and held-out days, with
   *better* spread fidelity and no diversity/memorization cost.
4. **GAN vs Diffusion (capstone)** — in the same gap-space, WGAN-GP and a DDPM are
   **near-equivalent**: both need the construction to be valid (both ~0% free, 100% constructed),
   both diverse, similar fidelity. **The inductive bias dominates the paradigm choice.** Notably,
   the GAN did not collapse — HRT's GAN-collapse concern did not reproduce here (WGAN-GP + GP).

Full numbers with mean±std in `data_analysis/` (`experiment_comparison*.md`, `comparison_final.md`)
and the reasoning trail in `journal.md`.

## Deferred future work
Temporal/sequence generation (order-flow), TSTR downstream-utility, and a C++ matching engine.

## Layout
```
data/loader.py         FI-2010 loader (cumulative-file aware, .npy cache, temporal split)
lob_layout.py          column layout of the 40 LOB features
gap_transform.py       exact book<->gap-space bijection (valid by construction)
models/wgan.py         WGAN-GP Generator, Critic, gradient_penalty, validity penalty
models/constrained.py  gap-space GAN generator (valid by construction)
models/diffusion.py    DDPM (denoiser + Gaussian diffusion)
metrics.py             validity / spread / depth / imbalance / diversity metrics
train.py               WGAN-GP training loop (scaler, wandb, checkpoints)
train_diffusion.py     DDPM training loop (free / constrained modes)
data_analysis/eda.py   evidence for the split + design choices (writes FINDINGS.md)
scripts/               unit_check, report, compare, run_phase{3,4}, compare_final
journal.md             running log of decisions + progress
problems_encountered.MD  substantive problems + how they were resolved
```

## Dataset
FI-2010 (open-access, etsin/Fairdata `73eb48d7-…`). Download `BenchmarkDatasets.zip`
(~1.86 GB) and extract so the files live at:
```
dataset/BenchmarkDatasets/NoAuction/3.NoAuction_DecPre/NoAuction_DecPre_Training/Train_Dst_NoAuction_DecPre_CF_{1..9}.txt
```
The `CF_k` files are **cumulative** (`CF_k` contains `CF_{k-1}` as a prefix); the
loader recovers per-day rows and builds a temporal split (train days 1–7 / val 8 /
test 9 ≈ 70/15/15).

## Setup (Windows, RTX 5070 / Blackwell → CUDA 12.8)
```bash
pip install uv
uv python install 3.12
uv venv --python 3.12
# NOTE: --index-strategy unsafe-best-match is required so pip pulls torch from the
# cu128 index while resolving the other pins from PyPI.
uv pip install -r requirements.txt --index-strategy unsafe-best-match
```
> The RTX 5070 is Blackwell (`sm_120`); it needs `torch>=2.7` built for `cu128`.
> The older `cu124` build has no Blackwell kernels and fails at runtime.

## Run
```bash
.venv/Scripts/python -m scripts.unit_check     # smoke checks (no dataset needed)
.venv/Scripts/python -m data_analysis.eda      # EDA -> figures + FINDINGS.md
.venv/Scripts/python train.py --smoke          # tiny end-to-end training run
.venv/Scripts/python train.py                  # full WGAN-GP training
.venv/Scripts/python train.py --lambda_valid 1.0   # add the validity penalty
```

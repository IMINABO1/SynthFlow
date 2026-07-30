# SynthFlow

Structure-aware **generative model for synthetic limit-order-book (LOB) data**.
A WGAN-GP learns the distribution of FI-2010 order-book snapshots and synthesizes
new, plausible books. Framed against HRT AI Labs' synthetic-market-data work: we
**measure** mode collapse and financial-property fidelity rather than trusting loss
curves.

## Status
- **v1 (in progress):** unconditional WGAN-GP over single 40-dim snapshots (MLP
  generator + critic), soft order-book-validity penalty (λ=0 by default → pure
  baseline), training-time HRT-style metrics.
- **v2 (planned):** temporal/sequence generation, a diffusion baseline for
  comparison, a standalone evaluation report, and a C++ limit-order-book matching
  engine for order-flow simulation.

## Layout
```
data/loader.py        FI-2010 loader (cumulative-file aware, .npy cache, temporal split)
lob_layout.py         column layout of the 40 LOB features
models/wgan.py        Generator, Critic, gradient_penalty, lob_violation_penalty
metrics.py            validity / spread / depth / imbalance / diversity metrics
train.py              WGAN-GP training loop (scaler, wandb, checkpoints)
data_analysis/eda.py  evidence for the split + design choices (writes FINDINGS.md)
scripts/unit_check.py data-independent smoke checks
journal.md            running log of decisions + progress
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

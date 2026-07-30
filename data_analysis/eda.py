"""Exploratory data analysis to justify design choices from evidence, not guesses.

Writes figures to data_analysis/figures/ and a data_analysis/FINDINGS.md.

Uses the benchmark's own files correctly (see problems_encountered.MD #3):
  - Train_CF_k  = cumulative days 1..k, concatenated PER STOCK (pooled training).
  - Test_CF_k   = a self-contained held-out day (day k+1).
So per-day analysis uses the Test files (each is one clean day), NOT prefix
slices of the training file.

Questions answered:
  1. split sizes (train=Train_CF_7, val=Test_CF_7, test=Test_CF_8)
  2. per-day row counts (Test_CF_1..9 == days 2..10)
  3. cross-day distribution drift (is the temporal holdout meaningful?)
  4. adjacent-row similarity within a day (would a random split leak?)
  5. per-feature value ranges after DecPre (generator output activation)
  6. cross-level monotonicity in DecPre space (can the validity penalty run here?)

Run:  python -m data_analysis.eda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.loader import DataSetLoader, DEFAULT_DATA_PATH
import metrics as M

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "data_analysis" / "figures"


def _sub(n: int, k: int) -> np.ndarray:
    return np.arange(n) if n <= k else np.linspace(0, n - 1, k).astype(int)


def run(data_path: str, normalization: str, sample: int) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ld = DataSetLoader(data_path, normalization=normalization)
    out: list[str] = ["# Data Analysis Findings", ""]

    # --- 1: split sizes --------------------------------------------------- #
    train, val, test = ld.load_split(train_fold=7)
    tot = len(train) + len(val) + len(test)
    out += ["## 1. Split (train=Train_CF_7, val=Test_CF_7 day8, test=Test_CF_8 day9)", "",
            f"- train: {len(train):>7d} ({len(train)/tot:.1%})",
            f"- val:   {len(val):>7d} ({len(val)/tot:.1%})",
            f"- test:  {len(test):>7d} ({len(test)/tot:.1%})", ""]

    # --- 2: per-day counts (Test_CF_1..9 = days 2..10) -------------------- #
    days = {k + 1: ld.load_file(k, dataset="Testing") for k in range(1, 10)}  # day 2..10
    out += ["## 2. Per-day row counts (held-out Test files, day = fold+1)", ""]
    for d in sorted(days):
        out.append(f"- day {d:>2d}: {len(days[d]):>7d} rows")
    out.append("")

    # --- 3: cross-day drift (each Test file is a clean day) --------------- #
    dkeys = sorted(days)
    means = np.stack([days[d][_sub(len(days[d]), sample)].mean(0) for d in dkeys])
    drift = np.linalg.norm(means[:, None, :] - means[None, :, :], axis=2)
    plt.figure(figsize=(5.2, 4.2))
    plt.imshow(drift, cmap="viridis")
    plt.colorbar(label="L2 between per-day mean vectors")
    plt.xticks(range(len(dkeys)), dkeys); plt.yticks(range(len(dkeys)), dkeys)
    plt.xlabel("day"); plt.ylabel("day"); plt.title("Cross-day distribution drift")
    plt.tight_layout(); plt.savefig(FIG_DIR / "cross_day_drift.png", dpi=120); plt.close()
    iu = np.triu_indices(len(dkeys), 1)
    out += ["## 3. Cross-day drift (L2 between per-day mean vectors)", "",
            f"- mean pairwise drift across held-out days: {drift[iu].mean():.4f}",
            f"- min / max pairwise drift: {drift[iu].min():.4f} / {drift[iu].max():.4f}",
            "- see figures/cross_day_drift.png", ""]

    # --- 4: adjacent vs random within one clean day ----------------------- #
    d = days[dkeys[0]]
    k = min(sample, len(d) - 1)
    adj = np.linalg.norm(d[1:k + 1] - d[:k], axis=1).mean()
    idx = _sub(len(d), k)
    rnd = np.linalg.norm(d[idx] - d[idx[::-1]], axis=1).mean()
    out += ["## 4. Adjacent vs random row distance (within one day)", "",
            f"- mean L2 between consecutive rows: {adj:.5f}",
            f"- mean L2 between random rows:      {rnd:.5f}",
            f"- ratio adjacent/random: {adj/(rnd+1e-9):.3f} "
            "(<<1 => a random split would leak near-duplicates -> use temporal)", ""]

    # --- 5: per-feature ranges (train) ------------------------------------ #
    ts = train[_sub(len(train), sample)]
    fmin, fmax = ts.min(0), ts.max(0)
    plt.figure(figsize=(9, 3))
    plt.plot(fmin, label="min"); plt.plot(fmax, label="max")
    plt.xlabel("feature index (0-39)"); plt.ylabel("value"); plt.legend()
    plt.title(f"Per-feature value range after {normalization}")
    plt.tight_layout(); plt.savefig(FIG_DIR / "feature_ranges.png", dpi=120); plt.close()
    out += [f"## 5. Per-feature ranges (train, after {normalization})", "",
            f"- global min={ts.min():.4f}  max={ts.max():.4f}",
            "- all non-negative => min-max to [-1,1] + tanh output is appropriate",
            "- see figures/feature_ranges.png", ""]

    # --- 6: monotonicity in normalized space ------------------------------ #
    vs = M.validity_stats(ts)
    out += [f"## 6. Cross-level monotonicity in {normalization} space (real train data)", "",
            f"- asks ascending across levels: {vs['valid/asks_monotonic']:.3%}",
            f"- bids descending across levels: {vs['valid/bids_monotonic']:.3%}",
            f"- positive spread: {vs['valid/positive_spread']:.3%}",
            f"- fully valid books: {vs['valid/all']:.3%}",
            "",
            "_~100% => the layout assumption holds and the validity penalty/metrics "
            "are meaningful directly in this space._", ""]

    (REPO / "data_analysis" / "FINDINGS.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nWrote figures to {FIG_DIR} and FINDINGS.md")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    p.add_argument("--normalization", default="DecPre")
    p.add_argument("--sample", type=int, default=20000)
    a = p.parse_args()
    run(a.data_path, a.normalization, a.sample)


if __name__ == "__main__":
    main()

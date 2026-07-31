"""Aggregate Phase-3b runs into a mean+/-std comparison across seeds and days.

For each config (constrained / mlp_l0 / mlp_l10) it evaluates every seed on each
held-out day (Test_CF_7/8/9 = days 8/9/10) and reports mean +/- std of the key
metrics. Writes data_analysis/experiment_comparison_phase3.md.

Run:  .venv/Scripts/python -m scripts.compare_phase3
"""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from data.loader import DataSetLoader
from scripts.compare import eval_run

REPO = Path(__file__).resolve().parents[1]
FOLDS = [7, 8, 9]  # Test_CF_k = held-out day k+1  -> days 8, 9, 10
METRICS = ["valid_all", "spread_err", "depth_err", "nn_dist", "diversity"]


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = 8192
    reals = {f: DataSetLoader().load_file(f, dataset="Testing") for f in FOLDS}

    run_dirs = sorted(p.parent for p in REPO.glob("checkpoints/p3_*/ckpt_final.pt"))
    groups = defaultdict(list)
    for rd in run_dirs:
        groups[re.sub(r"_s\d+$", "", rd.name)].append(rd)
    if not groups:
        print("no p3_* runs found")
        return

    rows = []
    for cfg_name, dirs in sorted(groups.items()):
        acc = defaultdict(list)
        for rd in dirs:
            for f in FOLDS:
                pool = reals[f]
                real = pool[np.random.choice(len(pool), min(n, len(pool)), replace=False)]
                r = eval_run(rd, real, n, dev)
                for k in METRICS:
                    acc[k].append(r[k])
            print(f"evaluated {rd.name}", flush=True)
        rows.append((cfg_name, len(dirs), {k: (np.mean(acc[k]), np.std(acc[k])) for k in METRICS}))

    cols = ["config", "seeds"] + METRICS
    lines = ["# Phase-3b comparison (mean +/- std over 3 seeds x held-out days 8/9/10)", "",
             "| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for cfg_name, nseeds, stats in rows:
        cells = [cfg_name, str(nseeds)]
        for k in METRICS:
            m, s = stats[k]
            cells.append(f"{m:.4f} +/- {s:.4f}")
        lines.append("| " + " | ".join(cells) + " |")

    text = "\n".join(lines)
    (REPO / "data_analysis" / "experiment_comparison_phase3.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()

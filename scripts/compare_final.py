"""Final 2x2 comparison: {GAN, Diffusion} x {free, constructed}.

GAN rows reuse Phase-3b runs (p3_mlp_l0_*, p3_constrained_*); diffusion rows use
the Phase-4 runs (p4_diff_*). Each config is evaluated over its seeds x held-out
days 8/9/10 and reported as mean+/-std. Writes data_analysis/comparison_final.md.

Run:  .venv/Scripts/python -m scripts.compare_final
"""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from data.loader import DataSetLoader
from train import load_scaler
from models.diffusion import Denoiser, GaussianDiffusion
import gap_transform as GT
import metrics as M
from scripts.compare import eval_run  # GAN eval

REPO = Path(__file__).resolve().parents[1]
FOLDS = [7, 8, 9]  # held-out days 8, 9, 10
KEYS = ["valid_all", "spread_err", "depth_err", "nn_dist", "diversity"]

# glob prefix -> (paradigm, mode)
CONFIGS = {
    "p3_mlp_l0": ("GAN", "free"),
    "p3_constrained": ("GAN", "constructed"),
    "p4_diff_free": ("Diffusion", "free"),
    "p4_diff_constrained": ("Diffusion", "constructed"),
}


def eval_diffusion(run_dir: Path, real: np.ndarray, n: int, dev: str) -> dict:
    ck = torch.load(run_dir / "ckpt_final.pt", map_location=dev, weights_only=False)
    cfg = ck["config"]
    scaler = load_scaler(run_dir / "scaler.npz")
    diff = GaussianDiffusion(Denoiser(dim=40), timesteps=cfg["timesteps"]).to(dev)
    diff.load_state_dict(ck["diffusion"])
    diff.eval()
    mean = torch.as_tensor(ck["std_mean"], device=dev)
    std = torch.as_tensor(ck["std_std"], device=dev)
    x = diff.sample(n, dim=40, device=dev) * std + mean
    book = x if cfg["diffusion_mode"] == "free" else GT.params_to_book(x)
    fake = scaler.inverse(book).detach().cpu().numpy()
    vs = M.validity_stats(fake)
    return {
        "valid_all": vs["valid/all"],
        "spread_err": abs(M.spread(fake).mean() - M.spread(real).mean()),
        "depth_err": abs(M.total_depth(fake).mean() - M.total_depth(real).mean()),
        "nn_dist": M.nn_distance(fake, real),
        "diversity": M.sample_diversity(fake),
    }


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = 8192
    reals = {f: DataSetLoader().load_file(f, dataset="Testing") for f in FOLDS}

    rows = []
    for prefix, (paradigm, mode) in CONFIGS.items():
        dirs = sorted(p.parent for p in REPO.glob(f"checkpoints/{prefix}_s*/ckpt_final.pt"))
        if not dirs:
            print(f"(skip {prefix}: no runs)")
            continue
        is_diff = prefix.startswith("p4_")
        acc = defaultdict(list)
        for rd in dirs:
            for f in FOLDS:
                pool = reals[f]
                real = pool[np.random.choice(len(pool), min(n, len(pool)), replace=False)]
                r = eval_diffusion(rd, real, n, dev) if is_diff else eval_run(rd, real, n, dev)
                for k in KEYS:
                    acc[k].append(r[k])
            print(f"evaluated {rd.name}", flush=True)
        rows.append((paradigm, mode, len(dirs), {k: (np.mean(acc[k]), np.std(acc[k])) for k in KEYS}))

    cols = ["paradigm", "mode", "seeds"] + KEYS
    lines = ["# Final 2x2 comparison: {GAN, Diffusion} x {free, constructed}",
             "mean +/- std over seeds x held-out days 8/9/10", "",
             "| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for paradigm, mode, nseeds, stats in rows:
        cells = [paradigm, mode, str(nseeds)] + [f"{stats[k][0]:.4f} +/- {stats[k][1]:.4f}" for k in KEYS]
        lines.append("| " + " | ".join(cells) + " |")

    text = "\n".join(lines)
    (REPO / "data_analysis" / "comparison_final.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()

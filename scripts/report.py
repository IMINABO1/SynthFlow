"""Evaluate a trained generator checkpoint against real held-out data.

Reports validity, fake-vs-real distribution summaries, and diversity /
memorization signals (all in raw DecPre space).

Run:  .venv/Scripts/python -m scripts.report --ckpt checkpoints/ckpt_final.pt
"""
import argparse

import numpy as np
import torch

from pathlib import Path

from train import load_scaler, build_generator
from data.loader import DataSetLoader
import metrics as M


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt_final.pt")
    p.add_argument("--scaler", default="checkpoints/scaler.npz")
    p.add_argument("--n", type=int, default=8192)
    p.add_argument("--val_fold", type=int, default=7, help="Test_CF_{fold} = held-out day")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev)
    sc = load_scaler(Path(a.scaler))
    G = build_generator(ck["config"], dev)
    G.load_state_dict(ck["generator"])
    G.eval()

    val = DataSetLoader().load_file(a.val_fold, dataset="Testing")
    fake = sc.inverse(G.sample(a.n, device=dev)).detach().cpu().numpy()
    real = val[np.random.choice(len(val), min(a.n, len(val)), replace=False)]

    print(f"checkpoint={a.ckpt}  epoch={ck['epoch']+1}  n={a.n}")
    print("--- validity (generated) ---")
    for k, v in M.validity_stats(fake).items():
        print(f"  {k}: {v:.3%}")
    print(f"--- validity (real sanity): {M.validity_stats(real)['valid/all']:.3%} ---")
    print("--- distributions: mean(std)  FAKE vs REAL ---")
    for name, fn in [("spread", M.spread), ("depth", M.total_depth), ("imbalance", M.imbalance)]:
        f, r = fn(fake), fn(real)
        print(f"  {name:9s} fake {f.mean():+.4f}({f.std():.4f})  real {r.mean():+.4f}({r.std():.4f})")
    print("--- diversity / memorization ---")
    print(f"  nn_distance(fake->real): {M.nn_distance(fake, real):.4f}  (near 0 => memorizing)")
    print(f"  pairwise_fake diversity: {M.sample_diversity(fake):.4f}")
    print(f"  pairwise_real diversity: {M.sample_diversity(real):.4f}  (fake<<real => collapse)")


if __name__ == "__main__":
    main()

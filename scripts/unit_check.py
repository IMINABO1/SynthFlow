"""Data-independent smoke checks: models, gradient penalty, validity penalty,
metrics, and the scaler round-trip. No dataset required.

Run:  .venv/Scripts/python -m scripts.unit_check
"""
import numpy as np
import torch

from models.wgan import Generator, Critic, gradient_penalty, lob_violation_penalty
from lob_layout import (ASK_PRICE_IDX, BID_PRICE_IDX, ASK_VOL_IDX, BID_VOL_IDX,
                        N_LOB_FEATURES)
import metrics as M
from train import MinMaxScaler

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(f"[{PASS if cond else FAIL}] {name}")


def make_valid_books(n=64, seed=0):
    """Construct order books that satisfy all geometry constraints."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, N_LOB_FEATURES), dtype=np.float32)
    mid = rng.uniform(10, 100, size=n)
    tick = rng.uniform(0.01, 0.05, size=n)
    for lvl in range(len(ASK_PRICE_IDX)):
        x[:, ASK_PRICE_IDX[lvl]] = mid + (lvl + 1) * tick     # ascending asks
        x[:, BID_PRICE_IDX[lvl]] = mid - (lvl + 1) * tick     # descending bids
        x[:, ASK_VOL_IDX[lvl]] = rng.uniform(1, 10, size=n)
        x[:, BID_VOL_IDX[lvl]] = rng.uniform(1, 10, size=n)
    return x


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  torch={torch.__version__}  cuda_avail={torch.cuda.is_available()}")

    G = Generator(latent_dim=100, out_dim=40).to(dev)
    D = Critic(in_dim=40).to(dev)

    # --- model shapes ---
    z = torch.randn(8, 100, device=dev)
    fake = G(z)
    check("Generator output shape [8,40]", tuple(fake.shape) == (8, 40))
    check("Generator.sample shape [5,40]", tuple(G.sample(5).shape) == (5, 40))
    check("Generator output in [-1,1] (tanh)", bool(fake.abs().max() <= 1.0 + 1e-5))
    score = D(fake)
    check("Critic output shape [8,1]", tuple(score.shape) == (8, 1))

    # --- gradient penalty ---
    real = torch.randn(8, 40, device=dev)
    gp = gradient_penalty(D, real, fake.detach())
    check("gradient_penalty is finite scalar", gp.dim() == 0 and torch.isfinite(gp))
    check("gradient_penalty non-negative", float(gp) >= 0)

    # --- validity penalty ---
    valid = torch.tensor(make_valid_books(), device=dev)
    pen_valid = lob_violation_penalty(valid, ASK_PRICE_IDX, BID_PRICE_IDX)
    check("validity penalty ~0 on valid books", float(pen_valid) < 1e-4)
    crossed = valid.clone()
    crossed[:, BID_PRICE_IDX[0]] = crossed[:, ASK_PRICE_IDX[0]] + 1.0  # force crossed book
    pen_bad = lob_violation_penalty(crossed, ASK_PRICE_IDX, BID_PRICE_IDX)
    check("validity penalty >0 on crossed books", float(pen_bad) > 0)

    # --- metrics ---
    vb = make_valid_books()
    vs = M.validity_stats(vb)
    check("validity_stats: 100% valid on valid books", abs(vs["valid/all"] - 1.0) < 1e-6)
    mets = M.all_metrics(vb, make_valid_books(seed=1))
    check("all_metrics returns finite values",
          all(np.isfinite(v) for v in mets.values()))

    # --- scaler round-trip ---
    data = make_valid_books(128)
    sc = MinMaxScaler.fit(data)
    rt = sc.inverse(sc.transform(data))
    check("MinMaxScaler round-trip ~ identity", float(np.abs(rt - data).max()) < 1e-3)
    t = torch.tensor(sc.transform(data), device=dev)
    check("MinMaxScaler.inverse works on torch tensors",
          tuple(sc.inverse(t).shape) == data.shape)

    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n{'ALL PASS' if n_fail == 0 else str(n_fail) + ' FAILED'} "
          f"({len(results) - n_fail}/{len(results)})")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

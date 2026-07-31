"""Train a DDPM to generate FI-2010 LOB snapshots.

Two modes:
  --diffusion_mode free         diffuse directly in [0,1] unit book space (may be invalid)
  --diffusion_mode constrained  diffuse in unconstrained gap-space; map back => valid by construction

Mirrors train.py (GlobalScaler, metrics, wandb, checkpoints) so results are comparable.
Usage:
    python train_diffusion.py --diffusion_mode constrained
    python train_diffusion.py --diffusion_mode free --smoke
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.loader import DataSetLoader, DEFAULT_DATA_PATH
from train import GlobalScaler, load_data
from models.diffusion import Denoiser, GaussianDiffusion
import gap_transform as G
import metrics as M


class Standardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = np.where(std == 0, 1.0, std).astype(np.float32)

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        return cls(x.mean(0), x.std(0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, x):
        if isinstance(x, torch.Tensor):
            return x * torch.as_tensor(self.std, device=x.device) + torch.as_tensor(self.mean, device=x.device)
        return x * self.std + self.mean


@dataclass
class DiffConfig:
    data_path: str = DEFAULT_DATA_PATH
    normalization: str = "DecPre"
    diffusion_mode: str = "constrained"  # "free" or "constrained"
    timesteps: int = 500
    batch_size: int = 512
    epochs: int = 100
    lr: float = 2e-4
    log_every: int = 100
    ckpt_every: int = 10
    metric_batch: int = 2048
    seed: int = 0
    run_name: str = "diff"
    out_dir: str = ""
    smoke: bool = False

    def __post_init__(self):
        if not self.out_dir:
            self.out_dir = f"checkpoints/{self.run_name}"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_x0(unit_books: np.ndarray, mode: str, device) -> np.ndarray:
    """Map unit-space books to the space the diffusion operates in."""
    if mode == "free":
        return unit_books
    t = torch.from_numpy(unit_books).to(device)
    return G.book_to_params(t).cpu().numpy()


def _to_books(x0: torch.Tensor, mode: str) -> torch.Tensor:
    """Map diffusion-space samples back to unit-space books."""
    return x0 if mode == "free" else G.params_to_book(x0)


def train(cfg: DiffConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw, val_raw = load_data(cfg)  # DecPre; smoke path handled in load_data
    scaler = GlobalScaler.fit(train_raw)
    scaler.save(out_dir / "scaler.npz")
    train_unit = scaler.transform(train_raw)

    x0 = _to_x0(train_unit, cfg.diffusion_mode, device)
    std = Standardizer.fit(x0)
    x0s = std.transform(x0)

    ds = TensorDataset(torch.from_numpy(x0s))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    ref_n = min(cfg.metric_batch, len(val_raw))
    val_ref_raw = val_raw[np.random.choice(len(val_raw), ref_n, replace=False)]

    model = Denoiser(dim=40).to(device)
    diff = GaussianDiffusion(model, timesteps=cfg.timesteps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    run = _init_wandb(cfg)

    def sample_books_raw(n):
        xs = diff.sample(n, dim=40, device=device)
        x = std.inverse(xs)
        book_unit = _to_books(x, cfg.diffusion_mode)
        return scaler.inverse(book_unit)

    step = 0
    for epoch in range(cfg.epochs):
        for (xb,) in dl:
            loss = diff.p_losses(xb.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % cfg.log_every == 0:
                _log(run, {"loss/mse": float(loss.item()), "epoch": epoch}, step)
            step += 1

        with torch.no_grad():
            fake_raw = sample_books_raw(cfg.metric_batch)
        logs = {"epoch": epoch}
        logs.update(M.all_metrics(fake_raw, val_ref_raw))
        _log(run, logs, step)
        print(f"[e{epoch}] mse={float(loss.item()):.4f} valid_all={logs['valid/all']:.3f} "
              f"diversity={logs['diversity/pairwise_fake']:.3f}", flush=True)

        if (epoch + 1) % cfg.ckpt_every == 0 or epoch == cfg.epochs - 1:
            _save_ckpt(out_dir / "ckpt_final.pt", diff, cfg, std, epoch, step)

    if run is not None:
        run.finish()
    print("done.", flush=True)


def _init_wandb(cfg: DiffConfig):
    if cfg.smoke:
        return None
    try:
        import wandb
        return wandb.init(project="synthflow", name=cfg.run_name, config=asdict(cfg))
    except Exception as e:  # noqa: BLE001
        print(f"[wandb disabled: {e}]")
        return None


def _log(run, logs, step):
    if run is not None:
        run.log(logs, step=step)


def _save_ckpt(path, diff, cfg, std, epoch, step):
    torch.save({"diffusion": diff.state_dict(), "config": asdict(cfg),
                "std_mean": std.mean, "std_std": std.std, "epoch": epoch, "step": step}, path)
    print(f"saved {path}", flush=True)


def parse_args() -> DiffConfig:
    p = argparse.ArgumentParser(description="Train a DDPM on FI-2010 LOB snapshots.")
    for fld in fields(DiffConfig):
        v = fld.default
        if isinstance(v, bool):
            p.add_argument(f"--{fld.name}", action="store_true")
        elif isinstance(v, (int, float, str)):
            p.add_argument(f"--{fld.name}", type=type(v), default=v)
    return DiffConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    cfg = parse_args()
    if cfg.smoke:
        cfg.epochs = min(cfg.epochs, 1)
    train(cfg)

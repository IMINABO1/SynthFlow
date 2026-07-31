"""WGAN-GP training for synthetic FI-2010 LOB snapshots.

Baseline (default lambda_valid=0.0) is a pure WGAN-GP. Turn --lambda_valid up
to add the soft order-book-validity penalty and watch validity metrics rise.

Usage:
    python train.py                      # full run, wandb online if available
    python train.py --smoke              # tiny end-to-end sanity run
    python train.py --lambda_valid 1.0   # add validity penalty
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.loader import DataSetLoader, DEFAULT_DATA_PATH
from models.wgan import Generator, Critic, gradient_penalty, lob_violation_penalty
from models.constrained import ConstrainedGenerator
from lob_layout import ASK_PRICE_IDX, BID_PRICE_IDX, ASK_VOL_IDX, BID_VOL_IDX
import metrics as M


# --------------------------------------------------------------------------- #
# Per-feature scaling to [-1, 1] (fit on train only; needed for the tanh output)
# --------------------------------------------------------------------------- #
class MinMaxScaler:
    def __init__(self, feature_min: np.ndarray, feature_max: np.ndarray) -> None:
        self.min = feature_min.astype(np.float32)
        self.max = feature_max.astype(np.float32)
        rng = self.max - self.min
        rng[rng == 0] = 1.0  # constant features -> avoid div by zero
        self.range = rng

    @classmethod
    def fit(cls, x: np.ndarray) -> "MinMaxScaler":
        return cls(x.min(axis=0), x.max(axis=0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (2.0 * (x - self.min) / self.range - 1.0).astype(np.float32)

    def inverse(self, x):
        # works for np.ndarray or torch.Tensor (raw-space reconstruction)
        if isinstance(x, torch.Tensor):
            mn = torch.as_tensor(self.min, device=x.device)
            rg = torch.as_tensor(self.range, device=x.device)
            return (x + 1.0) / 2.0 * rg + mn
        return ((x + 1.0) / 2.0 * self.range + self.min).astype(np.float32)

    def save(self, path: Path) -> None:
        np.savez(path, kind="minmax", min=self.min, max=self.max)

    @classmethod
    def load(cls, path: Path) -> "MinMaxScaler":
        d = np.load(path)
        return cls(d["min"], d["max"])


# --------------------------------------------------------------------------- #
# Global scaling to [0, 1] with ONE shared scale for all price columns and one
# for all volume columns. Unlike per-feature min-max, a shared price scale
# preserves cross-level price ordering, so a constructed (sorted) book stays
# sorted after transform/inverse. Used by the constrained generator.
# --------------------------------------------------------------------------- #
class GlobalScaler:
    _PIDX = ASK_PRICE_IDX + BID_PRICE_IDX
    _VIDX = ASK_VOL_IDX + BID_VOL_IDX

    def __init__(self, price_min, price_max, vol_min, vol_max) -> None:
        self.price_min = float(price_min)
        self.vol_min = float(vol_min)
        self.price_range = float(price_max) - float(price_min) or 1.0
        self.vol_range = float(vol_max) - float(vol_min) or 1.0

    @classmethod
    def fit(cls, x: np.ndarray) -> "GlobalScaler":
        return cls(x[:, cls._PIDX].min(), x[:, cls._PIDX].max(),
                   x[:, cls._VIDX].min(), x[:, cls._VIDX].max())

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = np.array(x, dtype=np.float32, copy=True)
        out[:, self._PIDX] = (x[:, self._PIDX] - self.price_min) / self.price_range
        out[:, self._VIDX] = (x[:, self._VIDX] - self.vol_min) / self.vol_range
        return out

    def inverse(self, x):
        is_torch = isinstance(x, torch.Tensor)
        out = x.clone() if is_torch else np.array(x, dtype=np.float32, copy=True)
        out[:, self._PIDX] = x[:, self._PIDX] * self.price_range + self.price_min
        out[:, self._VIDX] = x[:, self._VIDX] * self.vol_range + self.vol_min
        return out

    def save(self, path: Path) -> None:
        np.savez(path, kind="global", price_min=self.price_min,
                 price_max=self.price_min + self.price_range,
                 vol_min=self.vol_min, vol_max=self.vol_min + self.vol_range)

    @classmethod
    def load(cls, path: Path) -> "GlobalScaler":
        d = np.load(path)
        return cls(d["price_min"], d["price_max"], d["vol_min"], d["vol_max"])


def load_scaler(path: Path):
    """Load whichever scaler kind was saved (defaults to minmax for old files)."""
    d = np.load(path)
    kind = str(d["kind"]) if "kind" in d else "minmax"
    return GlobalScaler.load(path) if kind == "global" else MinMaxScaler.load(path)


def build_generator(cfg: dict, device):
    """Reconstruct the generator from a saved checkpoint config (for eval)."""
    hidden = tuple(cfg.get("gen_hidden", (128, 256, 256)))
    GenCls = ConstrainedGenerator if cfg.get("model_type", "mlp") == "constrained" else Generator
    return GenCls(latent_dim=cfg["latent_dim"], out_dim=40, hidden=hidden).to(device)


@dataclass
class Config:
    data_path: str = DEFAULT_DATA_PATH
    normalization: str = "DecPre"
    latent_dim: int = 100
    batch_size: int = 512
    epochs: int = 50
    lr: float = 1e-4
    beta1: float = 0.0
    beta2: float = 0.9
    n_critic: int = 5
    lambda_gp: float = 10.0
    lambda_valid: float = 0.0
    log_every: int = 100      # steps between metric logs
    ckpt_every: int = 5       # epochs between checkpoints
    metric_batch: int = 2048
    seed: int = 0
    run_name: str = "run"
    out_dir: str = ""  # defaults to checkpoints/<run_name> in __post_init__
    smoke: bool = False
    model_type: str = "mlp"  # "mlp" (free output) or "constrained" (valid by construction)
    gen_hidden: tuple = (128, 256, 256)
    critic_hidden: tuple = (256, 256, 128)

    def __post_init__(self):
        if not self.out_dir:
            self.out_dir = f"checkpoints/{self.run_name}"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(cfg: Config):
    loader = DataSetLoader(cfg.data_path, normalization=cfg.normalization)
    if cfg.smoke:
        # smallest footprint: just day 1, and only a slice of it
        train = loader.load_day(1)[:5000]
        val = loader.load_day(1)[5000:6000]
    else:
        train, val, _test = loader.load_split()  # temporal 1-7 / 8 / 9
    return train, val


def train(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data -------------------------------------------------------------- #
    train_raw, val_raw = load_data(cfg)
    # constrained generator builds sorted books, so it needs the order-preserving
    # GlobalScaler ([0,1] shared price scale); the plain MLP uses per-feature [-1,1].
    scaler = (GlobalScaler if cfg.model_type == "constrained" else MinMaxScaler).fit(train_raw)
    scaler.save(out_dir / "scaler.npz")
    train_s = scaler.transform(train_raw)

    ds = TensorDataset(torch.from_numpy(train_s))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    # fixed real reference (raw space) for validity/diversity metrics
    ref_n = min(cfg.metric_batch, len(val_raw))
    val_ref_raw = val_raw[np.random.choice(len(val_raw), ref_n, replace=False)]

    # ---- models ------------------------------------------------------------ #
    GenCls = ConstrainedGenerator if cfg.model_type == "constrained" else Generator
    G = GenCls(latent_dim=cfg.latent_dim, out_dim=train_s.shape[1],
               hidden=cfg.gen_hidden).to(device)
    D = Critic(in_dim=train_s.shape[1], hidden=cfg.critic_hidden).to(device)
    use_penalty = cfg.lambda_valid > 0 and cfg.model_type != "constrained"
    optG = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    optD = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

    run = _init_wandb(cfg)

    step = 0
    for epoch in range(cfg.epochs):
        for (real_s,) in dl:
            real = real_s.to(device)
            b = real.size(0)

            # ---- critic: n_critic steps ------------------------------------ #
            for _ in range(cfg.n_critic):
                z = torch.randn(b, cfg.latent_dim, device=device)
                fake = G(z).detach()
                d_real = D(real).mean()
                d_fake = D(fake).mean()
                gp = gradient_penalty(D, real, fake)
                loss_D = d_fake - d_real + cfg.lambda_gp * gp
                optD.zero_grad(set_to_none=True)
                loss_D.backward()
                optD.step()

            # ---- generator: 1 step ----------------------------------------- #
            z = torch.randn(b, cfg.latent_dim, device=device)
            fake = G(z)
            loss_G = -D(fake).mean()
            if use_penalty:
                fake_raw = scaler.inverse(fake)  # penalty lives in price-comparable space
                loss_G = loss_G + cfg.lambda_valid * lob_violation_penalty(
                    fake_raw, ASK_PRICE_IDX, BID_PRICE_IDX
                )
            optG.zero_grad(set_to_none=True)
            loss_G.backward()
            optG.step()

            if step % cfg.log_every == 0:
                logs = {
                    "loss/critic": float(loss_D.item()),
                    "loss/generator": float(loss_G.item()),
                    "loss/gradient_penalty": float(gp.item()),
                    "loss/w_distance_est": float((d_real - d_fake).item()),
                    "epoch": epoch,
                }
                with torch.no_grad():
                    fake_eval = G.sample(cfg.metric_batch, device=device)
                    fake_eval_raw = scaler.inverse(fake_eval)
                    vpen = float(lob_violation_penalty(
                        fake_eval_raw, ASK_PRICE_IDX, BID_PRICE_IDX))
                logs["loss/validity_penalty"] = vpen  # raw violation, pre-lambda
                logs.update(M.all_metrics(fake_eval_raw, val_ref_raw))
                _log(run, logs, step)
                print(f"[e{epoch} s{step}] D={logs['loss/critic']:.3f} "
                      f"G={logs['loss/generator']:.3f} gp={logs['loss/gradient_penalty']:.3f} "
                      f"vpen={vpen:.4f} valid_all={logs['valid/all']:.3f}", flush=True)
            step += 1

        if (epoch + 1) % cfg.ckpt_every == 0 or epoch == cfg.epochs - 1:
            _save_ckpt(out_dir / f"ckpt_e{epoch+1}.pt", G, D, optG, optD, cfg, epoch, step)

    _save_ckpt(out_dir / "ckpt_final.pt", G, D, optG, optD, cfg, cfg.epochs - 1, step)
    if run is not None:
        run.finish()
    print("done.", flush=True)


# --------------------------------------------------------------------------- #
# wandb + checkpoint helpers (all optional / defensive)
# --------------------------------------------------------------------------- #
def _init_wandb(cfg: Config):
    if cfg.smoke:
        return None
    try:
        import wandb
        return wandb.init(project="synthflow", name=cfg.run_name, config=asdict(cfg))
    except Exception as e:  # noqa: BLE001 - wandb missing / offline / not logged in
        print(f"[wandb disabled: {e}]")
        return None


def _log(run, logs, step):
    if run is not None:
        run.log(logs, step=step)


def _save_ckpt(path, G, D, optG, optD, cfg, epoch, step):
    torch.save(
        {
            "generator": G.state_dict(),
            "critic": D.state_dict(),
            "optG": optG.state_dict(),
            "optD": optD.state_dict(),
            "config": asdict(cfg),
            "epoch": epoch,
            "step": step,
        },
        path,
    )
    print(f"saved {path}", flush=True)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train WGAN-GP on FI-2010 LOB snapshots.")
    # Use raw field defaults (not a constructed Config, whose __post_init__ would
    # bake out_dir="checkpoints/run" and stop --run_name from propagating).
    for fld in fields(Config):
        v = fld.default
        if isinstance(v, bool):
            p.add_argument(f"--{fld.name}", action="store_true" if not v else "store_false")
        elif isinstance(v, (int, float, str)):
            p.add_argument(f"--{fld.name}", type=type(v), default=v)
        # non-scalar fields (gen_hidden, critic_hidden) are set programmatically only
    args = p.parse_args()
    return Config(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    if cfg.smoke:
        cfg.epochs = min(cfg.epochs, 1)
        cfg.log_every = 10
    train(cfg)

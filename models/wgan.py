"""WGAN-GP model for single 40-dim LOB snapshots.

Generator: noise -> 40 features (tanh output, paired with per-feature scaling
to [-1, 1] in train.py). Critic: 40 features -> scalar score (no final
activation; WGAN critic, not a classifier).

We use LayerNorm (not BatchNorm) in the critic: BatchNorm breaks the WGAN-GP
gradient-penalty assumption that the critic is a function of each sample
independently.
"""
from typing import Sequence

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: Sequence[int]) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.LeakyReLU(0.2, inplace=True)]
        prev = h
    return nn.Sequential(*layers), prev


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, out_dim: int = 40,
                 hidden: Sequence[int] = (128, 256, 256)) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        body, last = _mlp(latent_dim, hidden)
        self.net = nn.Sequential(body, nn.Linear(last, out_dim), nn.Tanh())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    @torch.no_grad()
    def sample(self, n: int, device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        return self.forward(z)


class Critic(nn.Module):
    def __init__(self, in_dim: int = 40, hidden: Sequence[int] = (256, 256, 128)) -> None:
        super().__init__()
        body, last = _mlp(in_dim, hidden)
        self.net = nn.Sequential(body, nn.Linear(last, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """WGAN-GP gradient penalty: E[(||grad critic(x_hat)||_2 - 1)^2] on points
    interpolated between real and fake samples."""
    device = real.device
    eps = torch.rand(real.size(0), 1, device=device)
    interp = (eps * real + (1.0 - eps) * fake).requires_grad_(True)
    scores = critic(interp)
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=interp,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


def lob_violation_penalty(
    batch: torch.Tensor,
    ask_price_idx: Sequence[int],
    bid_price_idx: Sequence[int],
) -> torch.Tensor:
    """Soft penalty for order-book geometry violations. Returns a scalar; the
    caller multiplies by lambda_valid (default 0.0 -> pure WGAN-GP baseline).

    NOTE: assumes ``batch`` is in a price-comparable space (all price columns
    share a scale, e.g. raw/DecPre). If prices are per-feature min-max scaled,
    unscale them before calling this. Whether DecPre preserves cross-level
    monotonicity is checked in data_analysis/eda.py.
    """
    ask = batch[:, ask_price_idx]   # [B, L] ask prices, should ascend with level
    bid = batch[:, bid_price_idx]   # [B, L] bid prices, should descend with level

    asc_viol = torch.relu(ask[:, :-1] - ask[:, 1:])   # >0 where asks descend
    desc_viol = torch.relu(bid[:, 1:] - bid[:, :-1])  # >0 where bids ascend
    spread_viol = torch.relu(bid[:, 0] - ask[:, 0])   # >0 where book is crossed

    return asc_viol.mean() + desc_viol.mean() + spread_viol.mean()

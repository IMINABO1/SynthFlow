"""Minimal DDPM for a 40-dim vector (LOB book, or its gap-space params).

Denoiser is an MLP with a sinusoidal time embedding; GaussianDiffusion holds the
linear-beta schedule, the closed-form forward `q_sample`, the noise-prediction
loss, and ancestral sampling. Small and fast — the data is only 40-dim.
"""
import math
from typing import Sequence

import torch
import torch.nn as nn


def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class Denoiser(nn.Module):
    def __init__(self, dim: int = 40, hidden: Sequence[int] = (256, 256, 256),
                 time_dim: int = 64) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(),
                                      nn.Linear(time_dim, time_dim))
        layers, prev = [], dim + time_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU()]
            prev = h
        layers += [nn.Linear(prev, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(_sinusoidal_embedding(t, self.time_dim))
        return self.net(torch.cat([x, temb], dim=1))


def _extract(a: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
    return a.gather(0, t).reshape(len(t), *([1] * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, model: Denoiser, timesteps: int = 500,
                 beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("acp", acp)
        self.register_buffer("sqrt_acp", torch.sqrt(acp))
        self.register_buffer("sqrt_one_minus_acp", torch.sqrt(1.0 - acp))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (_extract(self.sqrt_acp, t, x0.shape) * x0
                + _extract(self.sqrt_one_minus_acp, t, x0.shape) * noise)

    def p_losses(self, x0: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, self.timesteps, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        pred = self.model(self.q_sample(x0, t, noise), t)
        return nn.functional.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, n: int, dim: int = 40, device=None) -> torch.Tensor:
        device = device or self.betas.device
        x = torch.randn(n, dim, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            eps = self.model(x, t)
            alpha = _extract(self.alphas, t, x.shape)
            beta = _extract(self.betas, t, x.shape)
            somacp = _extract(self.sqrt_one_minus_acp, t, x.shape)
            mean = (x - beta / somacp * eps) / torch.sqrt(alpha)
            x = mean + (torch.sqrt(beta) * torch.randn_like(x) if i > 0 else 0.0)
        return x

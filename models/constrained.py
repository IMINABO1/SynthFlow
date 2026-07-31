"""Constrained generator: valid limit-order books by construction.

Instead of emitting 40 prices/volumes directly (which the plain MLP does, giving
~0% structurally valid books), this generator emits a base price plus positive
increments and assembles the book so that, mathematically:
  - ask prices ascend across levels,
  - bid prices descend across levels,
  - the spread (best ask - best bid) is positive,
  - all volumes are non-negative.

Everything is a differentiable transform of the MLP's 40 raw outputs, so WGAN
training is unchanged. Output lives in the same normalized [0,1] space as the
GlobalScaler (a single shared price scale, so ordering is preserved end-to-end).
"""
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from lob_layout import (ASK_PRICE_IDX, BID_PRICE_IDX, ASK_VOL_IDX, BID_VOL_IDX,
                        N_LEVELS, N_LOB_FEATURES)


def _mlp_body(in_dim: int, hidden: Sequence[int]):
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.LeakyReLU(0.2, inplace=True)]
        prev = h
    return nn.Sequential(*layers), prev


class ConstrainedGenerator(nn.Module):
    def __init__(self, latent_dim: int = 100, out_dim: int = N_LOB_FEATURES,
                 hidden: Sequence[int] = (128, 256, 256), gap_scale: float = 0.02,
                 min_spread: float = 1e-4) -> None:
        super().__init__()
        assert out_dim == N_LOB_FEATURES, "constrained generator assumes the 40-feature LOB"
        self.latent_dim = latent_dim
        self.gap_scale = gap_scale
        # A tiny positive floor so the spread can't underflow to exactly 0 in
        # float32 (real books always have a >=1-tick spread anyway).
        self.min_spread = min_spread
        body, last = _mlp_body(latent_dim, hidden)
        self.body = body
        self.head = nn.Linear(last, N_LOB_FEATURES)
        self.register_buffer("ap", torch.tensor(ASK_PRICE_IDX))
        self.register_buffer("bp", torch.tensor(BID_PRICE_IDX))
        self.register_buffer("av", torch.tensor(ASK_VOL_IDX))
        self.register_buffer("bv", torch.tensor(BID_VOL_IDX))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        o = self.head(self.body(z))          # [B, 40] raw
        g = self.gap_scale

        bid0 = torch.sigmoid(o[:, 0:1])               # best bid in [0,1]
        spread = F.softplus(o[:, 1:2]) * g + self.min_spread   # strictly > 0
        ask0 = bid0 + spread
        ask_gaps = F.softplus(o[:, 2:2 + (N_LEVELS - 1)]) * g   # 9 positive
        bid_gaps = F.softplus(o[:, 11:11 + (N_LEVELS - 1)]) * g

        asks = torch.cat([ask0, ask0 + torch.cumsum(ask_gaps, dim=1)], dim=1)  # ascending
        bids = torch.cat([bid0, bid0 - torch.cumsum(bid_gaps, dim=1)], dim=1)  # descending
        ask_vols = torch.sigmoid(o[:, 20:30])         # >= 0
        bid_vols = torch.sigmoid(o[:, 30:40])

        out = torch.empty(o.size(0), N_LOB_FEATURES, device=o.device, dtype=o.dtype)
        out[:, self.ap] = asks
        out[:, self.bp] = bids
        out[:, self.av] = ask_vols
        out[:, self.bv] = bid_vols
        return out

    @torch.no_grad()
    def sample(self, n: int, device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        return self.forward(torch.randn(n, self.latent_dim, device=device))

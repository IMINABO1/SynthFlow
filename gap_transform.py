"""Exact bijection between a valid LOB (in [0,1] unit space) and an unconstrained
parameter vector r in R^40.

A diffusion model diffuses in r-space (unconstrained, Gaussian-friendly); mapping
r back through ``params_to_book`` yields a book that is valid *by construction*
(ascending asks, descending bids, positive spread, non-negative volumes) — the
diffusion analogue of models/constrained.py.

Parametrization of r (per row): r0 -> best bid, r1 -> spread, r[2:11] -> 9 ask
gaps, r[11:20] -> 9 bid gaps, r[20:30]/r[30:40] -> ask/bid volumes. Prices/vols in
(0,1) use logit; positive gaps/spread use softplus and its inverse.
"""
import torch

from lob_layout import ASK_PRICE_IDX, BID_PRICE_IDX, ASK_VOL_IDX, BID_VOL_IDX, N_LEVELS

_EPS = 1e-6
MIN_SPREAD = 1e-4


def _logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(_EPS, 1.0 - _EPS)
    return torch.log(p) - torch.log1p(-p)


def _softplus_inv(y: torch.Tensor) -> torch.Tensor:
    # inverse of softplus: log(exp(y) - 1), stable for small positive y
    y = y.clamp(min=_EPS)
    return y + torch.log(-torch.expm1(-y))


def params_to_book(r: torch.Tensor) -> torch.Tensor:
    """Unconstrained r [B,40] -> valid unit-space book [B,40]."""
    bid0 = torch.sigmoid(r[:, 0:1])
    spread = torch.nn.functional.softplus(r[:, 1:2]) + MIN_SPREAD
    ask0 = bid0 + spread
    ask_gaps = torch.nn.functional.softplus(r[:, 2:2 + (N_LEVELS - 1)])
    bid_gaps = torch.nn.functional.softplus(r[:, 11:11 + (N_LEVELS - 1)])
    asks = torch.cat([ask0, ask0 + torch.cumsum(ask_gaps, dim=1)], dim=1)
    bids = torch.cat([bid0, bid0 - torch.cumsum(bid_gaps, dim=1)], dim=1)
    ask_vols = torch.sigmoid(r[:, 20:30])
    bid_vols = torch.sigmoid(r[:, 30:40])

    book = torch.empty_like(r)
    book[:, ASK_PRICE_IDX] = asks
    book[:, BID_PRICE_IDX] = bids
    book[:, ASK_VOL_IDX] = ask_vols
    book[:, BID_VOL_IDX] = bid_vols
    return book


def book_to_params(book: torch.Tensor) -> torch.Tensor:
    """Valid unit-space book [B,40] -> unconstrained r [B,40] (inverse of above)."""
    asks = book[:, ASK_PRICE_IDX]        # [B,10] ascending
    bids = book[:, BID_PRICE_IDX]        # [B,10] descending
    ask_vols = book[:, ASK_VOL_IDX]
    bid_vols = book[:, BID_VOL_IDX]

    bid0 = bids[:, 0:1]
    spread = asks[:, 0:1] - bid0
    ask_gaps = asks[:, 1:] - asks[:, :-1]     # positive
    bid_gaps = bids[:, :-1] - bids[:, 1:]     # positive

    r = torch.empty_like(book)
    r[:, 0:1] = _logit(bid0)
    r[:, 1:2] = _softplus_inv(spread - MIN_SPREAD)
    r[:, 2:2 + (N_LEVELS - 1)] = _softplus_inv(ask_gaps)
    r[:, 11:11 + (N_LEVELS - 1)] = _softplus_inv(bid_gaps)
    r[:, 20:30] = _logit(ask_vols)
    r[:, 30:40] = _logit(bid_vols)
    return r

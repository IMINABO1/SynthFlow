"""Training-time financial metrics for synthetic LOB snapshots.

Following the HRT lesson that synthetic market data must be judged by whether
*financial properties survive* (not by loss curves), we log:

- validity: % of books with monotonic asks/bids and positive spread,
- distribution summaries: spread, total depth, order-book imbalance,
- diversity / collapse signals: nearest-neighbour distance to real data
  (memorization check) and mean pairwise distance among fakes (mode-collapse
  check).

All metrics expect inputs in *raw / price-comparable* space (e.g. DecPre),
i.e. inverse-scaled back from the generator's [-1, 1] output.
"""
from typing import Dict

import numpy as np

from lob_layout import ASK_PRICE_IDX, BID_PRICE_IDX, ASK_VOL_IDX, BID_VOL_IDX


def _np(x) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def validity_stats(batch) -> Dict[str, float]:
    x = _np(batch)
    ask = x[:, ASK_PRICE_IDX]
    bid = x[:, BID_PRICE_IDX]
    asks_mono = np.all(np.diff(ask, axis=1) >= 0, axis=1)
    bids_mono = np.all(np.diff(bid, axis=1) <= 0, axis=1)
    pos_spread = (ask[:, 0] - bid[:, 0]) > 0
    return {
        "valid/asks_monotonic": float(asks_mono.mean()),
        "valid/bids_monotonic": float(bids_mono.mean()),
        "valid/positive_spread": float(pos_spread.mean()),
        "valid/all": float((asks_mono & bids_mono & pos_spread).mean()),
    }


def spread(batch) -> np.ndarray:
    x = _np(batch)
    return x[:, ASK_PRICE_IDX[0]] - x[:, BID_PRICE_IDX[0]]


def total_depth(batch) -> np.ndarray:
    x = _np(batch)
    return x[:, ASK_VOL_IDX].sum(axis=1) + x[:, BID_VOL_IDX].sum(axis=1)


def imbalance(batch) -> np.ndarray:
    x = _np(batch)
    ask_v = x[:, ASK_VOL_IDX].sum(axis=1)
    bid_v = x[:, BID_VOL_IDX].sum(axis=1)
    return (bid_v - ask_v) / (ask_v + bid_v + 1e-8)


def distribution_summary(batch) -> Dict[str, float]:
    out = {}
    for name, fn in (("spread", spread), ("depth", total_depth), ("imbalance", imbalance)):
        v = fn(batch)
        out[f"dist/{name}_mean"] = float(np.mean(v))
        out[f"dist/{name}_std"] = float(np.std(v))
    return out


def nn_distance(fake, real_ref, ref_sample: int = 2048, chunk: int = 256) -> float:
    """Mean nearest-neighbour L2 distance from each fake to the real reference
    set. Very small => the generator may be memorizing training rows."""
    f = _np(fake)
    r = _np(real_ref)
    if len(r) > ref_sample:
        idx = np.linspace(0, len(r) - 1, ref_sample).astype(int)
        r = r[idx]
    mins = []
    for i in range(0, len(f), chunk):
        d = np.linalg.norm(f[i:i + chunk, None, :] - r[None, :, :], axis=2)
        mins.append(d.min(axis=1))
    return float(np.concatenate(mins).mean())


def sample_diversity(fake, sample: int = 1024) -> float:
    """Mean pairwise L2 distance among fakes. Low => mode collapse."""
    f = _np(fake)
    if len(f) > sample:
        idx = np.linspace(0, len(f) - 1, sample).astype(int)
        f = f[idx]
    n = len(f)
    if n < 2:
        return 0.0
    d = np.linalg.norm(f[:, None, :] - f[None, :, :], axis=2)
    return float(d.sum() / (n * (n - 1)))


def all_metrics(fake, real_ref) -> Dict[str, float]:
    """Convenience bundle for one logging step (inputs in raw space)."""
    out = {}
    out.update(validity_stats(fake))
    out.update(distribution_summary(fake))
    out["diversity/nn_distance_to_real"] = nn_distance(fake, real_ref)
    out["diversity/pairwise_fake"] = sample_diversity(fake)
    return out

"""Compare all trained runs in checkpoints/*/ against the same held-out day.

Emits a markdown table (validity, fake-vs-real distributions, diversity/memorization)
to data_analysis/experiment_comparison.md.

Run:  .venv/Scripts/python -m scripts.compare
"""
from pathlib import Path

import numpy as np
import torch

from models.wgan import Generator
from train import MinMaxScaler
from data.loader import DataSetLoader
import metrics as M

REPO = Path(__file__).resolve().parents[1]
DEFAULT_GEN_HIDDEN = (128, 256, 256)  # v1 baseline ckpt has no gen_hidden in its config


def eval_run(run_dir: Path, real: np.ndarray, n: int, dev: str) -> dict:
    ck = torch.load(run_dir / "ckpt_final.pt", map_location=dev)
    sc = MinMaxScaler.load(run_dir / "scaler.npz")
    cfg = ck["config"]
    hidden = tuple(cfg.get("gen_hidden", DEFAULT_GEN_HIDDEN))
    G = Generator(latent_dim=cfg["latent_dim"], out_dim=40, hidden=hidden).to(dev)
    G.load_state_dict(ck["generator"])
    G.eval()
    fake = sc.inverse(G.sample(n, device=dev)).detach().cpu().numpy()

    vs = M.validity_stats(fake)
    row = {
        "run": run_dir.name,
        "lambda": cfg.get("lambda_valid", 0.0),
        "valid_all": vs["valid/all"],
        "asks_mono": vs["valid/asks_monotonic"],
        "bids_mono": vs["valid/bids_monotonic"],
        "spread+": vs["valid/positive_spread"],
        "spread_err": abs(M.spread(fake).mean() - M.spread(real).mean()),
        "depth_err": abs(M.total_depth(fake).mean() - M.total_depth(real).mean()),
        "nn_dist": M.nn_distance(fake, real),
        "diversity": M.sample_diversity(fake),
    }
    return row


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = 8192
    val = DataSetLoader().load_file(7, dataset="Testing")  # held-out day 8
    real = val[np.random.choice(len(val), min(n, len(val)), replace=False)]
    real_div = M.sample_diversity(real)

    run_dirs = sorted(p.parent for p in REPO.glob("checkpoints/*/ckpt_final.pt"))
    if not run_dirs:
        print("no runs found under checkpoints/*/ckpt_final.pt")
        return

    rows = []
    for rd in run_dirs:
        try:
            rows.append(eval_run(rd, real, n, dev))
            print(f"evaluated {rd.name}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"skip {rd.name}: {e}", flush=True)

    cols = ["run", "lambda", "valid_all", "asks_mono", "bids_mono", "spread+",
            "spread_err", "depth_err", "nn_dist", "diversity"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"

    def fmt(v):
        return v if isinstance(v, str) else (f"{v:.4f}")

    lines = ["# Experiment comparison (held-out day 8)", "",
             f"real diversity (reference): {real_div:.4f}", "",
             header, sep]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    text = "\n".join(lines)
    (REPO / "data_analysis" / "experiment_comparison.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nWrote data_analysis/experiment_comparison.md ({len(rows)} runs)")


if __name__ == "__main__":
    main()

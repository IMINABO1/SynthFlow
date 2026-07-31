"""Phase-3b rigor: 3 configs x 3 seeds, 100 epochs each, for a mean+/-std result.

Configs: constrained, mlp lambda=0, mlp lambda=10 (the informative Phase-2 corners).
Runs sequentially to checkpoints/p3_<name>_s<seed>/, wandb offline; skips finished runs.

Run:  .venv/Scripts/python -m scripts.run_phase3
"""
import os
os.environ.setdefault("WANDB_MODE", "offline")

from pathlib import Path

from train import Config, train

EPOCHS = 100
SEEDS = (0, 1, 2)
GEN, CRIT = (128, 256, 256), (256, 256, 128)


def build_configs():
    cfgs = []
    for seed in SEEDS:
        cfgs.append(Config(run_name=f"p3_constrained_s{seed}", model_type="constrained",
                           epochs=EPOCHS, seed=seed, gen_hidden=GEN, critic_hidden=CRIT))
        cfgs.append(Config(run_name=f"p3_mlp_l0_s{seed}", model_type="mlp", lambda_valid=0.0,
                           epochs=EPOCHS, seed=seed, gen_hidden=GEN, critic_hidden=CRIT))
        cfgs.append(Config(run_name=f"p3_mlp_l10_s{seed}", model_type="mlp", lambda_valid=10.0,
                           epochs=EPOCHS, seed=seed, gen_hidden=GEN, critic_hidden=CRIT))
    return cfgs


def main():
    cfgs = build_configs()
    print(f"=== Phase-3b: {len(cfgs)} runs (3 configs x {len(SEEDS)} seeds) x {EPOCHS} epochs ===",
          flush=True)
    for i, cfg in enumerate(cfgs, 1):
        tag = f"[{i}/{len(cfgs)}] {cfg.run_name}"
        if Path(cfg.out_dir, "ckpt_final.pt").exists():
            print(f"{tag}: already done, skipping", flush=True)
            continue
        print(f"{tag}: starting", flush=True)
        train(cfg)
        print(f"{tag}: finished", flush=True)
    print("PHASE3B DONE", flush=True)


if __name__ == "__main__":
    main()

"""Phase-4 capstone: diffusion {constrained, free} x seeds {0,1,2}, 100 epochs each.

Completes the 2x2 vs the existing GAN runs (p3_constrained_*, p3_mlp_l0_*). Runs
sequentially to checkpoints/p4_*/, wandb offline; skips finished runs.

Run:  .venv/Scripts/python -m scripts.run_phase4
"""
import os
os.environ.setdefault("WANDB_MODE", "offline")

from pathlib import Path

from train_diffusion import DiffConfig, train

EPOCHS = 100
SEEDS = (0, 1, 2)


def build_configs():
    cfgs = []
    for seed in SEEDS:
        cfgs.append(DiffConfig(run_name=f"p4_diff_constrained_s{seed}",
                               diffusion_mode="constrained", epochs=EPOCHS, seed=seed))
        cfgs.append(DiffConfig(run_name=f"p4_diff_free_s{seed}",
                               diffusion_mode="free", epochs=EPOCHS, seed=seed))
    return cfgs


def main():
    cfgs = build_configs()
    print(f"=== Phase-4: {len(cfgs)} diffusion runs x {EPOCHS} epochs ===", flush=True)
    for i, cfg in enumerate(cfgs, 1):
        tag = f"[{i}/{len(cfgs)}] {cfg.run_name}"
        if Path(cfg.out_dir, "ckpt_final.pt").exists():
            print(f"{tag}: already done, skipping", flush=True)
            continue
        print(f"{tag}: starting", flush=True)
        train(cfg)
        print(f"{tag}: finished", flush=True)
    print("PHASE4 DONE", flush=True)


if __name__ == "__main__":
    main()

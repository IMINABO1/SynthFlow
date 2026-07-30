"""Run the Phase-2 study: model {base, wide} x lambda_valid {0, 0.1, 1, 10}.

Runs sequentially, each to checkpoints/<run_name>/, wandb offline. Skips any run
whose ckpt_final.pt already exists (so it resumes if interrupted). base_lambda0 is
the v1 baseline (already trained, moved to checkpoints/base_lambda0/).

Run:  .venv/Scripts/python -m scripts.run_experiments
"""
import os
os.environ.setdefault("WANDB_MODE", "offline")

from pathlib import Path

from train import Config, train

BASE_GEN, BASE_CRIT = (128, 256, 256), (256, 256, 128)
WIDE_GEN, WIDE_CRIT = (256, 512, 512), (512, 512, 256)
EPOCHS = 100
LAMBDAS = (0.1, 1.0, 10.0)


def build_configs():
    cfgs = []
    # base model: lambda sweep (lambda=0 already done as base_lambda0)
    for lam in LAMBDAS:
        cfgs.append(Config(run_name=f"base_lambda{lam}", epochs=EPOCHS, lambda_valid=lam,
                           gen_hidden=BASE_GEN, critic_hidden=BASE_CRIT))
    # wide model: full lambda sweep incl. 0 (its own baseline)
    for lam in (0.0,) + LAMBDAS:
        cfgs.append(Config(run_name=f"wide_lambda{lam}", epochs=EPOCHS, lambda_valid=lam,
                           gen_hidden=WIDE_GEN, critic_hidden=WIDE_CRIT))
    return cfgs


def main():
    cfgs = build_configs()
    print(f"=== Phase-2 study: {len(cfgs)} runs x {EPOCHS} epochs ===", flush=True)
    for i, cfg in enumerate(cfgs, 1):
        tag = f"[{i}/{len(cfgs)}] {cfg.run_name}"
        if Path(cfg.out_dir, "ckpt_final.pt").exists():
            print(f"{tag}: already done, skipping", flush=True)
            continue
        print(f"{tag}: starting (lambda={cfg.lambda_valid}, gen={cfg.gen_hidden})", flush=True)
        train(cfg)
        print(f"{tag}: finished", flush=True)
    print("ALL EXPERIMENTS DONE", flush=True)


if __name__ == "__main__":
    main()

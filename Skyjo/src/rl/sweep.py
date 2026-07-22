"""wandb hyperparameter sweep over the Skyjo training loop.

Creates a Bayesian sweep (or joins an existing one via --sweep-id) and runs an
agent that executes truncated training runs, maximizing the best score margin
vs Phillips. Each run gets a unique `sweep_<run-id>` checkpoint prefix, so the
regular `skyjo_ppo` checkpoints are never touched.
"""

import argparse
import traceback

import wandb

from Skyjo.src.rl.train import WANDB_PROJECT, train, WANDB_ENTITY

SWEEP_TIMESTEPS = 20_000_000

SWEEP_CONFIG = {
    "name": "skyjo-ppo",
    "method": "bayes",
    "metric": {"name": "best/margin", "goal": "maximize"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 8e-4,
        },
        "ent_coef": {
            "distribution": "log_uniform_values",
            "min": 3e-3,
            "max": 3e-2,
        },
        "gamma": {"values": [0.99, 0.995, 0.997]},
        "net_arch": {
            "values": [
                [256, 256, 128],
                [512, 512, 256],
                [256, 256, 256, 128],
                [512, 512, 256, 128],
            ]
        },
        "total_timesteps": {"value": SWEEP_TIMESTEPS},
        # Skip periodic checkpoints; each run still saves its best/final model.
        "save_every": {"value": 10**12},
    },
}


def _run():
    # A crashed run must not kill the (unattended) agent.
    try:
        train(model_prefix=None)
    except Exception:
        traceback.print_exc()
        wandb.finish(exit_code=1)


def main():
    parser = argparse.ArgumentParser(
        description="Run a wandb hyperparameter sweep for the Skyjo agent."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=40,
        help="Number of runs this agent executes.",
    )
    parser.add_argument(
        "--sweep-id",
        default=None,
        help="Join an existing sweep instead of creating a new one.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=SWEEP_TIMESTEPS,
        help="Timesteps per sweep run.",
    )
    args = parser.parse_args()

    sweep_id = args.sweep_id
    if sweep_id is None:
        SWEEP_CONFIG["parameters"]["total_timesteps"]["value"] = args.timesteps
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_PROJECT, entity=WANDB_ENTITY)
        print(f"Created sweep {sweep_id} (resume with --sweep-id {sweep_id})")

    wandb.agent(
        sweep_id,
        function=_run,
        count=args.count,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
    )


if __name__ == "__main__":
    main()

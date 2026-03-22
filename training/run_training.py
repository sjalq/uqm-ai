#!/usr/bin/env python3
"""
Entry point for training.

Usage:
    python training/run_training.py
    python training/run_training.py --total-timesteps 500000 --num-envs 4
"""

import sys
import argparse
from training.config import TrainingConfig
from training.ppo import train


def main():
    parser = argparse.ArgumentParser(description="Train UQM Melee AI")

    # Override any TrainingConfig field via CLI
    config = TrainingConfig()
    for field_name, field_value in vars(config).items():
        field_type = type(field_value)
        if field_type == bool:
            parser.add_argument(f"--{field_name.replace('_', '-')}",
                                type=lambda x: x.lower() in ('true', '1', 'yes'),
                                default=field_value)
        else:
            parser.add_argument(f"--{field_name.replace('_', '-')}",
                                type=field_type, default=field_value)

    args = parser.parse_args()

    # Apply CLI overrides to config
    for field_name in vars(config):
        cli_name = field_name.replace('_', '-')
        if hasattr(args, field_name):
            setattr(config, field_name, getattr(args, field_name))

    print("Training configuration:")
    for k, v in vars(config).items():
        print(f"  {k}: {v}")
    print()

    results = train(config)
    return 0 if results.get("competency_reached_at_seconds") is not None else 1


if __name__ == "__main__":
    sys.exit(main())

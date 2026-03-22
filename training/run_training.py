#!/usr/bin/env python3
"""
Entry point for training.

Usage:
    python -m training.run_training
    python -m training.run_training --total-timesteps 500000 --num-envs 4
    python -m training.run_training --smoke-test
"""

import sys
import argparse
import logging
import traceback
import time
from training.config import TrainingConfig
from training.ppo import train


logger = logging.getLogger(__name__)


def smoke_test():
    """
    Quick smoke test: create one env, run 10 steps, report success/failure.
    Tests that libmelee.so loads, init works, step works, and reset works.
    """
    print("=== SMOKE TEST ===")
    print("Testing: env creation, reset, 10 steps, close")

    from uqm_env.melee_env import MeleeEnv
    env = None
    try:
        env = MeleeEnv(ship_p1=5, ship_p2=5, p2_cyborg=True,
                       frame_skip=4, headless=True, seed=42)
        print("[OK] Environment created")

        obs, info = env.reset()
        print(f"[OK] Reset complete. Obs shape: {obs.shape}, info: {info}")

        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            status = "DONE" if terminated else "ok"
            print(f"  Step {i+1:2d}: reward={reward:+.4f} p1_crew={info.get('p1_crew','?')} "
                  f"p2_crew={info.get('p2_crew','?')} [{status}]")
            if terminated:
                print("[INFO] Episode ended, resetting...")
                obs, info = env.reset()
                print(f"[OK] Re-reset complete. info: {info}")

        print("\n[OK] Smoke test PASSED - environment is functional")
        stats = env.get_stats()
        print(f"  Stats: {stats}")
        return 0

    except Exception as e:
        print(f"\n[FAIL] Smoke test FAILED: {e}")
        traceback.print_exc()
        return 1

    finally:
        if env is not None:
            try:
                env.close()
                print("[OK] Environment closed")
            except Exception as e:
                print(f"[WARN] Close failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train UQM Melee AI")

    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a quick 10-step smoke test instead of full training")

    # Override any TrainingConfig field via CLI
    config = TrainingConfig()
    for field_name, field_value in vars(config).items():
        field_type = type(field_value)
        if field_type == bool:
            parser.add_argument(f"--{field_name.replace('_', '-')}",
                                type=lambda x: x.lower() in ('true', '1', 'yes'),
                                default=field_value)
        elif isinstance(field_value, list):
            # Skip list fields (like combat_actions) - not easily CLI-configurable
            continue
        else:
            parser.add_argument(f"--{field_name.replace('_', '-')}",
                                type=field_type, default=field_value)

    args = parser.parse_args()

    if args.smoke_test:
        return smoke_test()

    # Apply CLI overrides to config
    for field_name in vars(config):
        if hasattr(args, field_name):
            setattr(config, field_name, getattr(args, field_name))

    print("Training configuration:")
    for k, v in vars(config).items():
        print(f"  {k}: {v}")
    print()

    try:
        results = train(config)
        success = results.get("competency_reached_at_seconds") is not None
        print(f"\nTraining complete. Best win rate: {results.get('best_win_rate', 0):.2%}")
        if success:
            print(f"Target reached at {results['competency_reached_at_seconds']:.1f}s")
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Training crashed: {e}\n{traceback.format_exc()}")
        print(f"\nTraining CRASHED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

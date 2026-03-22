#!/usr/bin/env python3
"""
IMMUTABLE EVALUATION CODE - DO NOT MODIFY.

This file is hash-checked by the evolution orchestrator.
Any modification results in automatic disqualification.

Evaluates a trained agent against the built-in Cyborg AI
and measures time-to-competency from training logs.
"""

import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path


def evaluate_checkpoint(checkpoint_path: str, ship_p1: int = 5, ship_p2: int = 5,
                        n_episodes: int = 100, frame_skip: int = 4) -> dict:
    """
    Load a trained agent checkpoint and play n_episodes against Cyborg AI.

    Returns:
        dict with win_rate, avg_episode_length, avg_crew_remaining, results
    """
    from training.agent import MeleeAgent
    from uqm_env.melee_env import MeleeEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load agent
    agent = MeleeAgent(encoder_type="siglip", hidden_dim=256, action_dim=32)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        agent.load_state_dict(state_dict)
    except RuntimeError:
        # Try CNN fallback
        agent = MeleeAgent(encoder_type="cnn", hidden_dim=256, action_dim=32)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        agent.load_state_dict(state_dict)

    agent = agent.to(device)
    agent.eval()

    env = MeleeEnv(
        ship_p1=ship_p1, ship_p2=ship_p2,
        p2_cyborg=True, frame_skip=frame_skip,
        headless=True, seed=42
    )

    episode_results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_length = 0

        while not done:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action, _, _, _ = agent.get_action_and_value(obs_tensor)

            obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            episode_length += 1

        episode_results.append({
            "winner": info.get("winner", -1),
            "episode_length": episode_length,
            "p1_crew_remaining": info.get("p1_crew", 0),
            "p2_crew_remaining": info.get("p2_crew", 0),
        })

    env.close()

    wins = sum(1 for r in episode_results if r["winner"] == 0)
    win_rate = wins / n_episodes

    return {
        "win_rate": win_rate,
        "wins": wins,
        "losses": n_episodes - wins,
        "n_episodes": n_episodes,
        "avg_episode_length": np.mean([r["episode_length"] for r in episode_results]),
        "avg_crew_remaining": np.mean([r["p1_crew_remaining"] for r in episode_results]),
        "results": episode_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--ship-p1", type=int, default=5, help="Player 1 ship index")
    parser.add_argument("--ship-p2", type=int, default=5, help="Player 2 ship index")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    args = parser.parse_args()

    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Ship matchup: {args.ship_p1} vs {args.ship_p2}")
    print(f"Episodes: {args.episodes}")

    results = evaluate_checkpoint(
        args.checkpoint,
        ship_p1=args.ship_p1,
        ship_p2=args.ship_p2,
        n_episodes=args.episodes,
    )

    print(f"\nResults:")
    print(f"  Win rate: {results['win_rate']:.2%} ({results['wins']}/{results['n_episodes']})")
    print(f"  Avg episode length: {results['avg_episode_length']:.1f} steps")
    print(f"  Avg crew remaining: {results['avg_crew_remaining']:.1f}")

    if args.output:
        # Don't include per-episode results in output to keep it small
        output = {k: v for k, v in results.items() if k != "results"}
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

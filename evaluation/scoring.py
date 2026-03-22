#!/usr/bin/env python3
"""
IMMUTABLE SCORING CODE - DO NOT MODIFY.

This file is hash-checked by the evolution orchestrator.
Any modification results in automatic disqualification.

Measures time-to-competency from training logs:
the wall-clock seconds until the agent first achieves >= threshold win rate.
"""

import sys
import json
import argparse
from pathlib import Path


def measure_time_to_competency(results_path: str,
                                threshold: float = 0.8,
                                window_size: int = 3) -> dict:
    """
    Read training results and find when the agent first reached
    the target win rate.

    Args:
        results_path: path to results.json from training
        threshold: win rate threshold (default 0.8)
        window_size: rolling window of evaluations to smooth

    Returns:
        dict with wall_clock_seconds, training_steps, final_win_rate, learning_curve
    """
    with open(results_path) as f:
        results = json.load(f)

    training_log = results.get("training_log", [])

    # Extract evaluation points (entries with win_rate)
    eval_points = [
        entry for entry in training_log
        if "win_rate" in entry
    ]

    if not eval_points:
        return {
            "wall_clock_seconds": None,
            "training_steps": None,
            "final_win_rate": 0.0,
            "learning_curve": [],
            "threshold": threshold,
            "reached": False,
        }

    learning_curve = [
        (ep["global_step"], ep["win_rate"], ep["wall_clock_seconds"])
        for ep in eval_points
    ]

    # Find first point where rolling average >= threshold
    wall_clock_seconds = None
    training_steps = None

    for i in range(len(learning_curve)):
        start = max(0, i - window_size + 1)
        window = learning_curve[start:i + 1]
        avg_win_rate = sum(wr for _, wr, _ in window) / len(window)

        if avg_win_rate >= threshold:
            training_steps = learning_curve[i][0]
            wall_clock_seconds = learning_curve[i][2]
            break

    final_win_rate = learning_curve[-1][1] if learning_curve else 0.0

    return {
        "wall_clock_seconds": wall_clock_seconds,
        "training_steps": training_steps,
        "final_win_rate": final_win_rate,
        "learning_curve": [(step, wr) for step, wr, _ in learning_curve],
        "threshold": threshold,
        "reached": wall_clock_seconds is not None,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure time-to-competency")
    parser.add_argument("--results", required=True, help="Path to results.json")
    parser.add_argument("--threshold", type=float, default=0.8, help="Win rate threshold")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    score = measure_time_to_competency(args.results, args.threshold)

    print(f"Time-to-competency analysis:")
    print(f"  Threshold: {score['threshold']:.0%}")
    print(f"  Final win rate: {score['final_win_rate']:.2%}")

    if score["reached"]:
        print(f"  Reached at: {score['wall_clock_seconds']:.1f}s "
              f"({score['training_steps']} steps)")
    else:
        print(f"  NOT REACHED")

    if score["learning_curve"]:
        print(f"\n  Learning curve:")
        for step, wr in score["learning_curve"]:
            bar = "#" * int(wr * 40)
            print(f"    Step {step:>8d}: {wr:.2%} {bar}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(score, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())

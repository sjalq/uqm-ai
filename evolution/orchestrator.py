#!/usr/bin/env python3
"""
Evolution orchestrator - runs rounds of competing agent builds.

Each round:
1. Creates 3 git worktrees from HEAD
2. Launches 3 Claude agents to modify training code
3. Runs training sequentially (1 GPU, each agent gets exclusive access)
4. Measures time-to-competency for each
5. Winner's changes get merged to main
6. Repeat

Usage:
    python evolution/orchestrator.py --rounds 5 --agents 3
"""

import sys
import os
import json
import time
import hashlib
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


REPO_ROOT = Path(__file__).parent.parent
EVAL_LOCKFILE = REPO_ROOT / "evaluation" / "eval_lockfile.sha256"
HISTORY_FILE = REPO_ROOT / "evolution" / "history.json"


def verify_eval_integrity(worktree_path: Path) -> bool:
    """Check that evaluation code hasn't been modified."""
    lockfile = REPO_ROOT / "evaluation" / "eval_lockfile.sha256"
    if not lockfile.exists():
        print("WARNING: eval_lockfile.sha256 not found, skipping integrity check")
        return True

    expected = {}
    for line in lockfile.read_text().strip().split("\n"):
        hash_val, filepath = line.split("  ", 1)
        expected[filepath] = hash_val

    for filepath, expected_hash in expected.items():
        full_path = worktree_path / filepath
        if not full_path.exists():
            print(f"INTEGRITY FAIL: {filepath} missing in worktree")
            return False
        actual_hash = hashlib.sha256(full_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            print(f"INTEGRITY FAIL: {filepath} was modified!")
            return False

    return True


def create_worktrees(n_agents: int, round_num: int) -> list[Path]:
    """Create git worktrees for each agent."""
    worktrees = []
    for i in range(1, n_agents + 1):
        branch = f"evolution/round-{round_num}/agent-{i}"
        wt_path = REPO_ROOT / "worktrees" / f"round-{round_num}-agent-{i}"

        # Clean up if exists
        if wt_path.exists():
            subprocess.run(["git", "worktree", "remove", str(wt_path), "--force"],
                           cwd=REPO_ROOT, capture_output=True)
        subprocess.run(["git", "branch", "-D", branch],
                       cwd=REPO_ROOT, capture_output=True)

        # Create worktree
        subprocess.run(
            ["git", "worktree", "add", str(wt_path), "-b", branch, "HEAD"],
            cwd=REPO_ROOT, check=True, capture_output=True
        )
        worktrees.append(wt_path)

    return worktrees


def cleanup_worktrees(worktrees: list[Path]):
    """Remove worktrees."""
    for wt in worktrees:
        if wt.exists():
            subprocess.run(["git", "worktree", "remove", str(wt), "--force"],
                           cwd=REPO_ROOT, capture_output=True)
    subprocess.run(["git", "worktree", "prune"], cwd=REPO_ROOT, capture_output=True)


def launch_agent(worktree: Path, agent_num: int, round_num: int,
                 previous_winner_summary: str = "") -> subprocess.Popen:
    """Launch a Claude agent in a worktree to modify training code."""
    prompt = f"""You are Agent {agent_num} in Round {round_num} of an evolutionary competition.

YOUR GOAL: Modify the training code so the AI learns UQM Super Melee as FAST as possible.
The metric is TIME-TO-COMPETENCY: wall-clock seconds to reach 80% win rate vs Cyborg AI.
Lowest time wins. You are competing against {2} other agents.

YOU CAN MODIFY:
- training/ppo.py (primary target - the PPO training loop)
- training/agent.py (model architecture)
- training/config.py (hyperparameters)
- uqm_env/reward.py (reward shaping)
- Any other training-related code

YOU CANNOT MODIFY (hash-checked, instant disqualification):
- evaluation/evaluate.py
- evaluation/scoring.py
- evaluation/eval_lockfile.sha256

{f"PREVIOUS WINNER'S APPROACH: {previous_winner_summary}" if previous_winner_summary else "This is the first round - start from the baseline."}

WHEN DONE modifying code, create a file called DONE in the repo root:
    touch DONE

Your changes will be trained from scratch and measured automatically."""

    proc = subprocess.Popen(
        ["claude", "--dangerously-skip-permissions", "-p", prompt],
        cwd=worktree,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def run_training_for_agent(worktree: Path, agent_num: int) -> dict:
    """Run training in a worktree and return results."""
    print(f"\n{'='*60}")
    print(f"Training Agent {agent_num}")
    print(f"{'='*60}")

    output_dir = worktree / "outputs"
    checkpoint_dir = worktree / "checkpoints"
    log_dir = worktree / "logs"

    start_time = time.time()

    result = subprocess.run(
        [sys.executable, "-m", "training.run_training",
         "--output-dir", str(output_dir),
         "--checkpoint-dir", str(checkpoint_dir),
         "--log-dir", str(log_dir)],
        cwd=worktree,
        capture_output=True,
        text=True,
        timeout=7200,  # 2 hour timeout
    )

    elapsed = time.time() - start_time

    print(f"Agent {agent_num} training completed in {elapsed:.1f}s")
    if result.returncode != 0:
        print(f"Agent {agent_num} training FAILED:")
        print(result.stderr[-500:] if result.stderr else "No error output")

    # Read results
    results_file = output_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)

    return {
        "wall_clock_seconds": elapsed,
        "best_win_rate": 0.0,
        "competency_reached_at_seconds": None,
        "error": result.stderr[-500:] if result.stderr else "Unknown error",
    }


def pick_winner(agent_results: list[dict]) -> int:
    """
    Pick the winning agent.

    Priority:
    1. Lowest time-to-competency (if reached threshold)
    2. Highest final win rate (if none reached threshold)
    """
    # Agents that reached competency
    competent = [
        (i, r["competency_reached_at_seconds"])
        for i, r in enumerate(agent_results)
        if r.get("competency_reached_at_seconds") is not None
    ]

    if competent:
        # Fastest to competency wins
        return min(competent, key=lambda x: x[1])[0]

    # Nobody reached competency - highest win rate wins
    return max(range(len(agent_results)),
               key=lambda i: agent_results[i].get("best_win_rate", 0.0))


def merge_winner(worktree: Path, round_num: int, agent_num: int,
                 results: dict, all_results: list[dict]):
    """Merge winner's changes to main and tag."""
    # Get the branch name
    branch = f"evolution/round-{round_num}/agent-{agent_num}"

    # Commit any uncommitted changes in the worktree
    subprocess.run(["git", "add", "-A"], cwd=worktree, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m",
         f"Round {round_num} Agent {agent_num} modifications"],
        cwd=worktree, capture_output=True
    )

    # Squash merge to main
    subprocess.run(["git", "checkout", "main"], cwd=REPO_ROOT, check=True, capture_output=True)

    # Build commit message with insights from all agents
    msg_lines = [
        f"Evolution round {round_num}: Agent {agent_num} wins",
        "",
        f"Time-to-competency: {results.get('competency_reached_at_seconds', 'N/A')}s",
        f"Best win rate: {results.get('best_win_rate', 0):.2%}",
        "",
        "All agents:",
    ]

    for i, r in enumerate(all_results):
        ttc = r.get("competency_reached_at_seconds", "N/A")
        wr = r.get("best_win_rate", 0)
        marker = " (WINNER)" if i == agent_num - 1 else ""
        msg_lines.append(f"  Agent {i+1}: ttc={ttc}s, win_rate={wr:.2%}{marker}")

    commit_msg = "\n".join(msg_lines)

    subprocess.run(
        ["git", "merge", "--squash", branch],
        cwd=REPO_ROOT, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd=REPO_ROOT, check=True, capture_output=True
    )

    # Tag
    tag = f"evolution/round-{round_num}"
    subprocess.run(
        ["git", "tag", tag],
        cwd=REPO_ROOT, capture_output=True
    )

    print(f"Winner merged and tagged as {tag}")


def load_history() -> list:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(history: list):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def run_round(round_num: int, n_agents: int = 3,
              skip_modification: bool = False) -> dict:
    """Run a single evolution round."""
    print(f"\n{'#'*60}")
    print(f"EVOLUTION ROUND {round_num}")
    print(f"{'#'*60}")

    history = load_history()
    previous_summary = ""
    if history:
        last = history[-1]
        previous_summary = last.get("winner_summary", "")

    # Step 1: Create worktrees
    print("\nCreating worktrees...")
    worktrees = create_worktrees(n_agents, round_num)

    # Step 2: Launch agents to modify code (unless skipped for testing)
    if not skip_modification:
        print("\nLaunching Claude agents...")
        procs = []
        for i, wt in enumerate(worktrees, 1):
            proc = launch_agent(wt, i, round_num, previous_summary)
            procs.append(proc)
            print(f"  Agent {i} launched (PID {proc.pid})")

        # Wait for agents to finish (with timeout)
        print("\nWaiting for agents to finish modifying code...")
        for i, proc in enumerate(procs, 1):
            try:
                proc.wait(timeout=1800)  # 30 min timeout for code modification
            except subprocess.TimeoutExpired:
                print(f"  Agent {i} timed out, killing...")
                proc.kill()

    # Step 3: Verify integrity and train sequentially
    agent_results = []
    for i, wt in enumerate(worktrees, 1):
        if not verify_eval_integrity(wt):
            print(f"Agent {i} DISQUALIFIED - modified evaluation code")
            agent_results.append({
                "wall_clock_seconds": float("inf"),
                "best_win_rate": 0.0,
                "competency_reached_at_seconds": None,
                "disqualified": True,
            })
            continue

        result = run_training_for_agent(wt, i)
        agent_results.append(result)

    # Step 4: Pick winner
    winner_idx = pick_winner(agent_results)
    winner_num = winner_idx + 1
    print(f"\nWINNER: Agent {winner_num}")

    # Step 5: Merge winner
    merge_winner(worktrees[winner_idx], round_num, winner_num,
                 agent_results[winner_idx], agent_results)

    # Step 6: Record history
    round_record = {
        "round": round_num,
        "timestamp": datetime.now().isoformat(),
        "winner": winner_num,
        "agent_results": agent_results,
        "winner_summary": json.dumps(agent_results[winner_idx]),
    }
    history.append(round_record)
    save_history(history)

    # Step 7: Cleanup
    cleanup_worktrees(worktrees)

    return round_record


def main():
    parser = argparse.ArgumentParser(description="Evolution orchestrator")
    parser.add_argument("--rounds", type=int, default=1, help="Number of evolution rounds")
    parser.add_argument("--agents", type=int, default=3, help="Agents per round")
    parser.add_argument("--skip-modification", action="store_true",
                        help="Skip agent modification step (test training only)")
    args = parser.parse_args()

    for r in range(1, args.rounds + 1):
        round_record = run_round(r, args.agents, args.skip_modification)
        print(f"\nRound {r} complete. Winner: Agent {round_record['winner']}")

    print(f"\n{'='*60}")
    print(f"Evolution complete. {args.rounds} rounds finished.")
    print(f"History saved to {HISTORY_FILE}")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# cleanup-agents.sh - Kill the agent tmux session and remove worktrees.
#
# Usage: ./scripts/cleanup-agents.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="uqm-agents"

cd "$REPO_ROOT"

# Kill tmux session
tmux kill-session -t "$SESSION" 2>/dev/null && echo "Killed tmux session '$SESSION'" || echo "No tmux session '$SESSION' found"

# Remove worktrees
for wt in "$REPO_ROOT"/worktrees/agent-*/; do
    if [[ -d "$wt" ]]; then
        git worktree remove "$wt" --force 2>/dev/null && echo "Removed worktree: $wt" || echo "Failed to remove: $wt"
    fi
done

# Prune worktree list
git worktree prune

echo "Cleanup complete."

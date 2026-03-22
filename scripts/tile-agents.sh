#!/usr/bin/env bash
# tile-agents.sh - Launch up to 9 Claude agents in tmux panes, each in its own git worktree.
#
# Usage: ./scripts/tile-agents.sh [N] [prompt]
#   N      - number of agents (1-9, default: 3)
#   prompt - optional prompt to pass to each agent
#
# Each agent gets its own worktree at ./worktrees/agent-N/ on branch agent/N.
# Agents are tiled in a single tmux session called "uqm-agents".

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NUM_AGENTS="${1:-3}"
PROMPT="${2:-}"
SESSION="uqm-agents"

if [[ "$NUM_AGENTS" -lt 1 || "$NUM_AGENTS" -gt 9 ]]; then
    echo "Error: number of agents must be 1-9, got $NUM_AGENTS"
    exit 1
fi

# Ensure we're in a git repo with at least one commit
cd "$REPO_ROOT"
if ! git rev-parse HEAD &>/dev/null; then
    echo "Error: need at least one commit on main before creating worktrees."
    echo "Run: cd $REPO_ROOT && git add -A && git commit -m 'initial commit'"
    exit 1
fi

# Kill existing session if present
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create worktrees directory
mkdir -p "$REPO_ROOT/worktrees"

# Set up worktrees and branches
for i in $(seq 1 "$NUM_AGENTS"); do
    BRANCH="agent/$i"
    WORKTREE="$REPO_ROOT/worktrees/agent-$i"

    # Clean up stale worktree entry if directory is gone
    if git worktree list --porcelain | grep -q "$WORKTREE" && [[ ! -d "$WORKTREE" ]]; then
        git worktree remove "$WORKTREE" --force 2>/dev/null || true
    fi

    # Create worktree if it doesn't exist
    if [[ ! -d "$WORKTREE" ]]; then
        # Delete branch if it exists (from a previous run)
        git branch -D "$BRANCH" 2>/dev/null || true
        git worktree add "$WORKTREE" -b "$BRANCH" HEAD
    fi
done

# Build the claude command
CLAUDE_CMD="claude"
if [[ -n "$PROMPT" ]]; then
    CLAUDE_CMD="claude -p \"$PROMPT\""
fi

# Create tmux session with first agent
FIRST_WORKTREE="$REPO_ROOT/worktrees/agent-1"
tmux new-session -d -s "$SESSION" -c "$FIRST_WORKTREE" "$CLAUDE_CMD; bash"

# Add panes for remaining agents
for i in $(seq 2 "$NUM_AGENTS"); do
    WORKTREE="$REPO_ROOT/worktrees/agent-$i"
    tmux split-window -t "$SESSION" -c "$WORKTREE" "$CLAUDE_CMD; bash"
    # Re-tile after each split to keep layout clean
    tmux select-layout -t "$SESSION" tiled
done

# Final tiled layout
tmux select-layout -t "$SESSION" tiled

# Attach
echo "Attaching to tmux session '$SESSION' with $NUM_AGENTS agents..."
echo "Worktrees at: $REPO_ROOT/worktrees/agent-{1..$NUM_AGENTS}/"
echo ""
echo "tmux cheatsheet:"
echo "  Ctrl-b arrow  - switch pane"
echo "  Ctrl-b z      - zoom/unzoom pane"
echo "  Ctrl-b d      - detach (agents keep running)"
echo ""
tmux attach -t "$SESSION"

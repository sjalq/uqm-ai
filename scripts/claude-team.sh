#!/usr/bin/env bash
# claude-team.sh - Launch Claude Code inside tmux with agent teams enabled.
#
# Claude auto-detects tmux and tiles teammates into split panes.
# Just tell Claude how many teammates you want (up to 9) in your prompt.
#
# Usage: ./scripts/claude-team.sh [prompt]
#   prompt - optional initial prompt (interactive if omitted)
#
# Examples:
#   ./scripts/claude-team.sh
#   ./scripts/claude-team.sh "Create a team of 3 to research UQM melee AI approaches"

set -euo pipefail

SESSION="uqm-team"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Preflight
command -v tmux >/dev/null || { echo "Error: tmux not installed. Run: sudo pacman -S tmux"; exit 1; }
command -v claude >/dev/null || { echo "Error: claude CLI not found in PATH"; exit 1; }

# Kill existing session if present
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Build claude command
CLAUDE_ARGS=(--teammate-mode tmux --dangerously-skip-permissions)
if [[ $# -gt 0 ]]; then
    CLAUDE_ARGS+=(-p "$*")
fi

# Start tmux session running claude in the repo root
# Claude's agent teams auto-detect tmux and tile teammates into split panes
tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" "claude ${CLAUDE_ARGS[*]}; bash"

echo "Attaching to tmux session '$SESSION'..."
echo "Claude will tile teammates into split panes automatically."
echo ""
echo "  Ctrl-b d      - detach (agents keep running)"
echo "  Ctrl-b arrow  - switch pane"
echo "  Ctrl-b z      - zoom/unzoom pane"
echo "  Shift+Down    - cycle teammates (in-process mode)"
echo ""
tmux attach -t "$SESSION"

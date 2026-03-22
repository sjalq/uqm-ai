#!/usr/bin/env bash
# claude-team.sh - Launch Claude Code inside tmux with agent teams enabled.
#
# Claude auto-detects tmux and tiles teammates into split panes.
# Each pane gets a distinct background color for easy identification.
#
# Usage: ./scripts/claude-team.sh [prompt]
#   prompt - optional initial prompt (interactive if omitted)
#
# Examples:
#   ./scripts/claude-team.sh
#   ./scripts/claude-team.sh "Create a team of 3 to research UQM melee AI approaches"

set -euo pipefail

SESSION="uqm-team"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

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
tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" "claude ${CLAUDE_ARGS[*]}; bash"

# Styling: distinct pane borders and colors
# Active pane gets a bright border, inactive panes get dim borders
tmux set-option -t "$SESSION" pane-border-style "fg=colour240"
tmux set-option -t "$SESSION" pane-active-border-style "fg=colour51,bold"
tmux set-option -t "$SESSION" pane-border-lines heavy
tmux set-option -t "$SESSION" pane-border-indicators arrows

# Show pane titles (agent names will appear here)
tmux set-option -t "$SESSION" pane-border-status top
tmux set-option -t "$SESSION" pane-border-format \
    "#{?pane_active,#[fg=colour16#,bg=colour51#,bold],#[fg=colour255#,bg=colour236]} #P: #{pane_title} "

# Hook: auto-color new panes when Claude spawns teammates
# Each pane gets a subtle tinted background so you can tell them apart at a glance
tmux set-hook -t "$SESSION" after-split-window \
    'run-shell "
        pane_count=$(tmux list-panes -t uqm-team | wc -l)
        case $pane_count in
            2) tmux select-pane -t uqm-team:.1 -P bg=colour233 \; \
                   select-pane -t uqm-team:.0 -P bg=colour234 ;;
            3) tmux select-pane -t uqm-team:.2 -P bg=colour232 \; \
                   select-pane -t uqm-team:.1 -P bg=colour233 \; \
                   select-pane -t uqm-team:.0 -P bg=colour234 ;;
            4) tmux select-pane -t uqm-team:.3 -P bg=colour17  \; \
                   select-pane -t uqm-team:.2 -P bg=colour232 \; \
                   select-pane -t uqm-team:.1 -P bg=colour233 \; \
                   select-pane -t uqm-team:.0 -P bg=colour234 ;;
            5) tmux select-pane -t uqm-team:.4 -P bg=colour52  \; \
                   select-pane -t uqm-team:.3 -P bg=colour17  \; \
                   select-pane -t uqm-team:.2 -P bg=colour232 \; \
                   select-pane -t uqm-team:.1 -P bg=colour233 \; \
                   select-pane -t uqm-team:.0 -P bg=colour234 ;;
            *) tmux select-pane -t uqm-team:.-1 -P bg=colour236 ;;
        esac
        tmux select-layout -t uqm-team tiled
    "'

# Set the lead pane color immediately
tmux select-pane -t "$SESSION:.0" -P "bg=colour234"
tmux select-pane -t "$SESSION:.0" -T "LEAD"

echo "Attaching to tmux session '$SESSION'..."
echo "Panes are tiled and color-coded:"
echo "  Lead:    dark grey    (colour234)"
echo "  Agent 1: darker grey  (colour233)"
echo "  Agent 2: darkest grey (colour232)"
echo "  Agent 3: dark blue    (colour17)"
echo "  Agent 4: dark red     (colour52)"
echo ""
echo "  Ctrl-b d      - detach (agents keep running)"
echo "  Ctrl-b arrow  - switch pane"
echo "  Ctrl-b z      - zoom/unzoom pane"
echo ""
tmux attach -t "$SESSION"

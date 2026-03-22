---
name: goteam
description: Launch agent team to tackle the current phase of work with coordinated non-overlapping exploration
user_invocable: true
---

You are the team lead. Read CLAUDE.md first to understand the project.

## PRE-FLIGHT CHECKLIST - COMPLETE BEFORE SPAWNING ANY AGENT

### 1. Tmux setup
```bash
SESSION=$(tmux display-message -p '#S')
tmux select-layout -t $SESSION tiled
tmux set-hook -t $SESSION after-split-window 'select-layout tiled'
```

### 2. Verify training pipeline works end-to-end
```bash
source .venv/bin/activate
# Quick 200-step test - verify: SigLIP loads, GPU used, pixels non-zero, rewards flow
SDL_VIDEODRIVER=offscreen python -c "
from uqm_env.melee_env import MeleeEnv
env = MeleeEnv(); obs, info = env.reset()
print(f'Pixels non-zero: {(obs>0).sum()}, Crew: {info}')
for i in range(20):
    obs, r, d, _, info = env.step(8)
    if r != 0 or d: print(f'Step {i}: reward={r}, done={d}, info={info}')
env.close()
"
```
If pixels are zero, SDL is broken. If crew never changes, C bridge is broken. FIX BEFORE PROCEEDING.

### 3. Create git worktrees manually (isolation: "worktree" parameter is BROKEN)
```bash
git worktree add /home/schalk/git/uqm-ai-rN-a1 -b rN-agent-1 main
git worktree add /home/schalk/git/uqm-ai-rN-a2 -b rN-agent-2 main
git worktree add /home/schalk/git/uqm-ai-rN-a3 -b rN-agent-3 main
# Copy deps to each
for d in /home/schalk/git/uqm-ai-rN-a{1,2,3}; do
  cp uqm_env/libmelee.so "$d/uqm_env/"
  ln -sf /home/schalk/git/uqm-ai/uqm-megamod/content "$d/uqm-megamod/content"
  ln -sf /home/schalk/git/uqm-ai/.venv "$d/.venv"
done
```

### 4. Spawn agents - verify EVERY pane
After EACH spawn:
```bash
tmux list-panes -F '#{pane_id} #{pane_current_command}'
```
Every non-lead pane MUST show `claude` or version number (e.g. `2.1.81`). If any shows `fish`, kill and respawn.

### 5. Color-code panes
```bash
tmux select-pane -t $SESSION:.0 -P "bg=colour234" -T "LEAD"
tmux select-pane -t $SESSION:.1 -P "bg=colour233" -T "AGENT-1"
tmux select-pane -t $SESSION:.2 -P "bg=colour232" -T "AGENT-2"
tmux select-pane -t $SESSION:.3 -P "bg=colour17"  -T "AGENT-3"
tmux select-layout -t $SESSION tiled
```

## AGENT PROMPTS - EVERY AGENT GETS TOLD:
- `cd /home/schalk/git/uqm-ai-rN-aX` (their worktree) as FIRST instruction
- DO NOT MODIFY training/agent.py (model is LOCKED)
- DO NOT run training (lead runs it after)
- Declare approach, code, commit, DONE

## COMPETITION RULES
- NEVER pick winners by reading code. Only MEASURED training results.
- The lead runs 5-min training in EACH worktree sequentially after agents finish.
- Compare: steps completed, SPS, win rate, damage dealt, survival time.
- Use composite scoring from CLAUDE.md when nobody wins.

## COORDINATION
- Spawn Agent 1 first, wait for approach declaration.
- Tell Agent 2 what Agent 1 is doing so it picks DIFFERENT.
- Same for Agent 3.
- Agents code immediately after declaring. No approval needed.
- Shut down agents AS SOON AS they say DONE. Don't leave them idle.
- One GPU shared - lead runs training sequentially.
- NEVER ask "want me to keep going?" - keep running rounds until user says stop.

## CURRENT TASK: $ARGUMENTS

If no task is specified, read the project state and determine what needs doing next.

## VERIFICATION - CRITICAL
When an agent claims done, independently verify:
1. Go into their worktree
2. Check agent.py is unmodified: `git diff main -- training/agent.py | wc -l` = 0
3. sha256sum evaluation files against eval_lockfile.sha256
4. Run training yourself and check REAL numbers

## GIT WORKFLOW
- Each agent works on its own branch (created by worktree)
- Push ALL branches (winners and losers)
- Squash-merge only winner to main
- Tag: goteam/round-N-winner, goteam/round-N-agent-X-loser
- Commit message includes real metrics and notable loser ideas

## CLEANUP
After each round:
```bash
git worktree remove --force /home/schalk/git/uqm-ai-rN-a{1,2,3}
```
Then TeamDelete.

---
name: goteam
description: Launch agent team to tackle the current phase of work with coordinated non-overlapping exploration
user_invocable: true
---

You are the team lead. Read CLAUDE.md first to understand the project.

IMPORTANT: Use TeamCreate to create an agent TEAM with TEAMMATES (not the Agent tool which creates subagents). Teammates get their own tmux panes, their own git worktrees, and can communicate with each other and with you. Each teammate has a full independent copy of the repo.

COORDINATION RULES:
- Create the team with TeamCreate, spawning 3 teammates each in their own worktree.
- Spawn Teammate 1 first, tell it to declare its approach then start coding immediately.
- Once you hear Teammate 1's approach, spawn Teammate 2 telling it what Teammate 1 is doing so it picks a DIFFERENT approach. Starts immediately after declaring.
- Same for Teammate 3.
- Teammates do NOT wait for approval before implementing. They declare and go.
- The only shared resource is the GPU. If multiple teammates need it, you sequence access.
- All other work runs fully in parallel.

CURRENT TASK: $ARGUMENTS

If no task is specified, read the project state and determine what needs doing next.

VERIFICATION - CRITICAL:
When an agent claims it is done, you DO NOT trust the claim. You independently verify:
1. Go into their worktree yourself
2. Build/run the code yourself to confirm it works
3. Verify evaluation code was not modified: sha256sum evaluation/evaluate.py evaluation/scoring.py and compare against evaluation/eval_lockfile.sha256
4. Check that outputs are real, not dummy data

Only after YOU confirm all checks pass do you declare a winner. If an agent's work crashes, segfaults, or returns fake results, it fails regardless of what it claims.

SCORING - ADAPTIVE:
- Each teammate trains for MAX 5 MINUTES (300 seconds wall clock). Hard limit.
- Primary metric: win rate against Cyborg AI
- But in early rounds nobody may win a game. So you MUST use the composite score from CLAUDE.md (survival time, damage dealt, damage taken, etc.)
- If the composite score still doesn't differentiate (e.g. all agents score ~0), YOU adjust the weights to find signal. Explain your adjustments in the commit message.
- The goal is always to pick the agent making the most PROGRESS toward winning, even if nobody wins yet.

GIT WORKFLOW:
- Each agent works on its own branch (created by the worktree).
- When an agent finishes, have it commit all its work to its branch with a descriptive message explaining its approach.
- Push ALL branches (winners and losers) so the full exploration history is preserved.
- Squash-merge only the winner's branch to main.
- Tag the merge: goteam/round-N-winner
- Tag losing branches: goteam/round-N-agent-X-loser
- Do NOT delete losing branches - they document explored avenues for future reference.
- Include interesting ideas from losers in the winner's merge commit message.

---
name: goteam
description: Launch agent team to tackle the current phase of work with coordinated non-overlapping exploration
user_invocable: true
---

You are the team lead. Read CLAUDE.md first to understand the project.

Each agent has its own git worktree - full independent copy of the repo. No file conflicts.

COORDINATION RULES:
- Spawn Agent 1, tell it to declare its approach then start coding immediately.
- Once you hear Agent 1's approach, spawn Agent 2 telling it what Agent 1 is doing so it picks a DIFFERENT approach. Agent 2 starts immediately after declaring.
- Same for Agent 3.
- Agents do NOT wait for approval before implementing. They declare and go.
- The only shared resource is the GPU. If multiple agents need it, you sequence access.
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

GIT WORKFLOW:
- Each agent works on its own branch (created by the worktree).
- When an agent finishes, have it commit all its work to its branch with a descriptive message explaining its approach.
- Push ALL branches (winners and losers) so the full exploration history is preserved.
- Squash-merge only the winner's branch to main.
- Tag the merge: goteam/round-N-winner
- Tag losing branches: goteam/round-N-agent-X-loser
- Do NOT delete losing branches - they document explored avenues for future reference.
- Include interesting ideas from losers in the winner's merge commit message.

# UQM Melee AI - Self-Evolving Training System

## Project Overview

AI that learns to play The Ur-Quan Masters Super Melee through evolutionary competition.
Three competing Claude agents modify training code each round; fastest to 80% win rate wins.

## Architecture

- **Model**: Frozen SigLIP-B/16 vision encoder + trainable actor-critic MLP heads
- **Game**: UQM-MegaMod built as `libmelee.so`, exposed via cffi as a Gymnasium environment
- **Training**: CleanRL-style single-file PPO (`training/ppo.py`)
- **Evaluation**: Immutable scoring measuring time-to-competency (wall-clock seconds to 80% win rate)

## For Competing Agents

You are in an evolutionary competition. Your goal: modify code so the AI reaches 80% win rate against the Cyborg AI as fast as possible.

### What you CAN modify

- `training/ppo.py` - PPO training loop (primary target)
- `training/agent.py` - model architecture
- `training/config.py` - hyperparameters
- `uqm_env/reward.py` - reward shaping

### What you CANNOT modify (hash-checked, disqualification)

- `evaluation/evaluate.py`
- `evaluation/scoring.py`
- `evaluation/eval_lockfile.sha256`
- `evolution/` directory

### Workflow

1. Read and understand the current training code
2. Design and implement improvements
3. Test with: `python -m training.run_training --total-timesteps 10000`
4. When done: `touch DONE`

### Key files

- `uqm_env/melee_env.py` - Gymnasium environment (observation: 240x320x3 RGB, action: Discrete(32))
- `uqm_env/melee_ffi.py` - cffi bindings to libmelee.so
- `training/ppo.py` - PPO implementation
- `training/agent.py` - SigLIP encoder + MLP heads

### Action space

5-bit mask = 32 discrete actions. Each bit maps to a button:
- Bit 0: LEFT
- Bit 1: RIGHT
- Bit 2: THRUST
- Bit 3: WEAPON (fire)
- Bit 4: SPECIAL

### Training Budget

Each agent gets exactly 5 MINUTES of training time. Not more. The hard wall-clock limit is 300 seconds. When time is up, training stops and the best checkpoint so far is evaluated.

### Scoring (composite, not just win rate)

The PRIMARY goal is win rate against Cyborg. But in early rounds, nobody may win a single game. So scoring uses a composite metric that the team lead evaluates:

```
SCORE = (win_rate * 100)                         # 0-100 points, primary
      + (avg_survival_frames / 1000)             # longer survival = better
      + (avg_damage_dealt / max_enemy_crew) * 10 # dealing damage = progress
      - (avg_damage_taken / max_own_crew) * 5    # taking less damage = better
      + (games_where_damage_dealt > 0) * 0.5     # even landing one hit counts
```

The team lead MAY adjust scoring weights between rounds if the current scoring isn't differentiating agents well enough. For example:
- If all agents die instantly, weight survival time heavily
- If agents survive but never fire, weight damage dealt
- If agents do damage but never win, weight win rate more
- The lead should explain any scoring adjustments in the commit message

This adaptive scoring ensures early rounds aren't just "everyone scored 0, pick randomly".

## Development

### Setup

```bash
./setup_env.sh
```

### Training

```bash
source .venv/bin/activate
python -m training.run_training
```

### Evaluation

```bash
python -m evaluation.evaluate --checkpoint checkpoints/best.pt
python -m evaluation.scoring --results outputs/results.json
```

## GPU

RTX 5060 Ti 16GB (SM_120). May need PyTorch nightly. SigLIP encoder is frozen so VRAM usage is ~2-4GB for training.

## ROBUSTNESS - READ THIS CAREFULLY

This system is designed to run UNATTENDED for hours or days. The user will walk away and come back. NOTHING may crash, hang, or require human intervention.

### Every process you launch MUST:

- Catch ALL exceptions at the top level and log them instead of crashing
- Use timeouts on every blocking operation (file I/O, network, subprocess, semaphores)
- Write progress to disk periodically so work is never lost if something dies
- Be restartable/resumable - if it dies and gets restarted, it picks up where it left off
- Never hold a lock indefinitely - use timeouts on mutexes, semaphores, file locks
- Log to files, not just stdout - stdout scrolls away in tmux

### C code (libmelee.so) MUST:

- Never segfault - validate all pointers before dereferencing
- Use signal handlers for SIGSEGV/SIGABRT that log the crash and exit cleanly
- Have watchdog timeouts on the game loop - if DoBattle hangs, kill and restart
- Handle SDL initialization failure gracefully (return error code, don't abort())
- Free all resources in melee_close() even after partial initialization

### Python training code MUST:

- Wrap the entire training loop in try/except that saves the best checkpoint before dying
- Checkpoint every N steps so a crash only loses minutes, not hours
- Handle GPU OOM gracefully - catch RuntimeError, reduce batch size, retry
- Handle libmelee.so crashes - if the C library segfaults, restart the environment
- Use subprocess isolation for the game environment if needed (multiprocessing with spawn)
- Write results.json incrementally, not just at the end

### The evolution orchestrator MUST:

- Handle agent timeouts (agent hangs forever, never touches DONE)
- Handle training crashes (agent's code crashes mid-training)
- Handle partial results (training died but left a checkpoint - evaluate what exists)
- Never leave orphaned processes - track PIDs and kill everything on cleanup
- Be idempotent - running it again after a crash doesn't corrupt state
- Log everything to evolution/logs/ with timestamps

### Teammates: when the team lead tells you to build something, YOU are responsible for making it robust. Do not write code that works in the happy path only. Handle every failure mode. The user will not be there to fix things.

## Rules

- No mocks. Actually train.
- Train from scratch each round (no loading previous weights).
- Evaluation integrity is hash-checked.
- One GPU shared across agents (train sequentially).
- NEVER write code that can hang, crash, or lose progress unrecoverably.

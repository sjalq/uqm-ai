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

### Metric

TIME-TO-COMPETENCY: wall-clock seconds until 80% win rate vs Cyborg over rolling window of 3 evaluations.

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

## Rules

- No mocks. Actually train.
- Train from scratch each round (no loading previous weights).
- Evaluation integrity is hash-checked.
- One GPU shared across agents (train sequentially).

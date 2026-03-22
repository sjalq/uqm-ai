"""
Training configuration - all hyperparameters in one place.

This file is MODIFIABLE by competing agents.
Agent 3 - Round 1: Training loop optimization + curriculum learning.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Environment
    ship_p1: int = 5                # Earthling Cruiser
    ship_p2: int = 5                # Earthling Cruiser
    p2_cyborg: bool = True          # Opponent is built-in AI
    frame_skip: int = 4             # Game frames per agent step
    num_envs: int = 8               # Parallel environments

    # Model architecture
    encoder_name: str = "ViT-B-16-SigLIP"
    encoder_pretrained: str = "webli"
    encoder_dim: int = 768          # SigLIP-B/16 output dimension
    hidden_dim: int = 512           # Larger MLP for more capacity
    action_dim: int = 32            # 5-bit action space

    # PPO hyperparameters - optimized for throughput
    learning_rate: float = 7e-4     # Higher LR for faster initial learning
    num_steps: int = 64             # Shorter rollouts = more frequent updates
    num_minibatches: int = 4        # Fewer minibatches = larger effective batch
    update_epochs: int = 3          # Fewer epochs = more env steps per wall-clock second
    gamma: float = 0.97             # Lower discount for short-horizon melee fights
    gae_lambda: float = 0.92        # Slightly lower GAE lambda
    clip_coef: float = 0.2          # PPO clip coefficient
    ent_coef: float = 0.05          # Higher entropy for exploration
    ent_coef_final: float = 0.005   # Anneal entropy down over training
    vf_coef: float = 0.5            # Value function loss coefficient
    max_grad_norm: float = 0.5      # Gradient clipping

    # Training duration
    total_timesteps: int = 1_000_000
    wall_clock_limit: float = 275.0  # Hard stop 25s before 5min limit
    eval_interval: int = 8_000      # Evaluate frequently but not too often
    eval_episodes: int = 8          # Fewer eval episodes = less time on eval
    checkpoint_interval: int = 20_000

    # Evaluation threshold
    target_win_rate: float = 0.8    # Win rate to consider "competent"

    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Curriculum learning: action space reduction
    # Phase 1 (0 to curriculum_phase1_steps): restricted to combat-useful actions
    # Phase 2 (after): full 32-action space
    curriculum_phase1_steps: int = 80_000
    # Useful combat actions (indices into Discrete(32)):
    # 0=idle, 1=left, 2=right, 4=thrust, 8=fire, 12=thrust+fire,
    # 5=left+thrust, 6=right+thrust, 9=left+fire, 10=right+fire,
    # 13=left+thrust+fire, 14=right+thrust+fire
    curriculum_actions_phase1: str = "0,1,2,4,5,6,8,9,10,12,13,14"
